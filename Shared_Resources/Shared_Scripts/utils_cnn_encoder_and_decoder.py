'''
Implementation of a CNN encoder and a CNN decoder


Encoder:
The encoder conists of a sequence of convolutions and downsampling operations. At each scale feature maps can be exported.


Decoder:
The Deocder consists of a sequence of upsampling and convolutio layers. At each scale it allows for extra feature maps (e.g. transferred from an encoder)


'''

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import math

import numpy as np

#----- U-Net architecture that combines the CNNEncoder and CNNDeocder to a model that maps image to image ----



#--- Architecture of Encoder and Decoder ------

class CNNDecoder(torch.nn.Module):
    def __init__(self,out_ch=3,output_activation = None,
                dimension_specification={0: {'width': 24, 'depth': 0, 'width_skip': 24},1:{'width': 12, 'depth': 1, 'width_skip': 6}},verbose=False):
        super(CNNDecoder, self).__init__()
        '''
        Upgrade, i.e. generalization of DecoderUnet.
        The architecture is parameterized in more detail. Instead of a simple complexity parameter
        we have to specify exactly the width at each resolution level as well as the width of the 
        tensor that is received from the encoder.

        Parameters:
            in_ch: Number of input channels
            dimension_specification: Dictionary that spacifies for each scale (or depth) the width and to be passed to the next scale, the depth on the scale level and the width to be passt to the decoder
                                    Example: We want an encoder that has at scale s; x channels, is y deep and passes z channels to the decoder. 
                                            This corresponds to an item: dimension_specification[s] = {'width': x, 'depth': y, 'width_skip': z}
        '''
        self.dimension_specification = dimension_specification
        self.out_ch=out_ch
        
        self.network_depth = len(dimension_specification)
        self.check_if_dimension_specification_is_valid()
        

        if output_activation=='sigmoid':
            self.output_activation = nn.Sigmoid() 
        elif output_activation=='relu': 
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh': 
            self.output_activation = nn.Tanh()
        elif output_activation is None: 
            self.output_activation = None
        else:
            raise ValueError('This output activation is not implemented: ',output_activation)
        
        self.decoder_layers = self.get_decoder_layers()
        self.decoder_head = self.get_decoder_head()

    def check_if_dimension_specification_is_valid(self):
        for i in range(self.network_depth):
            assert i in self.dimension_specification.keys(),'The keys of width_specification specify the scale level, integer numbered starting from 0'
            assert len(self.dimension_specification[i])==3,'Each scale level must be specified by 3 dimension keys: width, depth,width_skp'
            assert self.dimension_specification[i]['width']>=1,'the miminum width is 1 (otherwise we have no information flow)'
            assert self.dimension_specification[i]['depth']>=0, 'the minimum depth is 0, where we define zero depth as having only a layer that changes the scale'
            assert self.dimension_specification[i]['width_skip']>=0, '0 is valid for no skip connection, but negative values are not meaningful'

        self.dimension_specification[self.network_depth-1]['width_skip']>=1,'The deepest (lowest resolution) layer must have skip connection,i.e. an output at that last scale, otherwise this scale is useless.'
    


    def get_decoder_layers(self):
        
        decoder_layers = nn.ModuleDict()
        
        for i in range(self.network_depth-1, -1, -1):#counts backwards, because we start with the lowest scale (highest scale number)
            
            scale_name = 'scale_'+str(i)

            dimensions = self.dimension_specification[i]
            
            up_layer_i=[]
            
            #determining number of input channels from the current scale
            if i==self.network_depth-1:
                #we only have the skip connection input from the encoder at the lowest levevl
                n_channels_in = dimensions['width_skip']
            else:
                n_channels_in = dimensions['width_skip'] + dimensions['width']#additionally we expect the lower scale input to have dimensions['width']

                
            #projecting to the width of the current level
            up_layer_i.append(ConvLayerEqualBlock(n_channels_in,dimensions['width'],kernel_size=1))#kernelsize unclear, but it is an expensive projection (wide input)

            #adding depth to that scale
            for k in range(dimensions['depth']):
                up_layer_i.append(ResidualBlock2(dimensions['width'], kernel_size = 1))
            
            #upsampling if we are not in the highest scale
            if i>0:
                n_channels_out = self.dimension_specification[i-1]['width']

                #upsampling
                up_layer_i.append(UpsampleConvLayer(dimensions['width'], n_channels_out, kernel_size=3, upsample=2))

            decoder_layers[scale_name] = nn.Sequential(*up_layer_i)
        
        return decoder_layers


    def get_decoder_head(self,kernel_size=9, stride=1):
        return ConvLayerEqual(self.dimension_specification[0]['width'], self.out_ch, kernel_size=kernel_size)


    
    def apply_decoder(self,skip_connections):
        
        #apply decoder layers
        #for scale_name,layer_net in self.decoder_layers.items():
        for i in range(self.network_depth-1, -1, -1):#counts backwards, because we start with the lowest scale (highest scale number)
            scale_name = 'scale_'+str(i)
            layer_net = self.decoder_layers[scale_name]
            

            if i == self.network_depth-1:
                z_skip = skip_connections[scale_name]
                #we need to have guaranteed that the deepest level has a skip connection
                #otherwise the encoder encodes the deepest scale for nothing. Here we intend to catch that mistake
                assert scale_name in skip_connections.keys(),'the deepest level should have a skip connection'
                
                z = layer_net(z_skip)#we keep updating z
            else:

                if scale_name in skip_connections.keys():
                    #if there is a skip-connection, concatenate
                    z_skip = skip_connections[scale_name]
                    z=torch.cat([z, z_skip], dim=1)

                z = layer_net(z)#dimension of the provided skip-connections need to be compatible with the architecture specifications
        return z

    def undo_padding(self,y,pad):
        if pad is not None:
            #undo the initial padding
            s = y.size()
            y = y[:,:,pad[2]:s[2]-pad[3],pad[0]:s[3]-pad[1]]

        return y
    
    def forward(self,encoder_out,pad=None):
        '''
        Optionally can get initial padding
        '''

        pad = encoder_out['initial_padding']
        skip_connections = encoder_out['skip_connections']


        y = self.apply_decoder(skip_connections)

        y = self.decoder_head(y)

        y = self.undo_padding(y,pad)

            
        if self.output_activation is not None:
            y = self.output_activation(y)
            
        return y


class CNNEncoder(torch.nn.Module):
    def __init__(self,in_ch=3,dimension_specification={0: {'width': 12, 'depth': 1, 'width_skip': 6},1:{'width': 24, 'depth': 1, 'width_skip': 24}},
                                    pooling_mode='max',verbose=False):
        super(CNNEncoder, self).__init__()
        '''
        Upgrade, i.e. generalization of EncoderUnet.
        The architecture is parameterized in more detail. Instead of a simple complexity parameter
        we have to specify exactly the width at each resolution level as well as the width of the 
        tensor that is passed to the decoder at each resolution level.

        Parameters:
            in_ch: Number of input channels
            dimension_specification: Dictionary that spacifies for each scale (or depth) the width and to be passed to the next scale, the depth on the scale level and the width to be passt to the decoder
                                    Example: We want an encoder that has at scale s; x channels, is y deep and passes z channels to the decoder. 
                                            This corresponds to an item: dimension_specification[s] = {'width': x, 'depth': y, 'width_skip': z}
        '''
        self.in_ch=in_ch
        self.n_down=len(dimension_specification)-1#the number of downsampling steps equals the total number of scales minus one.
        self.down_sample_factor = 2**self.n_down
        self.pooling_mode = pooling_mode
        self.dimension_specification = dimension_specification

        
        self.network_depth = len(dimension_specification)
        self.check_if_dimension_specification_is_valid()
    

        if verbose:
            print('dimension_specification: ',dimension_specification)
        
        self.scale_layers, self.skip_layers = self.get_encoder_layers()

    def check_if_dimension_specification_is_valid(self):
        for i in range(self.network_depth):
            assert i in self.dimension_specification.keys(),'The keys of width_specification specify the scale level, integer numbered starting from 0'
            assert len(self.dimension_specification[i])==3,'Each scale level must be specified by 3 dimension keys: width, depth,width_skp'
            assert self.dimension_specification[i]['width']>=1,'the miminum width is 1 (otherwise we have no information flow)'
            assert self.dimension_specification[i]['depth']>=0, 'the minimum depth is 0, where we define zero depth as having only a layer that changes the scale'
            assert self.dimension_specification[i]['width_skip']>=0, '0 is valid for no skip connection, but negative values are not meaningful'

        self.dimension_specification[self.network_depth-1]['width_skip']>=1,'The last layer must have skip connection,i.e. an output at that last scale, otherwise this scale is useless.'

        
    def get_encoder_layers(self):
        
        scale_layers = nn.ModuleDict()
        skip_layers = nn.ModuleDict()

        
        #Layers
        for i in range(self.network_depth):
            scale_name = 'scale_'+str(i)
            dimensions = self.dimension_specification[i]

            
            scale_layer_i=[]

            if i==0:
                #first layer we have to tread slightly extra, because of the initial batchnorm and the fact, that we don't start with down-sampling
                scale_layer_i.append(nn.BatchNorm2d(self.in_ch,affine=False,momentum=None)),#affine=False to ensure (mean,std) = (0,1), momentum=None is cumulative average (simple average)
                scale_layer_i.append(ConvLayerEqualBlock(self.in_ch,dimensions['width'],kernel_size=9,momentum=0.05)),#small momentum momentum=0.05, because we don't expect much fluctuation in the first layer
                #adding depth at that scale
                for k in range(dimensions['depth']):
                    scale_layer_i.append(ResidualBlock2(dimensions['width'], kernel_size = 1))
            else:
                #downsampling
                scale_layer_i.append(DownsampleConvLayer(self.dimension_specification[i-1]['width'], dimensions['width'], kernel_size=3,
                                     pooling_mode=self.pooling_mode))
                #adding depth at that scale
                for k in range(dimensions['depth']):
                    scale_layer_i.append(ResidualBlock2(dimensions['width'], kernel_size = 1))


            #post-processing the skip-connection (e.g. to reduce width for memory efficiency)
            if dimensions['width_skip']>0:
                skip_layers[scale_name] = ConvLayerEqualBlock(dimensions['width'],dimensions['width_skip'],kernel_size=1)


            scale_layers[scale_name] = nn.Sequential(*scale_layer_i)

        
        return scale_layers, skip_layers
    
    def apply_preprocessing(self,x):
        x, pad = self.add_padding_if_needed(x)#add initial padding if needed
        return x, pad

    def apply_encoder(self,x,pad):
        
        skip_connections={}

        for i in range(self.network_depth):
            layer_name = 'scale_'+str(i)
            scale_layer = self.scale_layers[layer_name]

            x = scale_layer(x)#note: x is redefined in each layer

            if layer_name in self.skip_layers.keys():#if there is a skip connection at this scale
                x_skip = self.skip_layers[layer_name](x)#store a representation for this scale
                skip_connections[layer_name] = x_skip
        
        out = {'skip_connections': skip_connections,
               'initial_padding': pad}
        
        return out
        
        
    def forward(self, x):
        x, pad = self.apply_preprocessing(x)
        out = self.apply_encoder(x,pad)   
        return out
    
    
    def add_padding_if_needed(self,x):
        '''
        Make sure the input image is of size (n * self.down_sample_factor, m * self.down_sample_factor); n,m >= 2
        '''
        d2 = x.size(2) % self.down_sample_factor
        d3 = x.size(3) % self.down_sample_factor
        
        padding_size_2 = ((self.down_sample_factor - d2) % self.down_sample_factor)
        padding_size_3 = ((self.down_sample_factor - d3) % self.down_sample_factor)
        
        reflection_padding_20 = padding_size_2 // 2#floor
        reflection_padding_21 = padding_size_2 - reflection_padding_20#s.t. reflection_padding_21>=reflection_padding_20
        reflection_padding_30 = padding_size_3 // 2
        reflection_padding_31 = padding_size_3 - reflection_padding_30#s.t. reflection_padding_31>=reflection_padding_30
        
        pad = (reflection_padding_30,reflection_padding_31,
               reflection_padding_20,reflection_padding_21)

        return nn.functional.pad(x, pad, mode='reflect'),pad


def get_classname_to_idx(model):
    name_to_idx_semantic = {name: idx for idx,name in enumerate(model.class_names_semantic)}
    name_to_idx_focus = {name: idx for idx,name in enumerate(model.class_names_focus)}
    name_to_idx_object = {name: idx for idx,name in enumerate(model.class_names_object)}
    mask_type_to_name_to_idx = {'semantic': name_to_idx_semantic,
                               'focus': name_to_idx_focus,
                               'object': name_to_idx_object}
    return mask_type_to_name_to_idx



#---------------- Basic building blocks for CNNs ----------------------------------------


class ConvLayerEqual(torch.nn.Module):
    #keeps the size (W,H) equal
    def __init__(self, in_channels, out_channels, kernel_size,bias=True):
        super(ConvLayerEqual, self).__init__()
        'I think with pytorch it should now be possible to do equal padding directly with nn.Conv2d by setting padding="same", but I have to check'
        #padding succh that the image dimension stays the same fo rthe given kernelsize
        reflection_padding_left = (kernel_size-1) // 2
        reflection_padding_right = (kernel_size-1) - reflection_padding_left
        self.reflection_pad = nn.ReflectionPad2d(
            (reflection_padding_left,reflection_padding_right,reflection_padding_left,reflection_padding_right))#(left,right,top,bottom)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,bias=bias)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class ConvLayerEqualBlock(torch.nn.Module):
    #This takes the ConvLayerEqual and adds a relu and a bn
    def __init__(self, in_channels, out_channels, kernel_size,bias=True,momentum=0.9):
        super(ConvLayerEqualBlock, self).__init__()
        self.conv_layer_equal = ConvLayerEqual(in_channels, out_channels, kernel_size=kernel_size,bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels,momentum=momentum)

    def forward(self,x):
        x = self.conv_layer_equal(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d=ConvLayerEqual(in_channels, out_channels, kernel_size=kernel_size,bias=False)# bias=False, since followed by bn
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.conv2d(x)
        x = self.relu(self.bn(x))
        return x

class DownsampleConvLayer(torch.nn.Module):
    '''
    Downsamples by a factor of 2
    '''
    def __init__(self, in_channels, out_channels, kernel_size,bias=True,pooling_mode='max'):
        super(DownsampleConvLayer, self).__init__()
        self.conv=ConvLayerEqual(in_channels, out_channels, kernel_size=kernel_size,bias=bias)#can not set bias=False, because maxpool() and relu() are not invariant to additive constant
        self.relu = torch.nn.ReLU()
        if pooling_mode == 'max':
            self.pooling = nn.MaxPool2d(2, stride=2)
        elif pooling_mode == 'avg':
            self.pooling = nn.AvgPool2d(2, stride=2)
        else:
            raise ValueError('Pooling mode unknown: ',pooling_mode)
        self.bn=nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.pooling(out)
        out = self.relu(out)
        out = self.bn(out)
        return out



class ResidualBlock2(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    #remove biase whenever it is followed by batchnorm, because it is redundant. The BN output is invariant under additive constants.
    """

    def __init__(self, channels,kernel_size=3,bias=True):
        super(ResidualBlock2, self).__init__()
        self.conv1 = ConvLayerEqual(channels, channels, kernel_size=kernel_size, bias=bias)#I should set bias=False, since followed by bn
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.conv2 = ConvLayerEqual(channels, channels, kernel_size=kernel_size, bias=bias)#I should set bias=False, since followed by bn
        self.bn2 = nn.BatchNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return out



#---- Image processing functions -----


def apply_geometric_transformation(x,angle,flip=False):
    #applys a rotation by angle and a reflexion if flip.
    #note that angles should be integerfactors of 90 in order to not have border effects
    if angle!=0:
        if angle%90 == 0:
            #integer factor of a 90 degrees rotation -> we use the rotation without zeropadding outside of the square
            n_90_degrees = int(angle/90)
            x = torch.rot90(x, n_90_degrees , dims=[2,3])
        else:
            x=TF.rotate(x, angle)
    if flip:
        x=TF.hflip(x)#horizontal flip



    return x


