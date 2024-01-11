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

from typing import Tuple, Optional, Dict, List,TypedDict

import numpy as np

#----- U-Net architecture that combines the CNNEncoder and CNNDeocder to a model that maps image to image ----



#--- Architecture of Encoder and Decoder ------

class CNNDecoder(torch.nn.Module):
    """
    Decoder part of the U-Net architecture.
    """

    __slots__ = ('dimension_specification','out_ch','network_depth','output_activation','decoder_layers','decoder_head')

    def __init__(self,out_ch: int, output_activation: str = 'identity',
                dimension_specification: Dict[int, Dict[str,int]] = {0: {'width': 24, 'depth': 0, 'kernel_size': 3, 'width_skip': 6},
                                                                                    1:{'width': 12, 'depth': 1, 'kernel_size': 2, 'width_skip': 24}},
                verbose: bool = False,efficiency_optimized: bool = False):
        super().__init__()
        '''
        :param out_ch: Number of output channels
        :param output_activation: Activation function to be used in the output layer. Should be one of 'sigmoid', 'relu', 'tanh', or None.
        :param dimension_specification: Dictionary that spacifies for each scale (or depth) the width and to be passed to the next scale, the depth on the scale level and the width to be passt to the decoder
                                    Example: We want an encoder that has at scale s; x channels, is y deep and passes z channels to the decoder. CNNs with kernelsize k.
                                            This corresponds to an item: dimension_specification[s] = {'width': x, 'depth': y, 'width_skip': z, 'kernel_size': k}
        :param efficiency_optimized: Several time and memory efficient replacements for submodules 
        '''
        self.dimension_specification = dimension_specification
        self.out_ch=out_ch
        
        self.network_depth = len(dimension_specification)
        self.check_if_dimension_specification_is_valid()

        self.efficiency_optimized = efficiency_optimized
        

        if output_activation=='sigmoid':
            self.output_activation = nn.Sigmoid() 
        elif output_activation=='relu': 
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh': 
            self.output_activation = nn.Tanh()
        elif output_activation == 'identity': 
            self.output_activation = None
        else:
            raise ValueError('This output activation is not implemented: ',output_activation)
        
        self.decoder_layers = self.get_decoder_layers()
        self.decoder_head = self.get_decoder_head()

    def check_if_dimension_specification_is_valid(self):
        """
        Check if the dimension specification is valid

        :raises assert error: If the dimensions fo the self.dimension_specification do not satisfy the necessary constraints
        """
        for i in range(self.network_depth):
            assert i in self.dimension_specification.keys(),'The keys of width_specification specify the scale level, integer numbered starting from 0'
            assert len(self.dimension_specification[i])==4,'Each scale level must be specified by 4 dimension keys: width, depth, kernel_size, width_skp'
            assert self.dimension_specification[i]['width']>=1,'the miminum width is 1 (otherwise we have no information flow)'
            assert self.dimension_specification[i]['depth']>=0, 'the minimum depth is 0, where we define zero depth as having only a layer that changes the scale'
            assert self.dimension_specification[i]['kernel_size']>=1, 'the minimum kernel_size is 1 for each scale'
            assert self.dimension_specification[i]['width_skip']>=0, '0 is valid for no skip connection, but negative values are not meaningful'

        self.dimension_specification[self.network_depth-1]['width_skip']>=1,'The deepest (lowest resolution) layer must have skip connection,i.e. an output at that last scale, otherwise this scale is useless.'
    


    def get_decoder_layers(self) -> nn.ModuleDict:
        """
        Computes and returns the decoder model layers at each scale

        :returns: A nn.ModuleDict(), with keys the scale as 'scale_'+str(layer_number) and values the pytorch module at that scale.
        """
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
            up_layer_i.append(ConvLayerEqualBlock(n_channels_in,dimensions['width'],kernel_size=dimensions['kernel_size'],efficiency_optimized = self.efficiency_optimized))

            #adding depth to that scale
            for k in range(dimensions['depth']):
                if self.efficiency_optimized:
                    block = ConvLayerEqualBlock(dimensions['width'],dimensions['width'], kernel_size = dimensions['kernel_size'],efficiency_optimized = self.efficiency_optimized)
                else:
                    block = ResidualBlock(dimensions['width'], kernel_size = dimensions['kernel_size'])
                    
                up_layer_i.append(block)
            
            #upsampling if we are not in the highest scale
            if i>0:
                n_channels_out = self.dimension_specification[i-1]['width']

                #upsampling
                up_layer_i.append(UpsampleConvLayer(dimensions['width'], n_channels_out, kernel_size=dimensions['kernel_size'],efficiency_optimized = self.efficiency_optimized))

            decoder_layers[scale_name] = nn.Sequential(*up_layer_i)
        
        return decoder_layers


    def get_decoder_head(self) -> nn.Module:
        """
        Defines the output layer of the decoder

        :returns: The pytorch module of the last layer that maps to the decoder output.
        """
        return ConvLayerEqual(self.dimension_specification[0]['width'], self.out_ch, kernel_size=self.dimension_specification[0]['kernel_size'],efficiency_optimized = self.efficiency_optimized)


    
    def apply_decoder(self,skip_connections: dict) -> torch.Tensor:
        """
        Applies decoder layers to the passed skip_connections from the encoder
        
        :params skip_connections: A dictionary that maps each scale_name to the corresponding pytorch tensor received from the encoder at that level

        :returns: Decoder representation layers right before the output layer.

        """

        for i in range(self.network_depth-1, -1, -1):#counts backwards, because we start with the lowest scale (highest scale number)
            scale_name = 'scale_'+str(i)
            layer_net = self.decoder_layers[scale_name]
            

            if i == self.network_depth-1:
                z_skip = skip_connections[scale_name]
                #we need to have guaranteed that the deepest level has a skip connection
                #otherwise the encoder encodes the deepest scale for nothing. Here we intend to catch that mistake
                assert scale_name in skip_connections.keys(),'the deepest level should have a skip connection'
                
                tensor_latent = layer_net(z_skip)#we keep updating tensor_latent
            else:
                if scale_name in skip_connections.keys():
                    #if there is a skip-connection, concatenate
                    z_skip = skip_connections[scale_name]
                    tensor_latent=torch.cat([tensor_latent, z_skip], dim=1)

                tensor_latent = layer_net(tensor_latent)#dimension of the provided skip-connections need to be compatible with the architecture specifications
        return tensor_latent

    def undo_padding(self,output_tensor: torch.Tensor,pad: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        The input image might have needed some padding to match the geometric constraint. 
        This function clips away the padded pixels, such that the original image shape is restored.

        :param output_tensor: The output tensor of the decoder head
        :param pad: A tuple containing four integers that describe the padding applied to each side
            of the input tensor. The order is (padding_top, padding_bottom, padding_left, padding_right).

        :raises ValueError: If `pad` is not a 4-element tuple.

        :return: The tensor after removing the padding, restoring it to its original dimensions
            before padding was applied.

        """

        if not isinstance(pad, tuple) or len(pad) != 4:
            raise ValueError("pad must be a 4-element tuple")

        if pad is not None:
            #undo the initial padding
            size = output_tensor.size()
            output_tensor = output_tensor[:,:,pad[2]:size[2]-pad[3],pad[0]:size[3]-pad[1]]

        return output_tensor
    
    def forward(self,encoder_out: dict) -> torch.Tensor:
        '''
        Applies the CNNencoder given the skip_connections of the decoder

        :param encoder_out: A dictionary that contains the initial padding and the skip_connections (as pytorch tensors) from the encoder

        :return: Output tensor after applying the model.
        '''

        pad = encoder_out['initial_padding']
        skip_connections = encoder_out['skip_connections']

        tensor = self.apply_decoder(skip_connections)
        tensor = self.decoder_head(tensor)
        tensor = self.undo_padding(tensor,pad)

        if self.output_activation is not None:
            tensor = self.output_activation(tensor)
            
        return tensor



class CNNEncoder(torch.nn.Module):
    """
    Encoder class for U-Net architecture.
    """

    __slots__ = ('in_ch','n_down','down_sample_factor','dimension_specification','efficiency_optimized','network_depth','scale_layers','scale_name_to_n_skip_channels')


    def __init__(self,in_ch: int,dimension_specification: Dict[int, Dict[str,int]] = {0: {'width': 12, 'depth': 1, 'kernel_size': 3,'width_skip': 6},
                                                                                        1:{'width': 24, 'depth': 1, 'kernel_size': 2,'width_skip': 24}},
                 verbose: bool = False, efficiency_optimized: bool = False):
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
            efficiency_optimized: ResidualBlock are replaced with ConvLayerEqualBlock, because they are computationally much more heavy (but have better gradient flow) 
        '''

        self.in_ch=in_ch
        self.n_down=len(dimension_specification)-1#the number of downsampling steps equals the total number of scales minus one.
        self.down_sample_factor = 2**self.n_down
        self.dimension_specification = dimension_specification

        self.efficiency_optimized = efficiency_optimized

        self.network_depth = len(dimension_specification)
        self.check_if_dimension_specification_is_valid()
    

        if verbose:
            print('dimension_specification: ',dimension_specification)
        
        self.scale_layers, self.scale_name_to_n_skip_channels = self.get_encoder_layers()


    def check_if_dimension_specification_is_valid(self):
        """
        Check if the dimension specification is valid

        :raises assert error: If the dimensions fo the self.dimension_specification do not satisfy the necessary constraints
        """
        for i in range(self.network_depth):
            assert i in self.dimension_specification.keys(),'The keys of width_specification specify the scale level, integer numbered starting from 0'
            assert len(self.dimension_specification[i])==4,'Each scale level must be specified by 3 dimension keys: width, depth, kernel_size, width_skp'
            assert self.dimension_specification[i]['width']>=1,'the miminum width is 1 (otherwise we have no information flow)'
            assert self.dimension_specification[i]['depth']>=0, 'the minimum depth is 0, where we define zero depth as having only a layer that changes the scale'
            assert self.dimension_specification[i]['kernel_size']>=1, 'the minimum kernel_size is 1 for each scale'
            assert self.dimension_specification[i]['width']>=self.dimension_specification[i]['width_skip']>=0, '0 is valid for no skip connection, but negative values are not meaningful. We can not select more channels than the width.'

        self.dimension_specification[self.network_depth-1]['width_skip']>=1,'The last layer must have skip connection,i.e. an output at that last scale, otherwise this scale is useless.'

        
    def get_encoder_layers(self) -> nn.ModuleDict:
        """
        Get the encoder layers.

        :return: Encoder layers.
        """
        scale_layers = nn.ModuleDict()
        scale_name_to_n_skip_channels = {}

        #Layers
        for i in range(self.network_depth):
            scale_name = 'scale_'+str(i)
            dimensions = self.dimension_specification[i]

            scale_layer_i=[]

            if i==0:
                #first layer we have to tread slightly extra, because of the initial batchnorm and the fact, that we don't start with down-sampling
                scale_layer_i.append(nn.BatchNorm2d(self.in_ch,affine=False,momentum=None)),#affine=False to ensure (mean,std) = (0,1), momentum=None is cumulative average (simple average)
                scale_layer_i.append(ConvLayerEqualBlock(self.in_ch,dimensions['width'],kernel_size=dimensions['kernel_size'],momentum=0.05,efficiency_optimized = self.efficiency_optimized)),#small momentum momentum=0.05, because we don't expect much fluctuation in the first layer
            else:
                #downsampling
                scale_layer_i.append(DownsampleConvLayer(self.dimension_specification[i-1]['width'], dimensions['width'], 
                                        kernel_size=self.dimension_specification[i-1]['kernel_size'],efficiency_optimized = self.efficiency_optimized))
                
            #adding depth at that scale
            for k in range(dimensions['depth']):
                if self.efficiency_optimized:
                    block =  ConvLayerEqualBlock(dimensions['width'],dimensions['width'],kernel_size=dimensions['kernel_size'],efficiency_optimized = self.efficiency_optimized)
                else:
                    block = ResidualBlock(dimensions['width'], kernel_size = dimensions['kernel_size'])
                scale_layer_i.append(block)


            #store how many channels we want to skip at that scale
            if dimensions['width_skip']>0:
                scale_name_to_n_skip_channels[scale_name] = dimensions['width_skip']

            scale_layers[scale_name] = nn.Sequential(*scale_layer_i)

        
        return scale_layers,scale_name_to_n_skip_channels#, skip_layers
    

    def apply_encoder(self,input_tensor: torch.Tensor) -> Dict[str,torch.Tensor]:
        """
        Apply encoder to the input.

        :param input_tensor: Input tensor.
        :return: Encoded output.
        """
        
        skip_connections={}

        for i in range(self.network_depth):
            layer_name = 'scale_'+str(i)
            scale_layer = self.scale_layers[layer_name]

            input_tensor = scale_layer(input_tensor)#note: input_tensor is redefined in each layer

            if layer_name in self.scale_name_to_n_skip_channels.keys():#if there is a skip connection at this scale
                n_skip_channels = self.scale_name_to_n_skip_channels[layer_name]
                x_skip = input_tensor[:,:n_skip_channels,:,:]#skip the first n_skip channels 
                skip_connections[layer_name] = x_skip
    
        return skip_connections
        
        
    def forward(self,input_tensor: torch.Tensor)  -> Dict[str, Dict[str,torch.Tensor]]:
        """
        Forward pass of the model.

        :param input_tensor: Input tensor.
        :return: Output tensor of the model
        """

        input_tensor, pad = self.add_padding_if_needed(input_tensor)#add initial padding if needed
        
        skip_connections = self.apply_encoder(input_tensor)  
        
        out = {'skip_connections': skip_connections,
               'initial_padding': pad}

        return out
    
    
    def add_padding_if_needed(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Add padding to the input tensor if its size is not divisible by
        the downsample factor.

        :param input_tensor: Input tensor.
        :return: Padded input tensor and padding sizes.
        """
        remainder_width = input_tensor.size(2) % self.down_sample_factor
        remainder_height = input_tensor.size(3) % self.down_sample_factor

        total_padding_width = ((self.down_sample_factor - remainder_width) % self.down_sample_factor)
        total_padding_height = ((self.down_sample_factor - remainder_height) % self.down_sample_factor)

        padding_left = total_padding_width // 2  # floor
        padding_right = total_padding_width - padding_left
        padding_top = total_padding_height // 2
        padding_bottom = total_padding_height - padding_top

        pad = (padding_top, padding_bottom, padding_left, padding_right)

        # mode = 'reflect' seems for some reason faster than mode = 'constant' from my tests. Therefore for self.efficiency_optimized==True we still keep mode='reflect'

        return nn.functional.pad(input_tensor, pad, mode='reflect'), pad


#---------------- Basic building blocks for CNNs ----------------------------------------

class ConvLayerEqual(nn.Module):
    """
    A 2D Convolutional Layer with equal reflection padding on both sides.

    Attributes:
        conv2d (torch.nn.Conv2d): 2D convolutional layer.
    """

    __slots__ = ('conv2d')

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True ,efficiency_optimized: bool = False):
        """
        Initialize ConvLayerEqual with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param bias: If True, adds a learnable bias to the output. Optional, defaults to True.
        :param efficiency_optimized: If True, adds zero padding instead of reflection padding, which is slightly faster. Optional, defaults to False.
        """
        super().__init__()

        if efficiency_optimized:
            padding_mode = 'zeros'
        else:
            padding_mode = 'reflect'

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,bias=bias,padding = 'same',padding_mode = padding_mode)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        :param input_tensor: Input tensor of type torch.Tensor.
        :return: Output tensor of type torch.Tensor.
        """
        input_tensor = self.conv2d(input_tensor)
        return input_tensor


class ConvLayerEqualBlock(torch.nn.Module):
    """UpsampleConvLayer
    Applies a shape conserving convolution followed by batch-normalization and relu
    """

    __slots__ = ('conv_layer_equal','relu','batch_norm')

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, momentum: float = 0.9, efficiency_optimized: bool = False):
        """
        Initialize ConvLayerEqualBlock with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param momentum: Momentum of the batch normalization.
        :param efficiency_optimized: If True, adds zero padding instead of reflection padding, which is slightly faster. Optional, defaults to False.
        """
        super().__init__()
        self.conv_layer_equal = ConvLayerEqual(in_channels, out_channels, kernel_size=kernel_size, bias = False,
                                                    efficiency_optimized=efficiency_optimized)# bias=False, since followed by bn
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels,momentum=momentum)

    def forward(self,input_tensor):
        """
        Defines the forward pass of the module.

        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        output_tensor = self.conv_layer_equal(input_tensor)
        output_tensor = self.relu(self.batch_norm(output_tensor))#BN first
        return output_tensor


def ICNR(tensor, upscale_factor=2):
    """
    Reference: https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
    :param tensor: the 2-dimensional Tensor or more
    :param upscale_factor: The upscaling factor used in the nn.PixelShuffle (needed to know the subkernel size)
    """
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = nn.init.normal_(sub_kernel, mean=0.0, std=0.02)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution.
    """

    __slots__ = ('conv_block','mode','pixel_shuffle')

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 mode: str = 'pixel_shuffle', efficiency_optimized: bool = False) -> None:
        """
        :param in_channels: Number of channels in the input image.
        :param out_channels: Number of channels produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        :param mode: The upsampling mode, can be one of 'nearest','bilinear','pixel_shuffle'
        :param efficiency_optimized: If True, the ConvLayerEqualBlock will use zero padding instead of reflection padding
        :raise: Assert error if the mode is unknown
        """
        super().__init__()

        
        assert mode in ['nearest','bilinear','pixel_shuffle'],'supported upsampling modes are: nearest, bilinear and pixel_shuffle'
        self.mode = mode
        
        if mode == 'pixel_shuffle':
            upscale_factor = 2
            out_channels = out_channels * (upscale_factor ** 2)# * upscale_factor ** 2
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
            
        self.conv_block = ConvLayerEqualBlock(in_channels, out_channels, kernel_size=kernel_size, 
                                              efficiency_optimized=efficiency_optimized)

        if mode == 'pixel_shuffle':
            #ICNR (initialized to CNN Resize) initialization: subkernel symmetries at initialization to reduce checkerboard patterns by enforcing nearest-upsampling at initialization time
            conv = self.conv_block.conv_layer_equal.conv2d
            weight = ICNR(conv.weight,upscale_factor=upscale_factor)
            conv.weight.data.copy_(weight)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        if self.mode in ['nearest','bilinear']:
            output_tensor = F.interpolate(input_tensor, mode=self.mode, scale_factor=2)
            output_tensor = self.conv_block(output_tensor)
        
        elif self.mode == 'pixel_shuffle':
            output_tensor = self.conv_block(input_tensor)
            output_tensor = self.pixel_shuffle(output_tensor)
        
        return output_tensor



class DownsampleConvLayer(torch.nn.Module):
    '''
    A custom neural network module that applies a convolution and a downsampling by a factor of 2. 
    '''

    __slots__ = ('conv', 'relu', 'pooling', 'batch_norm')

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int ,efficiency_optimized: bool = False):

        """
        :param in_channels: Number of channels in the input tensor.
        :param out_channels: Number of channels produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        :param efficiency_optimized: If True, strided convolution is applied instead of equal convolution followed by max-pooling. This is faster.

        :return: The output tensor after applying the module.
        """
        
        super().__init__()

        self.efficiency_optimized = efficiency_optimized
        
        self.relu = torch.nn.ReLU()
        self.bn=nn.BatchNorm2d(out_channels)
        
        if not efficiency_optimized:
            self.conv=ConvLayerEqual(in_channels, out_channels, 
                                        kernel_size=kernel_size,bias=True,efficiency_optimized=efficiency_optimized)#can not set bias=False, because maxpool() and relu() are not invariant to additive constant

            self.pooling = nn.MaxPool2d(2, stride=2)
        
        else:
            #apply a padding as if we want padding == 'same', but then we apply a stride=2, resulting in a downsampling of a factor of 2.
            #If the image is even in shape this guarantees that the downsampled image is also even.
            #for kernelsizes sufficiently large this is allowing to learn lowpass filtering (less aliasing) and better shift equivariance than e.g. max-pooling
            #Instead we could simply apply a fixed low-pass filter (e.g. Gaussian) with stride=2. But then we computational cost without degrees of freedom which is not efficient.
            
            kernel_size = 3#we fix kernel_size==3 for this downsampling 
            padding = 1#for kernel_size =3 the same-padding on all sides is 1
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2,bias=False,padding = padding, padding_mode = 'zeros')#no bias needed because bn direclty after conv
        

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        :param input_tensor: The input tensor to the module.
        :return: The output tensor of the module.
        """
        out = self.conv(input_tensor)
        if not self.efficiency_optimized:
            out = self.pooling(out)
        out = self.relu(self.bn(out))#BN first
        return out



class ResidualBlock(nn.Module):
    """
    Residual Block.

    This block is a part of the Residual Networks introduced in the paper
    "Deep Residual Learning for Image Recognition" (`paper link <https://arxiv.org/abs/1512.03385>`_).
    The recommended architecture can be found in the `Torch blog post <http://torch.ch/blog/2016/02/04/resnets.html>`_.

    When using Batch Normalization (BN), the bias in the preceding Convolutional Layer is redundant because
    the BN output is invariant under additive constants.
    """
    __slots__ = ('conv1', 'batch_norm1', 'conv2', 'batch_norm2', 'relu')

    def __init__(self, channels: int, kernel_size: int = 3, bias: bool = True):
        """
        :param channels: Number of channels in the input and output.
        :param kernel_size: Size of the kernel in the convolutional layers. Default is 3.
        :param bias: Whether to use bias in the convolutional layers. Default is True.
        """
        super().__init__()
        self.conv1 = ConvLayerEqual(channels, channels, kernel_size=kernel_size, bias=bias)
        self.batch_norm1 = nn.BatchNorm2d(channels, affine=True)
        self.conv2 = ConvLayerEqual(channels, channels, kernel_size=kernel_size, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        :param input_tensor: Input tensor.
        :return: Output tensor.
        """

        residual = input_tensor
        out = self.relu(self.batch_norm1(self.conv1(input_tensor)))
        out = self.batch_norm2(self.conv2(out))
        out = out + residual
        return out




#---- Image processing functions -----


def apply_geometric_transformation(input_tensor: torch.Tensor,angle: float,flip: bool = False) -> torch.Tensor:
    """
    Applys a rotation by angle and a reflexion if flip.
    Note that angles should be integer factors of 90 in order to not have border effects.
    :param input_tensor: The pytorch tensor to transform.
    :param angle: The rotation angle in degrees.
    :param flip: If true, the image is flipped.
    :return: Transformed tensor.
    """
    if angle!=0:
        if angle%90 == 0:
            #integer factor of a 90 degrees rotation -> we use the rotation without zeropadding outside of the square
            n_90_degrees = int(angle/90)
            input_tensor = torch.rot90(input_tensor, n_90_degrees , dims=[2,3])
        else:
            input_tensor=TF.rotate(input_tensor, angle)
    if flip:
        input_tensor=TF.hflip(input_tensor)#horizontal flip

    return input_tensor


