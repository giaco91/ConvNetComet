
'''

The Unet is constructed using the cnn encoder and decoder.

'''



import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import math

import numpy as np

from utils_cnn_encoder_and_decoder import CNNEncoder, CNNDecoder



class UNet(torch.nn.Module):

    '''
        UNet that is formalized as a composition of an encoder (CNNEncoder) that has skip connection at different scales and
        a decoder (CNNDecoder) that upsamples to the original scale processing and accumulating the information from all scales
    '''

    def __init__(self,in_ch,out_ch,n_decoders=1,verbose=False,pooling_mode='max',
                    output_groupname_to_n_channels = None,
                    dimension_specification =   {0: {'width_encoder': 10, 'depth_encoder': 1, 
                                                    'width_decoder': 10, 'depth_decoder': 1,'width_skip': 5},
                                                1:{'width_encoder': 20, 'depth_encoder': 1, 
                                                    'width_decoder': 20, 'depth_decoder': 1,'width_skip': 5}
                                                }):

        super(UNet, self).__init__()

        '''

        Parameteres:
            in_ch: Number of input channels
            out_ch: Number of output channels
            output_groupname_to_n_channels: If the output is grouped (e.g. softmax applied separately to different output channel groups).
                    Example: output_groupname_to_n_channels = {'object_mask': 3,'semantic_mask': 10, 'focus_mask': 2}
                    The values in output_groupname_to_n_channels have to add up to out_ch
            n_decoders: Allows to initialize more than one decoders, resulting in an ensemble of decoders
            verbose: If true, prints extra information about the model architecture
            pooling_mode: The pooling mode of the downasmpling in the encoder
        '''

        assert in_ch>0
        assert out_ch>0
        assert n_decoders>0

        if output_groupname_to_n_channels is not None:
            assert np.asarray(list(output_groupname_to_n_channels.values())).sum() == out_ch,'The values in output_groupname_to_n_channels have to add up to out_ch'

        self.output_groupname_to_n_channels = output_groupname_to_n_channels
      
        self.n_decoders = n_decoders
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.dimension_specification = dimension_specification
        self.pooling_mode = pooling_mode
        self.verbose = verbose
        
        self.model_type_to_dimension_specification = self.get_model_type_to_dimension_specification()

        self.encoder = CNNEncoder(in_ch = in_ch, dimension_specification = self.model_type_to_dimension_specification['encoder'],
                                    pooling_mode=pooling_mode, verbose = verbose)
        self.decoders = self.get_decoders()


    def get_model_type_to_dimension_specification(self):
        #returns the dimension specification that should go in to the CNNEncoder and CNNDecoder respectively
        #returns it as a dictionary with the keys 'encoder' and 'decoder' identifying the corresponding dimension specifications.
        model_type_to_dimension_specification = {'encoder': {}, 'decoder': {}}
        for scale, specification in self.dimension_specification.items():
            model_type_to_dimension_specification['encoder'][scale] = {'width': specification['width_encoder'],
                                                                    'depth': specification['depth_encoder'],
                                                                    'width_skip': specification['width_skip']}
            model_type_to_dimension_specification['decoder'][scale] = {'width': specification['width_decoder'],
                                                                    'depth': specification['depth_decoder'],
                                                                    'width_skip': specification['width_skip']}
        return model_type_to_dimension_specification


    
    def get_decoders(self):
        decoders = nn.ModuleList()
        for _ in range(self.n_decoders):
            decoder = CNNDecoder(out_ch=self.out_ch,output_activation = None,
                                    dimension_specification = self.model_type_to_dimension_specification['decoder'], verbose=self.verbose)
            decoders.append(decoder)
        return decoders
        
    
    
    def apply_single_decoder(self,decoder_idx,skip_connections,pad=None):
        assert 0 <= decoder_idx<self.n_decoders,'The decoder_index must be smaller than the number of decoders available'
        decoder = self.decoders[decoder_idx]
        y = decoder(skip_connections,pad=pad)
        return y

    def split_output_channels(self,y):
        if self.output_groupname_to_n_channels is None:
            out = {'Unet_out': y}
        else:
            out = {}
            ch_count = 0
            for out_name, out_ch in self.output_groupname_to_n_channels.items():
                new_ch_count = ch_count + out_ch
                out[out_name] = y[:,ch_count:new_ch_count,:,:]
                ch_count = new_ch_count
        
        return out


    def apply_decoders(self,encoder_out,ensemble_size=None):
        
        decoder_out_members = []

        if ensemble_size is None:
            n_decoders = self.n_decoders 
        else:
            n_decoders = max(1, min(ensemble_size,self.n_decoders))
            if self.verbose:
                if ensemble_size>self.n_decoders:
                    print('Warning: The ensemble size can not be larger than the maximum number of decoders. Set beack to the maximum value')

        
        for decoder_idx in range(n_decoders):
            y = self.apply_single_decoder(decoder_idx,encoder_out)
            decoder_out = self.split_output_channels(y)
            decoder_out_members.append(decoder_out)
            
            #Sum the decoders outputs in the probability domain
            #we never need gradients through the ensembled ouptut
            #because for training we have to individually backprop through the members. The ensemble is always used in inferrecne only
            with torch.no_grad():
                if decoder_idx == 0:
                    decoder_out_ensemble = {k: 0 for k in decoder_out.keys()}
                for out_key, out in decoder_out.items():
                    decoder_out_ensemble[out_key]+=F.softmax(out,dim=1)
        
        #normalizing by the number of decoders
        for out_key, out in decoder_out_ensemble.items():
            decoder_out_ensemble[out_key] = out/n_decoders
        
        return decoder_out_members, decoder_out_ensemble

        
    def forward(self, x, ensemble_size=None):

        encoder_out = self.encoder(x)

        decoder_out_members, decoder_out_ensemble = self.apply_decoders(encoder_out,ensemble_size=ensemble_size)

        #return decoder_out_members, decoder_out_ensemble
        return decoder_out_members[0]['Unet_out']



    def forward_with_test_time_augmentation(self,x, n_augmentations = 8, ensemble_size=None):
        '''
        Uses the 8 element reflection and 90 degrees rotation group to produce an output that is invariant under those geometric group transformations
        
        Note: Because we assume to do this only at inference, we don't also process the individual member outputs in the same way. 
        We just return the average predictions of all augmented versions
        '''
        angles = [0,90,180,270]
        flips = [False,True]
        n_transformations_available = len(angles)*len(flips)
        n_augmentations = max(1, min(n_transformations_available, n_augmentations))#clipping

        count=0
        break_out=False
        for angle in angles:
            for flip in flips:
                #forward transformation
                x_tf = apply_geometric_transformation(x,angle,flip)
                
                #model propagation
                decoder_out_members, decoder_out_ensemble = self.forward(x_tf, ensemble_size = ensemble_size)

                if angle==0 and not flip:
                    #Because we assume to do this only at inference, we save memory and time to not return the transformed and aberaged member outputs. 
                    #We just return the one without transformation
                    decoder_out_members_to_return = decoder_out_members

                #initialize
                if count == 0:
                    decoder_out_tf = {k: 0 for k in decoder_out_ensemble.keys()}

                #inverse transformation using the same sequence of operation (rotation first then reflection)
                inverse_angle = (int(flip)*2-1)*angle
                inverse_flip = flip

                for out_key, out in decoder_out_ensemble.items():
                    decoder_out_tf[out_key] += apply_geometric_transformation(out,inverse_angle,inverse_flip)

                count+=1

                if count>=n_augmentations:
                    break_out = True
                    break

            if break_out:
                break

        #normalizing by the number of transformations
        for out_key, out in decoder_out_tf.items():
            decoder_out_tf[out_key] = out/n_augmentations

        return decoder_out_members_to_return, decoder_out_tf



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
