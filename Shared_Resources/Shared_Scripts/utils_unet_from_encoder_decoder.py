
'''

The Unet is constructed using the cnn encoder and decoder.

'''



import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import math

import numpy as np

from typing import Tuple, Optional, Dict, List,TypedDict

from utils_cnn_encoder_and_decoder import CNNEncoder, CNNDecoder



#----- U-Net architecture that combines the CNNEncoder and CNNDeocder to a model that maps image to image ----

class UNet(torch.nn.Module):

    '''
        UNet that is formalized as a composition of an encoder (CNNEncoder) that has skip connection at different scales and
        a decoder (CNNDecoder) that upsamples to the original scale processing and accumulating the information from all scales
    '''

    __slots__ = ('in_ch', 'out_ch', 'output_groupname_to_n_channels', 'output_groupname_to_name_to_idx', 'efficiency_optimized',
                 'n_decoders', 'dimension_specification', 'verbose', 'model_type_to_dimension_specification', 'encoder',
                 'decoders')

    def __init__(self,in_ch: int,out_ch: int,n_decoders: int = 1,verbose: bool = False,
                    output_groupname_to_n_channels: Dict[str, int] = {}, 
                    output_groupname_to_name_to_idx: Dict[str,Dict[str,int]] = {},
                    efficiency_optimized: bool=False,
                    dimension_specification: Dict[int,Dict[str,int]] = {0: {'width_encoder': 10, 'depth_encoder': 1,'kernel_size_encoder': 3,
                                                                            'width_decoder': 10, 'depth_decoder': 1,'kernel_size_decoder': 3,'width_skip': 5},
                                                                        1:{'width_encoder': 20, 'depth_encoder': 1,'kernel_size_encoder': 2, 
                                                                            'width_decoder': 20, 'depth_decoder': 1,'kernel_size_decoder': 2,'width_skip': 5}
                                                                        }):

        super().__init__()

        '''
        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param output_groupname_to_n_channels: If the output is grouped (e.g. softmax applied separately to different output channel groups).
                    Example: output_groupname_to_n_channels = {'object_mask': 3,'semantic_mask': 10, 'focus_mask': 2}
                    The values in output_groupname_to_n_channels have to add up to out_ch.
                    Empty dict allowed for no grouping.
        :param output_groupname_to_name_to_idx: For each output groupname we require a dictionary that the classname to its class index. Empty dictionary allowed, if a default assinement should be made.
        :param n_decoders: Allows to initialize more than one decoders, resulting in an ensemble of decoders
        :param verbose: If true, prints extra information about the model architecture
        :param efficiency_optimized: Applies trades that reduces computational cost (giving up residual convolutions, reflection padding and max-pooling)
        :param dimension_specification: A nested dictionary, that maps each scale level to the dimension specifications of the CNN-encoder and decoder.
        '''

        assert in_ch>0
        assert out_ch>0
        assert n_decoders>0

        self.in_ch=in_ch
        self.out_ch=out_ch

        if len(output_groupname_to_n_channels) == 0:
            output_groupname_to_n_channels['Unet_out'] = out_ch

        if len(output_groupname_to_name_to_idx) == 0:
            output_groupname_to_name_to_idx['Unet_out'] = {'class-'+str(i): i for i in range(out_ch)}

        self.output_groupname_to_n_channels = output_groupname_to_n_channels
        self.output_groupname_to_name_to_idx = output_groupname_to_name_to_idx
        self.check_output_groupname_consistency()

        self.efficiency_optimized = efficiency_optimized

        self.n_decoders = n_decoders

        self.dimension_specification = dimension_specification
        self.verbose = verbose
        
        self.model_type_to_dimension_specification = self.get_model_type_to_dimension_specification()

        self.encoder = self.get_encoder()
        self.decoders = self.get_decoders()

    def check_output_groupname_consistency(self):
        """
        Consistency check of_output_groupname_to_n_channels and output_groupname_to_name_to_idx. The output channels have to add up to self.out_ch

        :raises Assert Error: If the check failes

        """
        assert np.asarray(list(self.output_groupname_to_n_channels.values())).sum() == self.out_ch,'The values in output_groupname_to_n_channels have to add up to out_ch'
        assert len(self.output_groupname_to_name_to_idx) == len(self.output_groupname_to_n_channels)
        for group_name in self.output_groupname_to_name_to_idx.keys():
            assert len(self.output_groupname_to_name_to_idx[group_name]) == self.output_groupname_to_n_channels[group_name], 'the number of channels for a group must correspond the number of classes'


    def get_model_type_to_dimension_specification(self) -> Dict[str,Dict[int,Dict[str,int]]]:
        '''
        Splits the dimension specification for the encoder and decoder

        :returns: The dimension specification that should go in to the CNNEncoder and CNNDecoder respectively
                    as a dictionary with the keys 'encoder' and 'decoder' identifying the corresponding dimension specifications.
        '''
        model_type_to_dimension_specification = {'encoder': {}, 'decoder': {}}
        for scale, specification in self.dimension_specification.items():
            model_type_to_dimension_specification['encoder'][scale] = {'width': specification['width_encoder'],
                                                                    'depth': specification['depth_encoder'],
                                                                    'kernel_size': specification['kernel_size_encoder'],
                                                                    'width_skip': specification['width_skip']}
            model_type_to_dimension_specification['decoder'][scale] = {'width': specification['width_decoder'],
                                                                    'depth': specification['depth_decoder'],
                                                                    'kernel_size': specification['kernel_size_decoder'],
                                                                    'width_skip': specification['width_skip']}
        return model_type_to_dimension_specification

    def get_encoder(self) -> nn.Module:
        """
        Returns the CNN encoder

        :returns: the encoder
        """
        encoder = CNNEncoder(in_ch = self.in_ch, dimension_specification = self.model_type_to_dimension_specification['encoder'],
                                            verbose = self.verbose,efficiency_optimized = self.efficiency_optimized)
        return encoder

    
    def get_decoders(self) -> nn.ModuleList:
        """
        Creates an ensemble of CNN decoders

        :returns: the decoders
        """
        decoders = nn.ModuleList()
        for _ in range(self.n_decoders):
            decoder = CNNDecoder(out_ch=self.out_ch,output_activation = 'identity',
                                    dimension_specification = self.model_type_to_dimension_specification['decoder'], 
                                    verbose=self.verbose,efficiency_optimized = self.efficiency_optimized)
            decoders.append(decoder)
        return decoders
        
    
    
    def apply_single_decoder(self,decoder_idx: int, skip_connections: Dict[str, Dict[str,torch.Tensor]]) -> torch.Tensor:

        assert 0 <= decoder_idx<self.n_decoders,'The decoder_index must be smaller than the number of decoders available'
        decoder = self.decoders[decoder_idx]
        y = decoder(skip_connections)
        return y

    def split_output_channels(self,output_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Splits the raw output tensor of the decoder in separate subtensors (channel-wise), corresponding to different tasks, like semantic segmentation and focus segmentation.

        :param output_tensor: The raw output tensor with general shape (B,C,W,H)
        """
        if self.output_groupname_to_n_channels is None:
            out = {'Unet_out': output_tensor}
        else:
            out = {}
            ch_count = 0
            for out_name, out_ch in self.output_groupname_to_n_channels.items():
                new_ch_count = ch_count + out_ch
                out[out_name] = output_tensor[:,ch_count:new_ch_count,:,:]
                ch_count = new_ch_count
        
        return out


    def apply_decoders(self,encoder_out: Dict[str, Dict[str,torch.Tensor]],
                        ensemble_size: int = 0, with_member_output: bool = True) -> (List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]):
        """
        Applies the decoders to the encoder output. 

        :param encoder_out: The output of the encoder, that is the feature maps at various scales produced by calling the encoder using forward()
        :param ensemble_size: The number of decoder ensemble members used to over which to average. If 0, it will use all members available.
        :param with_member_output: If true, the individual ensemble member predictions are also returned (slightly more memory intense)

        :return: The member and the ensemble predictions

        """
        
        if with_member_output:
            decoder_out_members = []
        else:
            decoder_out_members = None

        if ensemble_size == 0:
            n_decoders = self.n_decoders 
        else:

            n_decoders = max(1, min(ensemble_size,self.n_decoders))
            if self.verbose:
                if ensemble_size>self.n_decoders:
                    print('Warning: The ensemble size can not be larger than the maximum number of decoders. Set beack to the maximum value')

        
        for decoder_idx in range(n_decoders):
            y = self.apply_single_decoder(decoder_idx,encoder_out)
            decoder_out = self.split_output_channels(y)
            if with_member_output:
                decoder_out_members.append(decoder_out)

            #Sum the decoders outputs in the probability domain
            #we never need gradients through the ensembled ouptut
            with torch.no_grad():
                if decoder_idx == 0:
                    decoder_out_ensemble = {k: 0 for k in decoder_out.keys()}
                for out_key, out in decoder_out.items():
                    decoder_out_ensemble[out_key]+=F.softmax(out,dim=1)
        
        #normalizing by the number of decoders
        with torch.no_grad():
            for out_key, out in decoder_out_ensemble.items():
                decoder_out_ensemble[out_key] = out/n_decoders
        
        return decoder_out_members, decoder_out_ensemble

        
    def forward(self, x, ensemble_size: int = 0, with_member_output=True):
        encoder_out = self.encoder(x)
        decoder_out_members, decoder_out_ensemble = self.apply_decoders(encoder_out,ensemble_size=ensemble_size,with_member_output=with_member_output)

        return decoder_out_members, decoder_out_ensemble


    def forward_with_test_time_augmentation(self,x, n_augmentations = 8, ensemble_size=None,with_member_output=True):
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
                decoder_out_members, decoder_out_ensemble = self.forward(x_tf, ensemble_size = ensemble_size,with_member_output=with_member_output)

                if angle==0 and not flip and with_member_output:
                    #Because we assume to do this only at inference, we save memory and time to not return the transformed and averaged member outputs. 
                    #We just return the one without transformation
                    decoder_out_members_to_return = decoder_out_members
                else:
                    decoder_out_members_to_return = None

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
