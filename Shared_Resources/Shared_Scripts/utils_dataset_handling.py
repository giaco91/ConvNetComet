import os
import torch
import torchvision

from enum import IntEnum




class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2


# Simple torchvision compatible transform to send an input tensor
# to a pre-specified device.
class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={device})"


# Create a dataset wrapper that allows us to perform custom image augmentations
# on both the target and label (segmentation mask) images.
#
# These custom image augmentations are needed since we want to perform
# transforms such as:
# 1. Random horizontal flip
# 2. Image resize
#
# and these operations need to be applied consistently to both the input
# image as well as the segmentation mask.
class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)


# Create a tensor for a segmentation trimap.
# Input: Float tensor with values in [0.0 .. 1.0]
# Output: Long tensor with values in {0, 1, 2}
def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs



def get_data_loader(data_save_dir, transform_dict, download=False, batch_size_train=12, batch_size_test=6):
    
    # Oxford IIIT Pets Segmentation dataset loaded via torchvision.
    pets_path_train = os.path.join(data_save_dir, 'OxfordPets', 'train')
    pets_path_test = os.path.join(data_save_dir, 'OxfordPets', 'test')
    
    # Create the train and test instances of the data loader for the
    # Oxford IIIT Pets dataset with random augmentations applied.

    pets_train = OxfordIIITPetsAugmented(
        root=pets_path_train,
        split="trainval",
        target_types="segmentation",
        download=download,
        **transform_dict,
    )
    
    pets_test = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        target_types="segmentation",
        download=download,
        **transform_dict,
    )
    
    pets_train_loader = torch.utils.data.DataLoader(
    pets_train,
    batch_size=batch_size_train,
    shuffle=True,
    )

    pets_test_loader = torch.utils.data.DataLoader(
        pets_test,
        batch_size=batch_size_test,
        shuffle=True,
    )
    
    return pets_train_loader, pets_test_loader
    