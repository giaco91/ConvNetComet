import os
import torch
from torch import nn
import torchmetrics as TM

from utils_segnet_metric import *
from utils_dataset_handling import *
from utils_image_visualization import *

# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.


def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")
# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def


# Train the model for a single epoch
def train_model(model, loader, optimizer = None, mode='train',max_batches=1e8):

    if mode=='train':
        to_device(model.train())
        assert optimizer is not None
    elif mode=='eval':
        to_device(model.eval())
    else:
        raise ValueError('Unknown mode: ',mode)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    running_loss = 0.0
    running_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader, 0):

        inputs = to_device(inputs)
        targets = to_device(targets)
        targets = targets.squeeze(dim=1)
        
        if mode=='train':
            optimizer.zero_grad()
            model_out_members, model_out_ensemble = model(inputs)
            model_out = model_out_members[0]['object_mask']#assuming just one member and that the output group name is 'object_mask'

            loss = criterion(model_out, targets)
            loss.backward()
            optimizer.step()

        elif mode=='eval':
            #evaluation
            with torch.no_grad():
                model_out_members, model_out_ensemble = model(inputs)
                model_out = model_out_members[0]['object_mask']#assuming just one member and that the output group name is 'object_mask'
                loss = criterion(model_out, targets)

        running_samples += targets.size(0)
        running_loss += loss.item()

        if batch_idx>=max_batches:
            break

    average_loss = running_loss / (batch_idx+1)
    return running_samples, average_loss


def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch=None, save_path=None, show_plot=None):
    to_device(model.eval())
    model_out_members, model_out_ensemble = model(to_device(test_pets_targets))
    pred = model_out_ensemble['object_mask']#assuming that the output group name is 'object_mask'

    test_pets_labels = to_device(test_pets_labels)

    pred_labels = pred.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1)
    pred_mask = pred_labels.to(torch.float)
    
    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND))
    iou_accuracy = iou(pred_mask, test_pets_labels)
    pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    pixel_accuracy = pixel_metric(pred_labels, test_pets_labels)
    custom_iou = IoUMetric(pred, test_pets_labels)

    if epoch is not None:
        title = f'Epoch: {epoch:02d}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'
    else:
        title = f'Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'

    # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_targets, nrow=7)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(pred_mask / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"epoch_{epoch:02}.png"), format="png", bbox_inches="tight", pad_inches=0.4)
    
    if show_plot is False:
        close_figures()
    else:
        plt.show()


def test_dataset_accuracy(model, loader,show_every_k_batches=10):
    to_device(model.eval())
    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND))
    pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    
    iou_accuracies = []
    pixel_accuracies = []
    custom_iou_accuracies = []
    
    print_model_parameters(model)
    print('Start evaluation ...')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader, 0):

            inputs = to_device(inputs)
            targets = to_device(targets)
            
            model_out_members, model_out_ensemble = model(inputs)
            pred_probabilities = model_out_ensemble['object_mask']#assuming that the output group name is 'object_mask'

            #pred_probabilities = nn.Softmax(dim=1)(predictions)
            
            pred_labels = pred_probabilities.argmax(dim=1)

            # Add a value 1 dimension at dim=1
            pred_labels = pred_labels.unsqueeze(1)
            # print("pred_labels.shape: {}".format(pred_labels.shape))
            pred_mask = pred_labels.to(torch.float)

            iou_accuracy = iou(pred_mask, targets)
            # pixel_accuracy = pixel_metric(pred_mask, targets)
            pixel_accuracy = pixel_metric(pred_labels, targets)
            custom_iou = IoUMetric(pred_probabilities, targets)
            iou_accuracies.append(iou_accuracy.item())
            pixel_accuracies.append(pixel_accuracy.item())
            custom_iou_accuracies.append(custom_iou.item())

            if batch_idx%show_every_k_batches==0:
                print_test_dataset_masks(model, inputs, targets,epoch=None,save_path=None, show_plot=True)
        
    print_test_dataset_masks(model, inputs, targets, 
        epoch=0, save_path=None, show_plot=True)

    iou_tensor = torch.FloatTensor(iou_accuracies)
    pixel_tensor = torch.FloatTensor(pixel_accuracies)
    custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)
    
    print("Test Dataset Accuracy")
    print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}, Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")


def plot_losses(loss_dict):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_dict['loss_train'], label='Training Loss')
    plt.plot(loss_dict['loss_test'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses Over Epochs')
    plt.legend()
    plt.show()

def train_loop(model, train_loader, test_loader, test_data, epochs, optimizer, scheduler, 
               save_path,print_every=2,show_plots=False):
    
    losses = {'loss_train': [],'loss_test': []}#here we store the losses
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
    test_inputs, test_targets = test_data
    epoch_i, epoch_j = epochs
    
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        running_samples, average_loss = train_model(model, train_loader, optimizer, mode='train')
        losses['loss_train'].append(average_loss)
        print("Trained {} samples, Loss: {:.4f}".format(running_samples,average_loss,))
        
        
        running_samples, average_loss = train_model(model, test_loader, mode='eval',max_batches=10)
        losses['loss_test'].append(average_loss)
        print("Tested {} samples, Loss: {:.4f}".format(running_samples,average_loss,))
        
        if i%print_every==0 or (epoch == epoch_j-1):   
            with torch.inference_mode():
                # Display the plt in the final training epoch.
                print_test_dataset_masks(model, test_inputs, test_targets,
                                         epoch=epoch, save_path=save_path, show_plot=(epoch == epoch_j-1) or show_plots)
            plot_losses(losses)

        if scheduler is not None:
            scheduler.step()