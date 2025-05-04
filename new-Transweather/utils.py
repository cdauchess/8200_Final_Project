import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import cv2

import skimage
import cv2
from skimage.measure import compare_psnr, compare_ssim
import pdb
from enum import Enum

class ResizeMethod(Enum):
    CROP = "crop"              # Crop parts of image to fit target aspect ratio
    PAD = "pad"                # Add padding to fit target aspect ratio
    STRETCH = "stretch"        # Stretch/squash image to target aspect ratio

def calc_psnr(im1, im2):

    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()


    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans

def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def validation(net, val_data_loader, device, exp_name, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, imgid, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_val(net, val_data_loader, device, exp_name, category, save_tag=False, saveExt = ""):

    psnr_list = []
    ssim_list = []
    imConv = 0

    for batch_id, val_data in enumerate(val_data_loader):
        imConv+=1
        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

# --- Calculate the average PSNR --- #
        #psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        #ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, imgid, exp_name,category, saveExt)


    avr_psnr = 0
    avr_ssim = 0
    #avr_psnr = sum(psnr_list) / len(psnr_list)
    #avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def save_image(pred_image, image_name, exp_name, category, saveExt = ""):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        imgName = image_name[ind].split('.')[0]
        saveImgName = imgName+saveExt+'.jpg'
        print(saveImgName)

        result = pred_image_images[ind]
        
        print(pred_image_images[ind].shape)
        #utils.save_image(pred_image_images[ind], './results/NSTest/transweather/{}'.format(image_name_1))
        #imgToSave = resize_image(result[0,:,:,:]*255., 1600, 900, method = ResizeMethod.CROP)
        result = result.cpu().data
        result = result.numpy()
        imgToSave = result.transpose((0, 2, 3, 1))
        imgToSave = imgToSave[0,:,:,:]*255.0
        #imgToSave = imgToSave.transpose((1,2,0))
        print(imgToSave.shape)
        imgToSave = resize_image(imgToSave, 1600, 900, method = ResizeMethod.CROP)
        #cv2.imwrite('./results/NSTest/transweather/{}'.format(image_name_1),imgToSave)
        cv2.imwrite(saveImgName,imgToSave)
        #utils.save_image(imgToSave, saveImgName)


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format( exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)



def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 100

    torch.Exponential(optimizer, gamma=0.95)

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


#Base AI Generated
def resize_image(image, target_width, target_height, 
                method=ResizeMethod.PAD, interpolation=cv2.INTER_CUBIC,
                bg_color=(0, 0, 0), pad_position="center"):
    """
    Resize an image to target dimensions using the specified method
    
    Parameters:
    image_path (str): Path to input image
    target_width (int): Target width in pixels
    target_height (int): Target height in pixels
    method (ResizeMethod): Method to handle aspect ratio differences
    interpolation (int): OpenCV interpolation method
    bg_color (tuple): Background color for padding (default: black)
    pad_position (str): Where to position the image when padding ("center", "left", "right")
    
    Returns:
    numpy.ndarray: Resized image
    """
    # Read the input image
    img = image
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get original dimensions
    orig_height, orig_width = img.shape[:2]
    
    # Calculate aspect ratios
    orig_aspect = orig_width / orig_height
    target_aspect = target_width / target_height
    
    # STRETCH method - simple resize without preserving aspect ratio
    if method == ResizeMethod.STRETCH:
        return cv2.resize(img, (target_width, target_height), interpolation=interpolation)
    
    # CROP method - scale and crop to fit target aspect ratio
    elif method == ResizeMethod.CROP:
        # Always scale the image so that the target dimension is completely filled,
        # which may require cropping in one dimension
        if orig_aspect > target_aspect:
            # Original is wider - crop width
            scale_factor = target_height / orig_height
            new_width = int(orig_width * scale_factor)
            new_height = target_height
            
            # Resize first
            resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            
            # Calculate how much to crop from sides
            crop_amount = new_width - target_width
            left_crop = crop_amount // 2
            
            # Crop center region
            result = resized[:, left_crop:left_crop+target_width]
            
        else:
            # Original is taller - crop height
            scale_factor = target_width / orig_width
            new_width = target_width
            new_height = int(orig_height * scale_factor)
            
            # Resize first
            resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            
            # Calculate how much to crop from top/bottom
            crop_amount = new_height - target_height
            top_crop = crop_amount // 2
            
            # Crop center region
            result = resized[top_crop:top_crop+target_height, :]
        
        return result
	