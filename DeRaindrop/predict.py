#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim
from enum import Enum

from tqdm import tqdm


import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/datasets')
from GetNuScImage import NS_ImageLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

#model is expecting a 480x720 input image, resize to match expected input
def resizeIm(image):
    return cv2.resize(image,(720,480), interpolation=cv2.INTER_AREA)

def upscale_noPad(image):
    return cv2.resize(image,(1600,900), interpolation=cv2.INTER_CUBIC)


def downscale_image(image, target_width, target_height, 
                   interpolation=cv2.INTER_AREA, bg_color=(0, 0, 0),
                   pad_position="center"):
    """
    Downscale an image to target dimensions without cropping.
    When aspect ratios differ, padding is added.
    
    Parameters:
    image_path (str): Path to input image
    target_width (int): Target width in pixels
    target_height (int): Target height in pixels
    interpolation (int): OpenCV interpolation method
    bg_color (tuple): Background color for padding (B, G, R format)
    pad_position (str): Where to place padding ("center", "left", or "right")
    
    Returns:
    numpy.ndarray: Downscaled image
    """
    # Read the input image
    img = image
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get original dimensions
    orig_height, orig_width = img.shape[:2]
    
    # Create a canvas of target size filled with the background color
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    result[:] = bg_color
    
    # Calculate the scaling factor that preserves aspect ratio
    # and ensures the image fits completely within the target dimensions
    scale_width = target_width / orig_width
    scale_height = target_height / orig_height
    
    # Use the smaller scale factor to ensure the entire image fits
    scale_factor = min(scale_width, scale_height)
    
    # Calculate new dimensions after scaling
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    
    # Resize the image using INTER_AREA for downscaling (better quality)
    resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    
    # Calculate offsets for positioning
    if pad_position == "center":
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
    elif pad_position == "left":
        x_offset = 0
        y_offset = (target_height - new_height) // 2
    elif pad_position == "right":
        x_offset = target_width - new_width
        y_offset = (target_height - new_height) // 2
    else:
        x_offset = 0
        y_offset = 0
    
    # Place the resized image on the canvas
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result

class ResizeMethod(Enum):
    CROP = "crop"              # Crop parts of image to fit target aspect ratio
    PAD = "pad"                # Add padding to fit target aspect ratio
    STRETCH = "stretch"        # Stretch/squash image to target aspect ratio

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
    
    # PAD method - scale and pad to fit target aspect ratio
    elif method == ResizeMethod.PAD:
        # Create a canvas of target size filled with the background color
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        result[:] = bg_color
        
        # Calculate the scaling factor that preserves aspect ratio
        # and ensures the image fits completely within the target dimensions
        scale_width = target_width / orig_width
        scale_height = target_height / orig_height
        
        # Use the smaller scale factor to ensure the entire image fits
        scale_factor = min(scale_width, scale_height)
        
        # Calculate new dimensions after scaling
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Resize the image
        resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        
        # Calculate offsets for positioning
        if pad_position == "center" or new_height == target_height:
            x_offset = (target_width - new_width) // 2
        elif pad_position == "left":
            x_offset = 0
        elif pad_position == "right":
            x_offset = target_width - new_width
        else:
            x_offset = 0  # Default to left
        
        y_offset = (target_height - new_height) // 2
        
        # Place the resized image on the canvas
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return result
    
    else:
        raise ValueError("Invalid resize method specified")

#Base generated with AI
def upscale_image_with_padding(image, target_width, target_height, 
                              interpolation=cv2.INTER_CUBIC, bg_color=(0, 0, 0)):
    """
    Upscale an image to target dimensions while preserving aspect ratio and padding with black
    
    Parameters:
    image: the image to be upscaled
    target_width (int): Target width
    target_height (int): Target height
    interpolation (int): OpenCV interpolation method
    bg_color (tuple): Background color for padding (default: black)
    
    Returns:
    numpy.ndarray: Upscaled and padded image
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
    
    # Create a black canvas of target size
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # Fill with background color
    result[:] = bg_color
    
    if target_aspect > orig_aspect:
        # Target is wider than original
        # Scale based on height
        scale_factor = target_height / orig_height
        new_width = int(orig_width * scale_factor)
        new_height = target_height
        
        # Resize the image
        resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        
        # Calculate x-offset to center the image
        x_offset = (target_width - new_width) // 2
        
        # Place the resized image on the black canvas
        result[:, x_offset:x_offset+new_width] = resized
    else:
        # Target is taller than original or same aspect ratio
        # Scale based on width
        scale_factor = target_width / orig_width
        new_width = target_width
        new_height = int(orig_height * scale_factor)
        
        # Resize the image
        resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        
        # Calculate y-offset to center the image
        y_offset = (target_height - new_height) // 2
        
        # Place the resized image on the black canvas
        result[y_offset:y_offset+new_height, :] = resized
    
    return result

def predict(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    out = model(image)[-1]

    

    out = out.cpu().data
    out = out.numpy()

    out = out.transpose((0, 2, 3, 1))
    print(out.shape)
    out = out[0, :, :, :]*255.

    print(out.shape)
    #cv2.imwrite('./demo/maskSave/mask.png', out)
    return out

def outputCompare(original, result, inName):
    plt.figure(figsize=(9, 4))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.axis('off')
    plt.xlim(0,1600)
    plt.ylim(900,0)
    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.imshow(img.astype(int))

    plt.subplot(1,2,2)
    plt.title("De-Rained Image")
    plt.axis('off')
    plt.xlim(0,1600)
    plt.ylim(900,0)
    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(img.astype(int))

    plt.savefig(f'./demo/NS_Val_Rain/{inName}-COMP.png')
    plt.close()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    args = get_args()



    model = Generator().cuda()
    print("Loading Weights...")
    model.load_state_dict(torch.load('./weights/gen.pkl'))

    

    if args.mode == 'demo':
        print('Loading Image List...')
        dataRoot = "/project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/data/nuScenes/"
        DL = NS_ImageLoader(dataRoot, 'val')
        #input_list = sorted(os.listdir(args.input_dir))
        input_list = DL.imageFiles(onlyRain=False) #Load a list of all files to be cleansed
        num = len(input_list)
        for i in tqdm(range(num)):
            for cam in DL.cams:
                fileName = input_list[i][cam].split('/')[-1]
                print ('Processing image: %s'%(input_list[i][cam]))
                #Origimg = cv2.imread(args.input_dir + input_list[i])
                Origimg = cv2.imread(input_list[i][cam])
                #img = resizeIm(Origimg)
                #img = resize_image(Origimg,720,480, method = ResizeMethod.STRETCH)
                img = downscale_image(Origimg, 720,480)
                img = align_to_four(img)
                result = predict(img)
                #outputCompare(img,result,(input_list[i]+'-smallRes'))
                #result = upscale_image_with_padding(result,1600,900)
                #result = upscale_noPad(result)
                result = resize_image(result, 1600, 900, method= ResizeMethod.CROP)
                img_name = input_list[i][cam].split('.')[0]
                #cv2.imwrite(args.output_dir + img_name + '.jpg', result)
                print('Saving Image: %s' %(img_name + '_DRD_C' + '.jpg')) #Add file extension to indicate new File
                cv2.imwrite(img_name + '_DRD_C' + '.jpg', result)
                outputCompare(Origimg,result,fileName.split('.')[0]) #Save a comparison of the original and derained image

    if args.mode =='smallTest':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in tqdm(range(num)):

            fileName = input_list[i].split('/')[-1]
            print ('Processing image: %s'%(input_list[i]))
            Origimg = cv2.imread(args.input_dir + input_list[i])
            #img = resizeIm(Origimg)
            #img = resize_image(Origimg,720,480, method = ResizeMethod.STRETCH)
            img = downscale_image(Origimg, 720,480)
            img = align_to_four(img)
            result = predict(img)
            #outputCompare(img,result,(input_list[i]+'-smallRes'))
            #result = upscale_image_with_padding(result,1600,900)
            #result = upscale_noPad(result)
            result = resize_image(result, 1600, 900, method= ResizeMethod.CROP)
            img_name = input_list[i].split('.')[0]
            #cv2.imwrite(args.output_dir + img_name + '.jpg', result)
            print('Saving Image: %s' %(img_name + '_DRD_C' + '.jpg')) #Add file extension to indicate new File
            cv2.imwrite(args.output_dir + img_name + '_DRD_C' + '.jpg', result)
            #outputCompare(Origimg,result,fileName.split('.')[0]) #Save a comparison of the original and derained image


    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])
            img = align_to_four(img)
            gt = align_to_four(gt)
            result = predict(img)
            result = np.array(result, dtype = 'uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

    else:
        print ('Mode Invalid!')
