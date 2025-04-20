import os

import mmcv
from PIL import Image
import cv2

from tqdm import tqdm
from enum import Enum



class ResizeMethod(Enum):
    CROP = "crop"              # Crop parts of image to fit target aspect ratio
    PAD = "pad"                # Add padding to fit target aspect ratio
    STRETCH = "stretch"        # Stretch/squash image to target aspect ratio

class NS_ImageLoader():
    def __init__(self,
                dataroot,
                subSet):

        self.curIm = 0
        self.curCam = 0
        self.dataRoot = dataroot
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        #Subset can be train or val
        if subSet == 'train':
            self.infoPath = self.dataRoot+"nuscenes_infos_train.pkl"
        elif subSet == 'val':
            self.infoPath = self.dataRoot+"nuscenes_infos_val.pkl"

        self.infos = mmcv.load(self.infoPath)
        
    def resetLoaderIDX(self):
        self.curIm = 0
        self.curCam = 0

    #Returns a list of dictionaries containing file paths for the Nuscenes images.
    #onlyRain limits the filepaths to only the images of rain scenes
    def imageFiles(self, onlyRain = False):
        imageFilePath = list()

        for sample in tqdm(self.infos):
            curSamp = dict()
            if onlyRain and not sample['RainScene']: #Skip the non rain scenes in the only rain mode
                pass
            else:
                for cam in self.cams:
                    curSamp[cam] = self.dataRoot + sample['cam_infos'][cam]['filename']

                imageFilePath.append(curSamp)
        return imageFilePath

    #Subsequent calls index through the file names. Presents a single filename at a time to the caller.
    #Returns 0 when the end has been reached
    def SingleImageLoader(self, onlyRain = False):

        if onlyRain == True:
            while self.infos[self.curIm]['RainScene'] == False:
                curIm += 1 #Skip through the non rain scenes if looking for only rain scenes
        if self.curIm < len(self.infos):
            fileReturn = self.dataRoot + self.infos[self.curIm]['cam_infos'][self.cams[self.curCam]]['filename']
        else:
            fileReturn = 0
        if self.curCam >= (len(self.cams)-1):
            self.curIm +=1
            self.curCam = 0
        else:
            self.curCam +=1

        return fileReturn

    def saveImage(self, newImage, origFileName, extension):
        img_name = origFileName.split('.')[0]
        newName = img_name + extension + '.jpg'
        cv2.imwrite(newName, newImage)

    #Base AI Generated
    def downscale_image(self,image, target_width, target_height, 
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

    #Base AI Generated
    def resize_image(self, image, target_width, target_height, 
                method=ResizeMethod.CROP, interpolation=cv2.INTER_CUBIC,
                bg_color=(0, 0, 0), pad_position="center"):
    """
    Resize an image to target dimensions using the specified method
    
    Default to crop resize method to not distort during upscaling

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
        

if __name__ == '__main__':
    dataRoot = "/scratch/cdauche/CRN/data/nuScenes/"
    loader = NS_ImageLoader(dataRoot, 'val')
    temp = loader.imageFiles(onlyRain = True)

    i = 0
    while i < 10:
        temp = loader.SingleImageLoader()
        print(temp)
        i+=1


    pic = Image.open(temp)
    print(pic.size)
    pic = pic.save('ChosenNuscenesImage.jpg')

