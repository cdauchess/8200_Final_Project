import os

import mmcv
from PIL import Image
import cv2

from tqdm import tqdm




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

