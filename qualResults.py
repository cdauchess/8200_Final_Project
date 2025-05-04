from tqdm import tqdm
from datasets.GetNuScImage import NS_ImageLoader
import cv2
import matplotlib.pyplot as plt



if __name__ == '__main__':
    #List of extensions to plot:
    extPlot = ["", "_DRD_C", "_UtilIR1", "_UtilIR2"]
    plotName = ["Original", "DeRaindrop", "UtilityIR 1", "UtilityIR 2"]

    #Load image list:
    dataRoot = "/project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/data/nuScenes/" 
    DL = NS_ImageLoader(dataRoot, 'val')

    imageList = DL.SingleListImages(onlyRain = True) #Only compare the rain images

    #Iterate through each image
    for imgFilePath in tqdm(imageList):
        images = []
    
        plt.figure(figsize=(16,4))
        #Load each version of the image:
        for j in range(len(extPlot)):
            filePathTemp = imgFilePath.split('.')[0]
            filePathUse = filePathTemp + extPlot[j] + '.jpg'
            images.append(cv2.imread(filePathUse))
            plt.subplot(1,len(extPlot), j+1)
            plt.title(plotName[j])
            plt.axis('off')
            plt.xlim(0,1600)
            plt.ylim(900,0)
            img = cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB)
            plt.imshow(img.astype(int))

        #Generate plot title
        imgName = imgFilePath.split('/')[-1]
        imgName = imgName.split('.')[0]
        plt.savefig(f'./outputCompare/{imgName}-COMP.png')
        plt.close()
