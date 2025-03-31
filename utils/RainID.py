import mmcv
import numpy as np

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def isRain(descript):
    testWord = "rain"
    testWord2 = "Rain"
    if testWord in cur_scene['description']:
        return True
    elif testWord2 in cur_scene['description']:
        return True
    else:
        return False


testWord = "rain"
testWord2 = "Rain"

count = 0


trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot='../data/nuScenes/',
                             verbose=True)

cur_scene = trainval_nusc.scene[0]

print(len(trainval_nusc.scene))

print(cur_scene['description'])

for cur_scene in tqdm(trainval_nusc.scene):
    if isRain(cur_scene['description']):
        count += 1
    pass


print(count)   

#infos = mmcv.load('../data/nuScenes/nuscenes_infos_val.pkl')

#print(len(infos))

#print(infos[1])

#infosT = mmcv.load('../data/nuScenes/nuscenes_infos_train.pkl')
#print(len(infosT))

#temp = infosT[1]
#print(temp['scene_token'])