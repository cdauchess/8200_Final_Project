import mmcv
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def isRain(descript):
    testWord = "rain"
    testWord2 = "Rain"
    if testWord in descript:
        return True
    elif testWord2 in descript:
        return True
    else:
        return False

def generate_info(nusc, scenes, rainOnly = False, max_cam_sweeps=6, max_lidar_sweeps=10):
    infos = list()
    rainIdx = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        #Skip the non rain scenes if we're looking for rain only scenes
        if not isRain(nusc.get('scene', cur_scene['token'])['description']) and rainOnly: 
            continue

        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['scene_name'] = nusc.get('scene', cur_scene['token'])['name']
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']

            if isRain(nusc.get('scene', cur_scene['token'])['description']):
                info['RainScene'] = True #Embed an identifier in the infos file for rain
                rainIdx.append(len(infos)) #Place the index of the rain scene in this list for easy recall later
            else:
                info['RainScene'] = False

            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                #TODO - Add conditional for a changed file name to load the cleansed images when working with Nuscenes
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()

            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos, rainIdx


def main():
    trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot='../data/nuScenes/',
                             verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val
    print('Starting Tiny...')
    train_infos_tiny, tinyRain = generate_info(trainval_nusc, train_scenes[:2])
    mmcv.dump(train_infos_tiny, '../data/nuScenes/nuscenes_infos_train-tiny.pkl')
    print('Starting Train...')
    train_infos, trainRain = generate_info(trainval_nusc, train_scenes)
    print('Number of Rain Samples in training set: %i', (len(trainRain)))
    mmcv.dump(train_infos, '../data/nuScenes/nuscenes_infos_train.pkl')
    mmcv.dump(trainRain, '../data/nuScenes/nuscenes_rainIdx_train.pkl')
    print('Starting Val...')
    val_infos, valRain = generate_info(trainval_nusc, val_scenes)
    print('Number of Rain Samples in validation set: %i', (len(valRain)))
    mmcv.dump(val_infos, '../data/nuScenes/nuscenes_infos_val.pkl')
    mmcv.dump(valRain, '../data/nuScenes/nuscenes_rainIdx_val.pkl')

    print('Starting TrainRain...')
    train_infos, trainRain = generate_info(trainval_nusc, train_scenes, rainOnly = True)
    print('Number of Rain Samples in training set: %i', (len(trainRain)))
    mmcv.dump(train_infos, '../data/nuScenes/nuscenes_infos_trainRain.pkl')
    print('Starting ValRain...')
    val_infos, valRain = generate_info(trainval_nusc, val_scenes, rainOnly = True)
    print('Number of Rain Samples in validation set: %i', (len(valRain)))
    mmcv.dump(val_infos, '../data/nuScenes/nuscenes_infos_valRain.pkl')
   

    # test_nusc = NuScenes(version='v1.0-test',
    #                      dataroot='./data/nuScenes/v1.0-test/',
    #                      verbose=True)
    # test_scenes = splits.test
    # test_infos = generate_info(test_nusc, test_scenes)
    # mmcv.dump(test_infos, './data/nuScenes/nuscenes_infos_test.pkl')


if __name__ == '__main__':
    main()
