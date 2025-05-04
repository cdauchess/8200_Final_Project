# CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception

Forked from original CRN work: https://github.com/youngskkim/CRN


## Getting Started

### Installation
```shell
# clone repo
git clone https://github.com/youngskkim/CRN.git

cd CRN

# setup conda environment
conda env create --file CRN.yaml
conda activate CRN

# install dependencies
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
mim install mmcv==1.6.0
mim install mmsegmentation==0.28.0
mim install mmdet==2.25.2

cd mmdetection3d
pip install -v -e .
cd ..

python setup.py develop  # GPU required
```

### Data preparation
**Step 0.** Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

**Step 1.** Symlink the dataset folder to `./data/nuScenes/`.
```
ln -s [nuscenes root] ./data/nuScenes/
```

**Step 2.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
```
python scripts/gen_info.py
```

**Step 3.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
```
python scripts/gen_depth_gt.py
```

**Step 4.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```

The folder structure will be as follows:
```
CRN
├── data
│   ├── nuScenes
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
|   |   ├── depth_gt
|   |   ├── radar_bev_filter  # temporary folder, safe to delete
|   |   ├── radar_pv_filter
|   |   ├── v1.0-trainval
```

### Training and Evaluation
**Training**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```


### Image Cleansing

* To cleanse with DeRaindrop see batch file /DeRaindrop/batchDerainDrop.sh
* To cleanse and evaluate with CRN with Transweather see batch file /new-Transweather/TWeather_cleanse.sh
* To cleanse and evaluate with both UtilityIR settings see batch file /utilityIR/utilityIR_Palmetto/UtilIR_NS_Conv.sh

### CRN Evaluate of Cleansed Images
A new experiment is made for each cleansing method. The experiment files are located at the following directory: /exps/det/

The nuScenes dataloader was modified with an additional field to indicate the image cleansing to be used. 

The options are as follows:
* None - loads the stock nuScenes images
* 'DRD' - loads the DeRaindrop cleansed images
* 'UtilIR1' - loads the UtilityIR1 cleansed images
* 'UtilIR2' - loads the UtilityIR2 cleansed images
* 'TransW' - loads the TransWeather cleansed images

