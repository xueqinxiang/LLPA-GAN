# LLPA-GAN
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/tobran/DF-GAN/blob/master/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.9.0-red.svg)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
# LLPA-GAN: Lightweight Large Pretrained Models Aided Generative Adversarial Networks for Text-to-Image Synthesis
Official Pytorch implementation for our paper [LLPA-GAN: Lightweight Large Pretrained Models Aided
Generative Adversarial Networks for Text-to-Image Synthesis]

# Framework 
<img src="frame.jpg" width="3259px" height="2217px"/>

# Samples
<img src="results.jpg" width="5671px" height="2049px"/>

## Requirements
- python 3.8
- Pytorch 1.9
- At least 1x12GB NVIDIA GPU
## Installation

Clone this repo.
```
git clone https://github.com/xueqinxiang/LLPA-GAN.git
pip install -r requirements.txt
cd LLPA-GAN/code/
```

## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`


## Training
  ```
  cd LLPA-GAN/code/
  ```
### Train the DE-Net model
  - For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/train.sh ./cfg/coco.yml`
  - For cc3m dataset: `bash scripts/train.sh ./cfg/cc3m.yml`
### Resume training process
If your training process is interrupted unexpectedly, set **resume_epoch** and **resume_model_path** in train.sh to resume training.
