# 基于强化学习的图像分割实例级反思方法

## &#x1F3AC; Getting Started

### :one: Download data

#### Pre-processed data from drive

We use a [adapted version](https://zenodo.org/records/10828417) of OpenEarthMap datasets. You can download the full .zip and directly extract it in the `data/` folder.

#### From scratch

Alternatively, you can prepare the datasets yourself. Here is the structure of the data folder for you to reproduce:

```
data
├── trainset
│   ├── images
│   └── labels
|
└── train.txt
```

### :two: Download pre-trained models

#### Pre-trained backbone and models
We use ConvNext_large pre-trained using CLIP as backbone. You can download the weight [here](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main) and move it to `pretrain/`.

## &#x1F5FA; Overview of the repo

Data are located in `data/` contains the train dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## &#x2699; Training 

监督学习训练过程
```bash
python train.py 
```
强化学习训练过程
```bash
python train_GNN_RL.py 
```


















