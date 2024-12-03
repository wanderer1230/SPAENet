# Spatial Pyramid Attention and Affinity Inference Embedding for Unsupervised Person Re-identification



This repository contains the implementation of [Spatial Pyramid Attention and Affinity Inference Embedding for Unsupervised Person Re-identification], which improves the previous unsupervised method [O2CAP].


## Requirements

### Environment
Python >= 3.6

PyTorch >= 1.1


```

### Prepare Datasets
Download the person datasets DukeMTMC-reID, Market-1501 and MSMT17. Then put them under a folder such as '/folder/to/dataset/'.


## Training

We utilize 1 GPU for training.

To train the model in the paper, run this command (example: train on MSMT17):
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir '/folder/to/dataset/' --dataset 'MSMT17' --logs_dir 'MSMT_logs'
```



## Citation
Our code is referenced o2cap
```
@article{2022_o2cap,
    title={Offline-Online Associated Camera-Aware Proxies for Unsupervised Person Re-identification},
    author={Menglin Wang and Jiachen Li and Baisheng Lai and Xiaojin Gong and Xian-Sheng Hua},
    journal={IEEE Transactions on Image Processing},
    year={2022}
}
```
