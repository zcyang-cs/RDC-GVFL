# RDC-GVFL (PRCV-2023)
![image](https://github.com/zcyang-cs/RDC-GVFL/blob/main/figure/RDC-GVFL.png)

This repository is PyTorch implementation for the PRCV 2023 paper RDC-GVFL: A Robust Detection and Correction Framework for GNN-based Vertical Federated Learning.

The supplemental material is also available in this repository.

## Data 
Here we provide the source code of data processing. And the Raw Data can be found in:  
* Cora Dataset: https://github.com/kimiyoung/planetoid/tree/master/data.  
* Cora_ML Dataset: https://github.com/danielzuegner/gnn-meta-attack/blob/master/data/cora_ml.npz.  
* Pubmed Dataset: https://github.com/kimiyoung/planetoid/tree/master/data. 

## Requirements 
>Python 3.9.16,  
>pytorch 1.12.0,  
>torch-geometric 2.3.1,  
>torch-scatter 2.1.0,  
>torch-sparse 0.6.16,  
>numpy 1.24.3,  
>scipy 1.10.1,  
>tqdm 4.65.0  

## Contact
Zhicheng Yang, zcyang@stu.xmu.edu.cn

Xiaoliang Fan (corresponding author), https://xiaoliangfan.github.io

## Citation
Please cite our paper in your publications if this code helps your research.
```
@inproceedings{yang2023robust,
  title={A Robust Detection and Correction Framework for GNN-Based Vertical Federated Learning},
  author={Yang, Zhicheng and Fan, Xiaoliang and Wang, Zheng and Wang, Zihui and Wang, Cheng},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={97--108},
  year={2023},
  organization={Springer}
}



```
