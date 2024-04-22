# RDC-GVFL (PRCV-2023)
![image](https://github.com/zcyang-cs/RDC-GVFL/blob/main/figure/RDC-GVFL.png)

This repository is PyTorch implementation for the PRCV 2023 paper RDC-GVFL: A Robust Detection and Correction Framework for GNN-based Vertical Federated Learning.

The supplemental material is also available in this repository.

## Data 
Here we provide the processed data. And the Raw Data can be found in: NYC OpenData: https://opendata.cityofnewyork.us/.  
We followed the settings in [[Zhang et al., 2020]](#R1) that 
apply taxi trip data as human mobility data and take the crime count, check-in count, land usage type as prediction tasks, respectively.  
For New York City zoning dataset, you can contact with Xu Yan(yanxu97@stu.xmu.edu.cn) or Shangbin Wu(shangbin@stu.xmu.edu.cn).

## Requirements 
>Python 3.7.9,   
>pytorch 1.5.1,  
>numpy 1.19.2,  
>pandas 0.25.3,  
>sklearn 0.24.1,  
>geopandas 0.13.0,  
>shapely 2.0.1  

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
