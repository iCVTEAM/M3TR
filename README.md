# M3TR
Pytorch implementation of [M3TR: Multi-modal Multi-label Recognition with Transformer](https://dl.acm.org/doi/abs/10.1145/3474085.3475191). ACM MM 2021

![M3TR](https://github.com/iCVTEAM/M3TR/blob/master/figs/motivation.png)

## Prerequisites

Python 3.6+

Pytorch 1.7

CUDA 10.1

Tesla V100 × 4

## Datasets

- MS-COCO: [train](http://images.cocodataset.org/zips/train2014.zip)  [val](http://images.cocodataset.org/zips/val2014.zip)  [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- VOC 2007: [trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)  [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)  [test_anno](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar)

## Train

```shell
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 3e-4 -b 64
```

## Test

```shell
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 3e-4 -b 64 -e --resume checkpoint/COCO2014/checkpoint_COCO.pth
```

## Citation

- If you find this work is helpful, please cite our paper

```
@inproceedings{Zhao2021M3TR,
author = {Zhao, Jiawei and Zhao, Yifan and Li, Jia},
title = {M3TR: Multi-Modal Multi-Label Recognition with Transformer},
year = {2021},
address = {New York, NY, USA},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {469–477},
}
```
