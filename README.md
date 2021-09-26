# M3TR
M3TR: Multi-modal Multi-label Recognition with Transformer. ACM MM 2021

### train
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 3e-4 -b 64

### test
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 3e-4 -b 64 -e --resume checkpoint/COCO2014/checkpoint_COCO.pth

### dataset
MS-COCO 2014 
VOC 2007

### checkpoint & result
* Loading checkpoint 'checkpoint/COCO2014/checkpoint_COCO.pth'
tensor([0.9831, 0.7950, 0.5944, 0.9139, 0.9772, 0.9708, 0.9909, 0.9342, 0.7536,
        0.8447, 0.8667, 0.9403, 0.7628, 0.7849, 0.8051, 0.9492, 0.9165, 0.8942,
        0.9158, 0.8825, 0.9719, 0.7355, 0.8586, 0.8528, 0.9009, 0.9556, 0.8103,
        0.8393, 0.9352, 0.9157, 0.9881, 0.8846, 0.8346, 0.9751, 0.9980, 0.5698,
        0.6222, 0.9637, 0.8666, 0.9052, 0.9886, 0.7663, 0.9284, 0.8534, 0.9577,
        0.9049, 0.8601, 0.9185, 0.7534, 0.9935, 0.9675, 0.7574, 0.8752, 0.8574,
        0.8476, 0.7514, 0.9771, 0.9389, 0.9857, 0.9622, 0.9103, 0.7066, 0.9185,
        0.8331, 0.8383, 0.9813, 0.9013, 0.9949, 0.8864, 0.3405, 0.9862, 0.8302,
        0.8912, 0.9808, 0.7976, 0.9098, 0.8879, 0.8308, 0.8379, 0.9966])
* Test
Loss: 0.1515     mAP: 0.8746    Data_time: 0.0036        Batch_time: 2.3486
OP: 0.881        OR: 0.797       OF1: 0.837     CP: 0.880        CR: 0.774       CF1: 0.823
OP_3: 0.926      OR_3: 0.695     OF1_3: 0.794   CP_3: 0.917      CR_3: 0.682     CF1_3: 0.782
