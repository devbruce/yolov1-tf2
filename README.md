# YOLO with Tensorflow 2

![tf-v2.4.1](https://img.shields.io/badge/TensorFlow-v2.4.1-orange)

For ease of implementation, i have not implemented exactly the same as paper.  
The things presented below are implemented differently from the paper.

- Backbone network. (I used Xception instead of network mentioned in paper.)

- I used Global Average Pooling instead of Fully Connected Layer.  
So there is a no Dropout layer for regularization.

- Learning Rate Schedule

- Hyper Parameters

- Data Augmentations

- And so on . . .

<br><br>

## Preview

### Prediction Visualization

<br>

### Tensorboard

<br><br>

## Build Environment with Docker

### Build Docker Image

```bash
$ docker build -t ${NAME}:${TAG} .
```

### Create a Container

```bash
$ docker run -d -it --gpus all --shm-size=${PROPER_VALUE} ${NAME}:${TAG} /bin/bash
```

<br><br>

## Support Dataset to Training

- **PascalVOC 2012** with [TFDS](https://www.tensorflow.org/datasets/overview) (Training Script: [train_voc2012.py](./train_scripts/train_voc2012.py))

```bash
$ python train_voc2012.py
```

**Options**  

Default option value is [configs.py](./configs/configs.py).  
If the option is given, the default config value is overridden.  

- `--epochs`: Number of training epochs
- `--init_lr`: Initial learning rate
- `--batch_size`
- `--val_step`: Validation interval during training
- `--tb_img_max_outputs `: Number of visualized prediction images in tensorboard
- `--val_sample_num`: Validation sampling. 0 means use all validation set (Recommend to use default 0)

<br><br>

## Citation

**You Only Look Once: Unified, Real-Time Object Detection** \<[arxiv link](https://arxiv.org/abs/1506.02640)\>

```
@misc{redmon2016look,
      title={You Only Look Once: Unified, Real-Time Object Detection}, 
      author={Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
      year={2016},
      eprint={1506.02640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
