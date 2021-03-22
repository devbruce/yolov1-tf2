# YOLO with Tensorflow 2

![tf-v2.4.1](https://img.shields.io/badge/TensorFlow-v2.4.1-orange)

For ease of implementation, i have not implemented exactly the same as paper.  
(Such as backbone network, learning rate scheduler, hyper paramters and etc ...)

<br>

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
