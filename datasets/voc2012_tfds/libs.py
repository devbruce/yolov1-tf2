import cv2
import numpy as np
import tensorflow as tf


__all__ = ['VOC_CLS_MAP', 'trim_img_zero_pad', 'prep_voc_data']


VOC_CLS_MAP = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor',
}


def trim_img_zero_pad(arr):
    non_zero_idx_ranges =  map(lambda e: range(e.min(), e.max()+1), np.where(arr != 0))
    mesh = np.ix_(*non_zero_idx_ranges)
    return arr[mesh]


def prep_voc_data(batch_data, input_height, input_width):
    """
    Returns:
      tuple: (batch_imgs, batch_labels)
        batch_imgs (EagerTensor dtype=tf.float32)
        batch_labels (EagerTensor dtype=tf.float32 in list)-> [center_x, center_y, width, height, cls_idx] (Absolute coordinates)
    """
    imgs, labels = batch_data['imgs'].numpy(), batch_data['labels']
    num_batch = len(imgs)
    batch_labels = list()
    batch_imgs = np.empty((num_batch, input_height, input_width, 3), dtype=np.float32)
    for i in range(num_batch):
        # Image resize
        origin_size_img = trim_img_zero_pad(imgs[i])
        img_resized = cv2.resize(origin_size_img, dsize=(input_height, input_width))
        batch_imgs[i] = img_resized
        
        # Class Indices
        cls_idx = labels['label'][i].numpy()
        
        # Sync coordinates with resized image
        origin_height, origin_width, _ = origin_size_img.shape
        
        pts = labels['bbox'][i].numpy()
        y_min, x_min = pts[:, 0] * input_height, pts[:, 1] * input_width
        y_max, x_max = pts[:, 2] * input_height, pts[:, 3] * input_width
        
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min)
        h = (y_max - y_min)
        
        label_preps = np.array([cx, cy, w, h, cls_idx], dtype=np.float32).T
        label_preps = label_preps[np.where(np.any(label_preps[:, :4], axis=1) == True)]  # Filter dummy data by padded batch
        label_preps = tf.convert_to_tensor(label_preps, dtype=tf.float32)
        batch_labels.append(label_preps)
        
    batch_imgs = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
    return batch_imgs, batch_labels
