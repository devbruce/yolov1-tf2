import numpy as np
import tensorflow as tf
from datasets.voc_tfds.augs import get_transform


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


def prep_voc_data(batch_data, input_height, input_width, val):
    batch_images = batch_data['image'].numpy()
    batch_bboxes = batch_data['objects']['bbox'].numpy()
    batch_class_indices = batch_data['objects']['label'].numpy()

    batch_images_prep = list()
    batch_label_list = list()
    for i in range(len(batch_images)):
        # Image preprocessing
        padded_img = batch_images[i]
        img = trim_img_zero_pad(padded_img).astype(np.float32) / 255.
        img_height, img_width, _ = img.shape

        # Box coordinates preprocessing
        bboxes = batch_bboxes[i].astype(np.float32)
        zero_pad_filter = np.any(bboxes, axis=1)
        bboxes = bboxes[zero_pad_filter]

        y_min, x_min, y_max, x_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        bboxes = np.array([x_min, y_min, x_max, y_max], dtype=np.float32).T

        # Class Indices preprocessing
        class_indices = batch_class_indices[i].astype(np.float32)
        class_indices = class_indices[zero_pad_filter]

        # Augmentation
        transform = get_transform(img_height, img_width, input_height=input_height, input_width=input_width, val=val)
        transformed = transform(image=img, bboxes=bboxes, class_indices=class_indices)
        transformed_image = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_class_indices = np.array(transformed['class_indices'], dtype=np.float32)

        if transformed_bboxes.size == 0:
            continue

        # Convert box coordinates [x_min, y_min, x_max, y_max] to [cx, cy, w, h]
        x_min, y_min = transformed_bboxes[:, 0], transformed_bboxes[:, 1]
        x_max, y_max = transformed_bboxes[:, 2], transformed_bboxes[:, 3]
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = (x_max - x_min), (y_max - y_min)
        transformed_labels = np.array([cx, cy, w, h, transformed_class_indices], dtype=np.float32).T
        transformed_labels = tf.convert_to_tensor(transformed_labels, dtype=tf.float32)

        # Append result
        batch_images_prep.append(transformed_image)
        batch_label_list.append(transformed_labels)
        
    batch_images_prep = np.stack(batch_images_prep, axis=0)
    batch_images_prep = tf.convert_to_tensor(batch_images_prep, dtype=tf.float32)
    return batch_images_prep, batch_label_list
    