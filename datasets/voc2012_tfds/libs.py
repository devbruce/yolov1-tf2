import cv2
import numpy as np
import tensorflow as tf


__all__ = ['VOC_CLS_MAP', 'normalize_img', 'trim_img_zero_pad', 'prep_voc_data', 'viz_voc_prep']


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


def normalize_img(data):
    imgs_normalized = tf.cast(data['image'], dtype=tf.float32) / 255.0
    ret = {
        'imgs': imgs_normalized,
        'labels': data['objects']
    }
    return ret


def trim_img_zero_pad(arr):
    non_zero_idx_ranges =  map(lambda e: range(e.min(), e.max()+1), np.where(arr != 0))
    mesh = np.ix_(*non_zero_idx_ranges)
    return arr[mesh]


def prep_voc_data(batch_data, input_height, input_width):
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
        y_min_rel, x_min_rel = pts[:, 0], pts[:, 1]
        y_max_rel, x_max_rel = pts[:, 2], pts[:, 3]
        
        cx_rel = (x_min_rel + x_max_rel) / 2
        cy_rel = (y_min_rel + y_max_rel) / 2
        w_rel = (x_max_rel - x_min_rel)
        h_rel = (y_max_rel - y_min_rel)
        
        label_preps = np.array([cx_rel, cy_rel, w_rel, h_rel, cls_idx], dtype=np.float32).T
        label_preps = label_preps[np.where(np.all(label_preps, axis=1) == True)]  # Filter dummy data by padded batch
        label_preps = tf.convert_to_tensor(label_preps, dtype=tf.float32)
        batch_labels.append(label_preps)
        
    batch_imgs = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
    return batch_imgs, batch_labels


def viz_voc_prep(batch_data, idx, input_height, input_width, box_color=(0, 1, 0), thickness=1, txt_color=(1, 0, 0)):
    imgs, labels = prep_voc_data(batch_data, input_height, input_width)
    img = imgs[idx].numpy().copy()
    label = labels[idx].numpy()
    
    for pts in label:
        cx_rel, cy_rel, w_rel, h_rel, cls_idx = pts
        cls_name = VOC_CLS_MAP[cls_idx]
        xmin_rel, ymin_rel = cx_rel - (w_rel / 2), cy_rel - (h_rel / 2)
        xmax_rel, ymax_rel = cx_rel + (w_rel / 2), cy_rel + (h_rel / 2)
        xmin, ymin = round(xmin_rel * input_width), round(ymin_rel * input_height)
        xmax, ymax = round(xmax_rel * input_width), round(ymax_rel * input_height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, thickness)
        cv2.putText(img, cls_name, (xmin, ymin), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=txt_color)
    return img
