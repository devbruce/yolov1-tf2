import cv2
import numpy as np
from .libs import VOC_CLS_MAP, trim_img_zero_pad, prep_voc_data


__all__ = ['viz_voc_origin', 'viz_voc_prep']


def viz_voc_origin(batch_data, idx, box_color=(0, 255, 0), thickness=1, txt_color=(255, 0, 0)):
    imgs, labels = batch_data['imgs'].numpy(), batch_data['labels']
    img = imgs[idx].copy()
    img = (img * 255).astype(np.uint8)
    img_no_pad = trim_img_zero_pad(img)
    height, width, _ = img_no_pad.shape

    cls_idxs = labels['label'][idx].numpy()
    pts_rel = labels['bbox'][idx].numpy()
    y_min, x_min = pts_rel[:, 0] * height, pts_rel[:, 1] * width
    y_max, x_max = pts_rel[:, 2] * height, pts_rel[:, 3] * width
    pts_abs = np.array([x_min, y_min, x_max, y_max, cls_idxs], dtype=np.float32).T
    pts_abs = np.around(pts_abs)

    for pts in pts_abs:
        x_min, y_min, x_max, y_max, cls_idx = pts
        cls_name = VOC_CLS_MAP[cls_idx]
        cv2.rectangle(img_no_pad, (x_min, y_min), (x_max, y_max), box_color, thickness)
        cv2.putText(img_no_pad, cls_name, (x_min, y_min), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=txt_color)
    return img_no_pad


def viz_voc_prep(batch_data, idx, input_height, input_width, box_color=(0, 255, 0), thickness=1, txt_color=(255, 0, 0)):
    imgs, labels = prep_voc_data(batch_data, input_height, input_width)
    img = imgs[idx].numpy().copy()
    img = (img * 255).astype(np.uint8)
    label = labels[idx].numpy()
    
    for pts in label:
        cx, cy, w, h, cls_idx = pts
        cls_name = VOC_CLS_MAP[cls_idx]
        xmin, ymin = cx - (w / 2), cy - (h / 2)
        xmax, ymax = cx + (w / 2), cy + (h / 2)
        xmin, ymin, xmax, ymax = map(round, [xmin, ymin, xmax, ymax])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, thickness)
        cv2.putText(img, cls_name, (xmin, ymin), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=txt_color)
    return img
