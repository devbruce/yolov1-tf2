import cv2
import numpy as np
import tensorflow as tf
from libs.utils import yolo_output2boxes, box_postp2use
from datasets.voc_tfds.libs import VOC_CLS_MAP
from configs import cfg


__all__ = ['get_model', 'get_pred', 'get_pred_viz']


def get_model(pb_dir_path):
    yolo = tf.saved_model.load(export_dir=pb_dir_path, tags=None, options=None)
    return yolo


def get_pred(img, model, nms_iou_thr=0.5, conf_thr=0.5, cfg=cfg):
    img = img.copy().astype(np.float32) / 255.
    img = cv2.resize(img, dsize=(cfg.input_height, cfg.input_width))
    img = np.expand_dims(img, axis=0)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    yolo_output_raw = model(img, training=False)
    yolo_boxes = yolo_output2boxes(yolo_output_raw, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
    yolo_boxes_postprep = box_postp2use(yolo_boxes[0], nms_iou_thr=nms_iou_thr, conf_thr=conf_thr)
    return yolo_boxes_postprep


def get_pred_viz(img, model, nms_iou_thr=0.5, conf_thr=0.5, cfg=cfg):
    img_with_pred = img.copy()
    origin_img_height, origin_img_width, _ = img_with_pred.shape

    preds = get_pred(img=img, model=model, nms_iou_thr=nms_iou_thr, conf_thr=conf_thr, cfg=cfg)
    for pred in preds:
        x_min, y_min, x_max, y_max, conf, cls_idx = pred
        x_min, y_min = x_min / cfg.input_width, y_min / cfg.input_height
        x_max, y_max = x_max / cfg.input_width, y_max / cfg.input_height

        x_min, y_min = round(x_min * origin_img_width), round(y_min * origin_img_height)
        x_max, y_max = round(x_max * origin_img_width), round(y_max * origin_img_height)

        cls_name = VOC_CLS_MAP[cls_idx]
        cv2.rectangle(img_with_pred, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            img=img_with_pred,
            text=f'{cls_name}: {conf:.2f}',
            org=(x_min, y_min),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
        )
    return img_with_pred
