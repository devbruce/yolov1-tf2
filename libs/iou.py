import tensorflow as tf


__all__ = ['calc_iou']


@tf.function
def calc_iou(pred_boxes, gt_box, cfg):
    """Calculate ious between pred_boxes and gt_box
    Args:
      pred_boxes (EagerTensor dtype=tf.float32): 4-D tensor [cell_size, cell_size, boxes_per_cell, 5] ==> [x_min, y_min, x_max, y_max, confidence] (Absolute coordinates)
      gt_box (EagerTensor dtype=tf.float32): 1-D tensor [4] ==> [x_center_rel, y_center_rel, width_rel, height_rel]
      cfg: YOLO config object
      
    Returns:
      iou (EagerTensor dtype=tf.float32): 3-D tensor [cell_size, cell_size, boxes_per_cell]
    """
    pred_xmin, pred_ymin = pred_boxes[:, :, :, 0], pred_boxes[:, :, :, 1]
    pred_xmax, pred_ymax = pred_boxes[:, :, :, 2], pred_boxes[:, :, :, 3]
    pred_w = pred_xmax - pred_xmin
    pred_h = pred_ymax - pred_ymin

    gt_cx_rel, gt_cy_rel, gt_w_rel, gt_h_rel = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
    gt_cx = gt_cx_rel * cfg.input_width
    gt_cy = gt_cy_rel * cfg.input_height
    gt_w = gt_w_rel * cfg.input_width
    gt_h = gt_h_rel * cfg.input_height
    gt_xmin, gt_ymin = gt_cx - (gt_w / 2), gt_cy - (gt_h / 2)
    gt_xmax, gt_ymax = gt_cx + (gt_w / 2), gt_cy + (gt_h / 2)

    inter_xmin, inter_ymin = tf.maximum(pred_xmin, gt_xmin), tf.maximum(pred_ymin, gt_ymin)
    inter_xmax, inter_ymax = tf.minimum(pred_xmax, gt_xmax), tf.minimum(pred_ymax, gt_ymax)
    inter_w = inter_xmax - inter_xmin
    inter_h = inter_ymax - inter_ymin
    inter_area = inter_w * inter_h
    
    # Mask whether intersection width > 0 and height > 0
    inter_mask = tf.cast(tf.greater_equal(inter_w, 0), tf.float32) * tf.cast(tf.greater_equal(inter_h, 0), tf.float32)
    
    # Intersection area
    inter_area = inter_area * inter_mask

    # Calculate Iou with Intersection and Union
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    union = (pred_area + gt_area) - inter_area + cfg.eps
    ious = inter_area / union
    return ious
