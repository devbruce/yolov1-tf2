import tensorflow as tf
import numpy as np
from libs.iou import calc_iou
from libs.utils import postprocess_yolo_format


__all__ = ['get_losses', 'get_batch_losses', 'train_step']


def get_losses(one_pred, one_label, cfg):
    """
    Args:
      one_pred (EagerTensor dtype=tf.float32): 3-D Tensor shape=[cell_size, cell_size, boxes_per_cell*5 + num_classes]
      one_label (EagerTensor dtype=tf.float32): 2-D Tensor shape=[obj_n, 5], 5 --> [x_center_in_cell, y_center_in_cell, width, height, class_index] (Absolute coordinates)
      cfg: YOLO config object

    Returns:
      dict: total_loss, coord_loss, obj_loss, noobj_loss, class_loss
    """
    pred_boxes_with_confidence = one_pred[:, :, :cfg.boxes_per_cell*5]  # 5 ==> [x_center, y_center, width, height, confidence]
    pred_boxes_with_confidence = tf.reshape(pred_boxes_with_confidence, [cfg.cell_size, cfg.cell_size, cfg.boxes_per_cell, 5])
    pred_boxes = pred_boxes_with_confidence[:, :, :, :4]
    pred_confidences = pred_boxes_with_confidence[:, :, :, 4]
    pred_cell_classes = one_pred[:, :, cfg.boxes_per_cell*5:]
    
    pred_cx_in_cell = pred_boxes[:, :, :, 0]
    pred_cy_in_cell = pred_boxes[:, :, :, 1]
    pred_w_sqrt = pred_boxes[:, :, :, 2]
    pred_h_sqrt = pred_boxes[:, :, :, 3]
    cell_w = cfg.input_width // cfg.cell_size
    cell_h = cfg.input_height // cfg.cell_size

    # Loop the number of objects
    losses = {
        'total_loss': 0.,
        'coord_loss': 0.,
        'obj_loss': 0.,
        'noobj_loss': 0.,
        'class_loss': 0.,
    }
    obj_n = len(one_label)
    for obj_idx in range(obj_n):
        label = one_label[obj_idx]

        # Label Info
        cls_idx = tf.cast(label[4], tf.int32)
        cx, cy, w, h = label[0], label[1], label[2], label[3]
        cx = tf.minimum(cx, cfg.input_width - cfg.eps)
        cy = tf.minimum(cy, cfg.input_height - cfg.eps)
        cx_in_cell = cx / cfg.cell_size
        cy_in_cell = cy / cfg.cell_size
        w_sqrt, h_sqrt = tf.sqrt(w), tf.sqrt(h)

        x_grid_idx = cx // cell_w
        y_grid_idx = cy // cell_h
        # Responsible cell mask
        # ==> If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object
        # (Refer to page2 2.Unified Detection of papar)
        responsible_cell_mask = np.zeros([cfg.cell_size, cfg.cell_size, 1])
        responsible_cell_mask[y_grid_idx, x_grid_idx] = 1

        #===========#===========#===========#===========#===========#===========#===========#===========#===========
        
        # Resonsible box mask (Max IoU per cell)
        pred_ltrb = postprocess_yolo_format(pred_boxes_with_confidence, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
        ious = calc_iou(pred_ltrb, label[:-1], cfg)

        max_ious = tf.reduce_max(ious, axis=2, keepdims=True)  # shape = [cell_size, cell_size, 1]
        resonsible_box_mask = tf.cast(tf.greater_equal(ious, max_ious), tf.float32)  # shape = [cell_size, cell_size, boxes_per_cell]

        # Responsible mask
        responsible_mask = responsible_cell_mask * resonsible_box_mask

        # Coordinate Loss
        cx_loss = tf.reduce_sum(tf.square(responsible_mask * (pred_cx_in_cell - cx_rel_in_cell)))
        cy_loss = tf.reduce_sum(tf.square(responsible_mask * (pred_cy_in_cell - cy_rel_in_cell)))
        sqrt_w_loss = tf.reduce_sum(tf.square(responsible_mask * (pred_w_sqrt - w_sqrt)))
        sqrt_h_loss = tf.reduce_sum(tf.square(responsible_mask * (pred_h_sqrt - h_sqrt)))
        coord_loss = cfg.lambda_coord * (cx_loss + cy_loss + sqrt_w_loss + sqrt_h_loss)

        # Objectness Loss
        # ==> ious: Objectness Label
        # For-mally we define confidence as Pr(Object) * IoU
        # (Refer to page2 2.Unified Detection of papar)
        obj_loss = cfg.lambda_obj * tf.reduce_sum(tf.square(responsible_mask * (pred_confidences - ious)))

        # Non-objectness Loss
        noobj_mask = 1. - responsible_mask
        responsible_noobj_n = len(noobj_mask[noobj_mask == 1.])
        noobj_loss = cfg.lambda_noobj * (tf.reduce_sum(tf.square(noobj_mask * (pred_confidences - 0.))) / responsible_noobj_n)
        
        # Class Loss
        cls_one_hot = tf.one_hot(cls_idx, cfg.num_classes, dtype=tf.float32)
        class_loss = cfg.lambda_class * tf.reduce_sum(tf.square(responsible_cell_mask * (pred_cell_classes - cls_one_hot)))

        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        losses['total_loss'] += total_loss
        losses['coord_loss'] += coord_loss
        losses['obj_loss'] += obj_loss
        losses['noobj_loss'] += noobj_loss
        losses['class_loss'] += class_loss
    return losses


def get_batch_losses(model, batch_imgs, batch_labels, cfg):
    batch_losses = {
        'total_loss': 0.,
        'coord_loss': 0.,
        'obj_loss': 0.,
        'noobj_loss': 0.,
        'class_loss': 0.,
    }
    preds = model(batch_imgs)
    # preds shape: [batch_size, cfg.cell_size * cfg.cell_size, cfg.boxes_per_cell*5 + cfg.num_classes]
    for i in range(len(preds)):
        one_loss = get_losses(one_pred=preds[i], one_label=batch_labels[i], cfg=cfg)
        batch_losses['total_loss'] += one_loss['total_loss']
        batch_losses['coord_loss'] += one_loss['coord_loss']
        batch_losses['obj_loss'] += one_loss['obj_loss']
        batch_losses['noobj_loss'] += one_loss['noobj_loss']
        batch_losses['class_loss'] += one_loss['class_loss']
    return batch_losses


def train_step(model, optimizer, batch_imgs, batch_labels, cfg):
    with tf.GradientTape() as tape:
        batch_losses = get_batch_losses(model, batch_imgs, batch_labels, cfg)
    grads = tape.gradient(batch_losses['total_loss'], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Average with batch size
    batch_losses['total_loss'] /= len(batch_imgs)
    batch_losses['coord_loss'] /= len(batch_imgs)
    batch_losses['obj_loss'] /= len(batch_imgs)
    batch_losses['noobj_loss'] /= len(batch_imgs)
    batch_losses['class_loss'] /= len(batch_imgs)
    return batch_losses
