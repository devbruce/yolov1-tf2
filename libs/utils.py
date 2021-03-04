import tensorflow as tf
import numpy as np


__all__ = ['postprocess_yolo_format', 'trim_img_zero_pad']


def postprocess_yolo_format(yolo_pred_boxes, cfg):
    """Postprocess yolo_pred_boxes
    Args:
      yolo_pred_boxes (EagerTensor dtype=tf.float32): 4-D tensor [cell_size, cell_size, boxes_per_cell, 5] ==> [cx_rel_in_cell, cy_rel_in_cell, w_rel, h_rel, confidence]
      cfg: YOLO config object

    Returns:
      EagerTensor (dtype=tf.float32): shape:[n, 5] ==> [x_min, y_min, x_max, y_max, confidence] (Absolute coordinates)
    """
    cell_w, cell_h = cfg.input_width / cfg.cell_size, cfg.input_height / cfg.cell_size
    ret = np.zeros(tf.shape(yolo_pred_boxes).numpy()).astype(np.float32)

    for y_grid_idx in range(cfg.cell_size):
        for x_grid_idx in range(cfg.cell_size):
            cell = yolo_pred_boxes[y_grid_idx, x_grid_idx]

            for i in range(cfg.boxes_per_cell):
                box = cell[i]
                cx_rel_in_cell, cy_rel_in_cell, w_rel, h_rel, confidence = box[0], box[1], box[2], box[3], box[4]
                cx_in_cell = cx_rel_in_cell * cell_w
                cy_in_cell = cy_rel_in_cell * cell_h
                cx = cx_in_cell + (x_grid_idx * cell_w)
                cy = cy_in_cell + (y_grid_idx * cell_h)
                w = w_rel * cfg.input_width
                h = h_rel * cfg.input_height

                x_min, y_min = cx - (w / 2), cy - (h / 2)
                x_max, y_max = cx + (w / 2), cy + (h / 2)
                
                ret[y_grid_idx, x_grid_idx, i] = np.array([x_min, y_min, x_max, y_max, confidence], dtype=np.float32)
    return tf.convert_to_tensor(ret, dtype=tf.float32)


def trim_img_zero_pad(arr):
    non_zero_idx_ranges =  map(lambda e: range(e.min(), e.max()+1), np.where(arr != 0))
    mesh = np.ix_(*non_zero_idx_ranges)
    return arr[mesh]
    