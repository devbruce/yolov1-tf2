import numpy as np
import tensorflow as tf


__all__ = ['postprocess_yolo_format', 'nms']


def postprocess_yolo_format(yolo_pred_boxes, input_height, input_width, cell_size, boxes_per_cell):
    """Postprocess yolo_pred_boxes
    Args:
      yolo_pred_boxes (EagerTensor dtype=tf.float32): 4-D tensor [cell_size, cell_size, boxes_per_cell, 5] ==> [cx_rel_in_cell, cy_rel_in_cell, w_rel, h_rel, confidence]
      input_height (int): Height of input image
      input_width (int): Width of input image
      cell_size (int): cell_size of YOLO Model
      boxes_per_cell (int): boxes_per_cell of YOLO Model

    Returns:
      EagerTensor (dtype=tf.float32): shape:[n, 5] ==> [x_min, y_min, x_max, y_max, confidence] (Absolute coordinates)
    """
    cell_w, cell_h = input_width / cell_size, input_height / cell_size
    ret = np.zeros(tf.shape(yolo_pred_boxes).numpy()).astype(np.float32)

    for y_grid_idx in range(cell_size):
        for x_grid_idx in range(cell_size):
            cell = yolo_pred_boxes[y_grid_idx, x_grid_idx]

            for i in range(boxes_per_cell):
                box = cell[i]
                cx_rel_in_cell, cy_rel_in_cell, w_rel, h_rel, confidence = box[0], box[1], box[2], box[3], box[4]
                cx_in_cell = cx_rel_in_cell * cell_w
                cy_in_cell = cy_rel_in_cell * cell_h
                cx = cx_in_cell + (x_grid_idx * cell_w)
                cy = cy_in_cell + (y_grid_idx * cell_h)
                w = w_rel * input_width
                h = h_rel * input_height

                x_min, y_min = cx - (w / 2), cy - (h / 2)
                x_max, y_max = cx + (w / 2), cy + (h / 2)
                
                ret[y_grid_idx, x_grid_idx, i] = np.array([x_min, y_min, x_max, y_max, confidence], dtype=np.float32)
    return tf.convert_to_tensor(ret, dtype=tf.float32)


def nms(pred_boxes, iou_thr=0.7, eps=1e-6):
    """Non-Maximum Suppression
    Args:
        pred_boxes (np.ndarray dtype=np.float32): [x_min, y_min, x_max, y_max, confidence, class_idx]
        iou_thr (float): IoU Threshold (Default: 0.7)
        eps (float): Epsilon value for prevent zero division (Default:1e-6)

    Returns:
        np.ndarray dtype=np.float32: Non-Maximum Suppressed prediction boxes
    """
    if len(pred_boxes) == 0:
        return np.array([], dtype=np.float32)

    x_min, y_min = pred_boxes[:,0], pred_boxes[:,1]
    x_max, y_max = pred_boxes[:,2], pred_boxes[:,3]
    width = np.maximum(x_max - x_min, 0.)
    height = np.maximum(y_max - y_min, 0.)
    area = width * height

    selected_idx_list = list()
    confidence = pred_boxes[:, 4]
    idxs_sorted = np.argsort(confidence)  # Sort in ascending order
    while len(idxs_sorted) > 0:
        max_confidence_idx = len(idxs_sorted) - 1
        non_selected_idxs = idxs_sorted[:max_confidence_idx]
        selected_idx = idxs_sorted[max_confidence_idx]
        selected_idx_list.append(selected_idx)

        inter_xmin = np.maximum(x_min[selected_idx], x_min[non_selected_idxs])
        inter_ymin = np.maximum(y_min[selected_idx], y_min[non_selected_idxs])
        inter_xmax = np.minimum(x_max[selected_idx], x_max[non_selected_idxs])
        inter_ymax = np.minimum(y_max[selected_idx], y_max[non_selected_idxs])
        inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
        inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
        inter_area = inter_w * inter_h

        union = (area[selected_idx] + area[non_selected_idxs]) - inter_area + eps
        iou = inter_area / union
        idxs_sorted = np.delete(idxs_sorted, np.concatenate(([max_confidence_idx], np.where(iou >= iou_thr)[0])))
    return pred_boxes[selected_idx_list]
