from termcolor import colored
import tensorflow as tf
import numpy as np


__all__ = ['postprocess_yolo_format', 'train_step_str_log']


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


def train_step_str_log(total_epoch, total_step, current_epoch, current_step, losses):
    progress = colored(f'* Epoch: {current_epoch:^4} / {total_epoch:^4} | Step: {current_step:^4} / {total_step:^4}', 'green')
    total_loss = f'>>> Total Loss: {losses["total_loss"]:<8.4f}'
    total_loss = colored(total_loss, 'red')
    loss_info = ' (coord: {:<8.4f}, obj: {:<8.4f}, noobj: {:<8.4f}, class: {:<8.4f})'
    loss_info = colored(loss_info.format(losses['coord_loss'], losses['obj_loss'], losses['noobj_loss'], losses['class_loss']), 'cyan')
    log = '\n' + progress + '\n' + total_loss + loss_info
    return log
