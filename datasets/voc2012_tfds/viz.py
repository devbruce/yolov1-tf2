import cv2
from .libs import VOC_CLS_MAP, prep_voc_data


__all__ = ['viz_voc_origin', 'viz_voc_prep']


def viz_voc_origin(batch_data, idx, input_height, input_width, box_color=(0, 255, 0), thickness=1, txt_color=(255, 0, 0)):
    pass


def viz_voc_prep(batch_data, idx, input_height, input_width, box_color=(0, 255, 0), thickness=1, txt_color=(255, 0, 0)):
    imgs, labels = prep_voc_data(batch_data, input_height, input_width)
    img = imgs[idx].numpy().copy()
    img = (img * 255).astype(np.uint8)
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
