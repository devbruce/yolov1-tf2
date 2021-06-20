import os
import tqdm
import pickle
import numpy as np
from datasets.voc_tfds.libs import prep_voc_data
from configs import ProjectPath


__all__ = ['get_labels']


def get_labels(ds, input_height, input_width, cls_map, full_save=False):
    labels = list()
    img_id = 0
    
    def cls_idx2name(data_list):
        for data in data_list:
            cls_idx = data[4]
            cls_name = cls_map[cls_idx]
            data[4] = cls_name
            
    print('\n====== ====== Get validation labels for mAP Calculation ====== ======\n')
    for step, batch_data in tqdm.tqdm(enumerate(ds, 1), total=len(ds), desc='Get val labels'):
        _, batch_labels = prep_voc_data(batch_data, input_height=input_height, input_width=input_width, val=True)
        for batch_label in batch_labels:
            batch_label = batch_label.numpy()
            img_id_arr = np.array([img_id] * len(batch_label), dtype=np.float32)
            cx_rel, cy_rel = batch_label[:, 0], batch_label[:, 1]
            w_rel, h_rel = batch_label[:, 2], batch_label[:, 3]
            cls_idx = batch_label[:, 4]

            cx, cy = cx_rel * input_width, cy_rel * input_height
            w, h = w_rel * input_width, h_rel * input_height
            left, top = cx - (w / 2), cy - (h / 2)
            right, bottom = cx + (w / 2), cy + (h / 2)

            converted_data = np.array([left, top, right, bottom, cls_idx, img_id_arr], dtype=np.float32).T
            converted_data = np.around(converted_data).astype(np.int32).tolist()
            cls_idx2name(converted_data)
            labels.extend(converted_data)
            img_id += 1
    print('\n====== ====== Get validation labels for mAP Calculation (Completed) ======\n')

    # Save as pickle file
    if full_save:
        voc2012_val_labels_path = os.path.join(ProjectPath.DATASETS_DIR.value, 'voc_tfds', 'eval', f'val_labels_448_full.pickle')
        with open(voc2012_val_labels_path, 'wb') as f:
            pickle.dump(labels, f)
    
    return labels
