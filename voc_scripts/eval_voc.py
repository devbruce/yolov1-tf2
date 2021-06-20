import _add_project_path

import os
import tqdm
import pickle
import tensorflow as tf
from termcolor import colored
from absl import flags, app
from calc4ap.voc import CalcVOCmAP
from libs.utils import yolo_output2boxes, box_postp2use
from datasets.voc_tfds.voc import GetVoc
from datasets.voc_tfds.libs import prep_voc_data, VOC_CLS_MAP
from datasets.voc_tfds.eval.prepare_eval import get_labels
from configs import ProjectPath, cfg


FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', default=cfg.batch_size, help='Batch size')
flags.DEFINE_string('pb_dir', default=os.path.join(ProjectPath.VOC_CKPTS_DIR.value, 'yolo_voc_448x448'), help='Save pb directory path')
flags.DEFINE_float('val_ds_sample_ratio', default=cfg.val_ds_sample_ratio, help='Validation dataset sampling ratio')


def main(_argv):
    yolo = tf.saved_model.load(
        export_dir=FLAGS.pb_dir,
        tags=None,
        options=None,
    )

    voc = GetVoc(batch_size=FLAGS.batch_size)
    val_ds = voc.get_val_ds(sample_ratio=FLAGS.val_ds_sample_ratio)
    val_preds = list()
    val_labels_path = os.path.join(ProjectPath.DATASETS_DIR.value, 'voc_tfds', 'eval', 'val_labels_448_full.pickle')
    if FLAGS.val_ds_sample_ratio == 1. and os.path.exists(val_labels_path):
        val_labels = pickle.load(open(val_labels_path, 'rb'))
    else:
        val_labels = get_labels(ds=val_ds, input_height=cfg.input_height, input_width=cfg.input_width, cls_map=VOC_CLS_MAP, full_save=False)

    img_id = 0
    for _step, batch_data in tqdm.tqdm(enumerate(val_ds, 1), total=len(val_ds), desc='Validation'):
        batch_imgs, _batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width, val=True)
        yolo_output_raw = yolo(batch_imgs, training=False)
        yolo_boxes = yolo_output2boxes(yolo_output_raw, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
        for i in range(len(yolo_boxes)):
            output_boxes = box_postp2use(yolo_boxes[i], cfg.nms_iou_thr, 0.)
            if output_boxes.size == 0:
                img_id += 1
                continue
            for output_box in output_boxes:
                *pts, conf, cls_idx = output_box
                cls_name = VOC_CLS_MAP[cls_idx]
                val_preds.append([*map(round, pts), conf, cls_name, img_id])
            img_id += 1
    
    voc_ap = CalcVOCmAP(labels=val_labels, preds=val_preds, iou_thr=0.5, conf_thr=0.0)
    ap_summary = voc_ap.get_summary()
    mAP = ap_summary.pop('mAP')
    APs_log = '\n====== mAP ======\n' + f'* mAP: {mAP:<8.4f}\n'
    for cls_name, ap in ap_summary.items():
        APs_log += f'- {cls_name}: {ap:<8.4f}\n'
    APs_log += '====== ====== ======\n'
    APs_log_colored = colored(APs_log, 'magenta')
    print(APs_log_colored)
    

if __name__ == '__main__':
    app.run(main)
