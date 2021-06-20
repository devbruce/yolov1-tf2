import _add_project_path

import os
import pickle
import tqdm
import numpy as np
import tensorflow as tf

from absl import flags, app
from termcolor import colored
from calc4ap.voc import CalcVOCmAP
from libs.models import YOLO, get_xception_backbone
from libs.losses import train_step, get_losses
from libs.loggers import TrainLogHandler, ValLogHandler
from libs.loggers.console_logs import get_logger
from libs.loggers.tb_logs import tb_write_sampled_voc_gt_imgs, tb_write_imgs
from libs.utils import yolo_output2boxes, box_postp2use, viz_pred
from datasets.voc_tfds.voc import GetVoc
from datasets.voc_tfds.libs import prep_voc_data, VOC_CLS_MAP
from datasets.voc_tfds.eval.prepare_eval import get_labels
from configs import cfg, ProjectPath


FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', default=cfg.epochs, help='Number of training epochs')
flags.DEFINE_float('init_lr', default=cfg.init_lr, help='Initial learning rate')
flags.DEFINE_float('lr_decay_rate', default=cfg.lr_decay_rate, help='Learning rate decay rate')
flags.DEFINE_integer('lr_decay_steps', default=cfg.lr_decay_steps, help='Learning rate decay steps')
flags.DEFINE_integer('batch_size', default=cfg.batch_size, help='Batch size')
flags.DEFINE_integer('val_step', default=cfg.val_step, help='Validation interval during training')
flags.DEFINE_integer('tb_img_max_outputs', default=cfg.tb_img_max_outputs, help='Number of visualized prediction images in tensorboard')
flags.DEFINE_float('train_ds_sample_ratio', default=cfg.train_ds_sample_ratio, help='Training dataset sampling ratio')
flags.DEFINE_float('val_ds_sample_ratio', default=cfg.val_ds_sample_ratio, help='Validation dataset sampling ratio')
# flags.mark_flag_as_required('')


# Save some gpu errors
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
VOC_PB_DIR = os.path.join(ProjectPath.VOC_CKPTS_DIR.value, f'yolo_voc_{cfg.input_height}x{cfg.input_width}')
    

def main(_argv):
    global voc, val_labels
    global logger, tb_train_writer, tb_val_writer, train_viz_batch_data, val_viz_batch_data
    global yolo, optimizer
    global VOC_PB_DIR, ckpt, ckpt_manager
    global val_metrics

    # Dataset (PascalVOC)
    voc = GetVoc(batch_size=FLAGS.batch_size)
    val_labels_path = os.path.join(ProjectPath.DATASETS_DIR.value, 'voc_tfds', 'eval', 'val_labels_448_full.pickle')
    if FLAGS.val_ds_sample_ratio == 1:
        if os.path.exists(val_labels_path):
            val_labels = pickle.load(open(val_labels_path, 'rb'))
        else:
            val_labels = get_labels(voc.get_val_ds(), cfg.input_height, cfg.input_width, VOC_CLS_MAP, full_save=True)
    else:
        val_labels = get_labels(voc.get_val_ds(sample_ratio=FLAGS.val_ds_sample_ratio), cfg.input_height, cfg.input_width, VOC_CLS_MAP)
        
    # Logger
    logger = get_logger()
    logger.propagate = False

    # Tensorboard
    tb_train_writer = tf.summary.create_file_writer(ProjectPath.TB_LOGS_TRAIN_DIR.value)
    tb_val_writer = tf.summary.create_file_writer(ProjectPath.TB_LOGS_VAL_DIR.value)
    train_viz_batch_data = next(iter(voc.get_train_ds(shuffle=False, drop_remainder=False).take(1)))
    val_viz_batch_data = next(iter(voc.get_val_ds().take(1)))
    
    # Prediction Visualization (Tensorboard)
    tb_write_sampled_voc_gt_imgs(
        batch_data=train_viz_batch_data,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        val=True,
        tb_writer=tb_train_writer,
        name='[Train] GT',
        max_outputs=FLAGS.tb_img_max_outputs,
    )
    tb_write_sampled_voc_gt_imgs(
        batch_data=val_viz_batch_data,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        val=True,
        tb_writer=tb_val_writer,
        name='[Val] GT',
        max_outputs=FLAGS.tb_img_max_outputs,
    )

    # Model
    backbone_xception = get_xception_backbone(input_height=cfg.input_height, input_width=cfg.input_width, freeze=False)
    yolo = YOLO(backbone=backbone_xception, cfg=cfg)

    # Optimizer
    # Paper Page 4. We continue training with 1e-2 for 75 epochs, then 1e-3 for 30 epochs, and finally 1e-4 for 30 epochs.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.init_lr,
        decay_steps=FLAGS.lr_decay_steps,
        decay_rate=FLAGS.lr_decay_rate,
        staircase=True,
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    # Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=yolo)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory=ProjectPath.VOC_CKPTS_DIR.value,
        max_to_keep=5
    )

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ProjectPath.VOC_CKPTS_DIR.value)
    latest_ckpt_log = '\n' + '=' * 60 + '\n'
    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        latest_ckpt_log += f'* Load latest checkpoint file [{latest_ckpt}]'
    else:
        latest_ckpt_log += '* Training from scratch'
    latest_ckpt_log += ('\n' + '=' * 60 + '\n')
    logger.info(latest_ckpt_log)
    print(colored(latest_ckpt_log, 'magenta'))

    # Val Metrics
    val_metrics = {'mAP_best': 0.}

    # Training
    train()

    
def train():
    for epoch in range(1, FLAGS.epochs+1):
        train_ds = voc.get_train_ds(shuffle=True, drop_remainder=True, sample_ratio=FLAGS.train_ds_sample_ratio)
        steps_per_epoch = len(train_ds)
        train_log_handler = TrainLogHandler(total_epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, optimizer=optimizer, logger=logger)

        for step, batch_data in enumerate(train_ds, 1):
            batch_imgs, batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width, val=False)
            losses = train_step(yolo, optimizer, batch_imgs, batch_labels, cfg)
            train_log_handler.logging(epoch=epoch, step=step, losses=losses, tb_writer=tb_train_writer)

        if epoch % FLAGS.val_step == 0:
            validation(epoch=epoch)
    
    
def validation(epoch):
    val_ds = voc.get_val_ds(sample_ratio=FLAGS.val_ds_sample_ratio)
    val_log_handler = ValLogHandler(total_epochs=FLAGS.epochs, logger=logger)
    val_losses_raw = {
        'total_loss': tf.keras.metrics.MeanTensor(),
        'coord_loss': tf.keras.metrics.MeanTensor(),
        'obj_loss': tf.keras.metrics.MeanTensor(),
        'noobj_loss': tf.keras.metrics.MeanTensor(),
        'class_loss': tf.keras.metrics.MeanTensor(),
    }

    img_id = 0
    val_preds = list()

    for step, batch_data in tqdm.tqdm(enumerate(val_ds, 1), total=len(val_ds), desc='Validation'):
        batch_imgs, batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width, val=True)
        yolo_output_raw = yolo(batch_imgs, training=False)

        # ====== ====== ====== Calc Losses ====== ====== ======
        batch_losses = {
            'total_loss': 0.,
            'coord_loss': 0.,
            'obj_loss': 0.,
            'noobj_loss': 0.,
            'class_loss': 0.,
        }
        for i in range(len(yolo_output_raw)):
            one_loss = get_losses(one_pred=yolo_output_raw[i], one_label=batch_labels[i], cfg=cfg)
            batch_losses['total_loss'] += one_loss['total_loss']
            batch_losses['coord_loss'] += one_loss['coord_loss']
            batch_losses['obj_loss'] += one_loss['obj_loss']
            batch_losses['noobj_loss'] += one_loss['noobj_loss']
            batch_losses['class_loss'] += one_loss['class_loss']

        val_losses_raw['total_loss'].update_state(batch_losses['total_loss'] / len(batch_imgs))
        val_losses_raw['coord_loss'].update_state(batch_losses['coord_loss'] / len(batch_imgs))
        val_losses_raw['obj_loss'].update_state(batch_losses['obj_loss'] / len(batch_imgs))
        val_losses_raw['noobj_loss'].update_state(batch_losses['noobj_loss'] / len(batch_imgs))
        val_losses_raw['class_loss'].update_state(batch_losses['class_loss'] / len(batch_imgs))

        # ====== ====== ====== mAP ====== ====== ======
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
    val_losses = dict()
    for loss_name in val_losses_raw:
        val_losses[loss_name] = val_losses_raw[loss_name].result().numpy()
        val_losses_raw[loss_name].reset_states()

    val_log_handler.logging(epoch=epoch, losses=val_losses, APs=ap_summary, tb_writer=tb_val_writer)

    # ========= Tensorboard Image: prediction output visualization =========
    # Training data output visualization
    sampled_voc_imgs, _ = prep_voc_data(train_viz_batch_data, input_height=cfg.input_height, input_width=cfg.input_width, val=True)
    sampled_voc_preds = yolo(sampled_voc_imgs)
    sampled_voc_output_boxes = yolo_output2boxes(sampled_voc_preds, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
    sampled_imgs_num = FLAGS.tb_img_max_outputs if len(sampled_voc_imgs) > FLAGS.tb_img_max_outputs else len(sampled_voc_imgs)
    pred_viz_imgs = np.empty([sampled_imgs_num, cfg.input_height, cfg.input_width, 3], dtype=np.uint8)
    for idx in range(sampled_imgs_num):
        img = sampled_voc_imgs[idx].numpy()
        labels = box_postp2use(pred_boxes=sampled_voc_output_boxes[idx], nms_iou_thr=cfg.nms_iou_thr, conf_thr=cfg.conf_thr)
        pred_viz_imgs[idx] = viz_pred(img=img, labels=labels, cls_map=VOC_CLS_MAP)
    tb_write_imgs(
        tb_train_writer,
        name=f'[Train] Prediction (confidence_thr: {cfg.conf_thr}, nms_iou_thr: {cfg.nms_iou_thr})',
        imgs=pred_viz_imgs,
        step=epoch,
        max_outputs=FLAGS.tb_img_max_outputs,
    )

    # Validation data output visualization
    sampled_voc_imgs, _ = prep_voc_data(val_viz_batch_data, input_height=cfg.input_height, input_width=cfg.input_width, val=True)
    sampled_voc_preds = yolo(sampled_voc_imgs)
    sampled_voc_output_boxes = yolo_output2boxes(sampled_voc_preds, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
    sampled_imgs_num = FLAGS.tb_img_max_outputs if len(sampled_voc_imgs) > FLAGS.tb_img_max_outputs else len(sampled_voc_imgs)
    pred_viz_imgs = np.empty([sampled_imgs_num, cfg.input_height, cfg.input_width, 3], dtype=np.uint8)
    for idx in range(sampled_imgs_num):
        img = sampled_voc_imgs[idx].numpy()
        labels = box_postp2use(pred_boxes=sampled_voc_output_boxes[idx], nms_iou_thr=cfg.nms_iou_thr, conf_thr=cfg.conf_thr)
        pred_viz_imgs[idx] = viz_pred(img=img, labels=labels, cls_map=VOC_CLS_MAP)
    tb_write_imgs(
        tb_val_writer,
        name=f'[Val] Prediction (confidence_thr: {cfg.conf_thr}, nms_iou_thr: {cfg.nms_iou_thr})',
        imgs=pred_viz_imgs,
        step=epoch,
        max_outputs=FLAGS.tb_img_max_outputs,
    )
    # ========= ================================================ =========

    # Save checkpoint and pb
    if ap_summary['mAP'] >= val_metrics['mAP_best']:
        ckpt_manager.save(checkpoint_number=ckpt.step)
        yolo.save(filepath=VOC_PB_DIR, save_format='tf')
        val_metrics['mAP_best'] = ap_summary['mAP']
        ckpt_log = '\n' + '=' * 100 + '\n'
        ckpt_log += f'* Save checkpoint file and pb file [{VOC_PB_DIR}]'
        ckpt_log += '\n' + '=' * 100 + '\n'
        logger.info(ckpt_log)
        print(colored(ckpt_log, 'green'))
    ckpt.step.assign_add(1)


if __name__ == '__main__':
    app.run(main)
