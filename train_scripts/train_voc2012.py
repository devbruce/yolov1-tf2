import _add_project_path

import os
import pickle
import tqdm
import cv2
import numpy as np
import tensorflow as tf

from absl import flags, app
from pascalvoc_ap.ap import get_ap
from libs.models import YOLO, get_xception_backbone
from libs.losses import train_step, get_batch_losses, get_losses
from libs.loggers import TrainLogHandler, ValLogHandler
from libs.loggers.console_logs import get_logger
from libs.loggers.tb_logs import tb_write_sampled_voc_gt_imgs, tb_write_imgs
from libs.utils import yolo_output2boxes, box_postp2use, viz_pred
from datasets.voc2012_tfds.voc2012 import GetVoc2012
from datasets.voc2012_tfds.libs import prep_voc_data, VOC_CLS_MAP
from datasets.voc2012_tfds.eval.prepare_eval import get_gts_all
from configs import cfg, ProjectPath


FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', default=cfg.epochs, help='Number of training epochs')
flags.DEFINE_integer('batch_size', default=cfg.batch_size, help='Batch size')
flags.DEFINE_integer('tb_img_max_outputs', default=cfg.tb_img_max_outputs, help='Number of visualized prediction images in tensorboard')
flags.DEFINE_integer('val_step', default=cfg.val_step, help='Validation interval during training')
flags.DEFINE_float('init_lr', default=cfg.learning_rate, help='Initial learning rate')
flags.DEFINE_float('lr_decay_rate', default=0.5, help='Decay rate of learning rate')
flags.DEFINE_integer('lr_decay_steps', default=25000, help='Learning rate decay steps')
# flags.mark_flag_as_required('')

# Save some gpu errors
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    

def main(argv):
    global voc2012, voc2012_val_gts_all
    global logger, tb_train_writer, tb_val_writer, val_viz_batch_data
    global yolo, optimizer
    global VOC2012_PB_PATH, ckpt, ckpt_manager

    # Dataset (PascalVOC2012)
    voc2012 = GetVoc2012(batch_size=FLAGS.batch_size)
    voc2012_val_gts_all_path = os.path.join(ProjectPath.DATASETS_DIR.value, 'voc2012_tfds', 'eval', 'val_gts_all_448.pickle')
    if os.path.exists(voc2012_val_gts_all_path):
        voc2012_val_gts_all = pickle.load(open(voc2012_val_gts_all_path, 'rb'))
    else:
        voc2012_val_gts_all = get_gts_all(voc2012.get_val_ds(), cfg.input_height, cfg.input_width, VOC_CLS_MAP)
        
    # Logger
    logger = get_logger()

    # Tensorboard
    tb_train_writer = tf.summary.create_file_writer(ProjectPath.TB_LOGS_TRAIN_DIR.value)
    tb_val_writer = tf.summary.create_file_writer(ProjectPath.TB_LOGS_VAL_DIR.value)
    val_viz_batch_data = next(iter(voc2012.get_val_ds().take(1)))

    # Tensorboard Visualization (Validation GT)
    tb_write_sampled_voc_gt_imgs(
        batch_data=val_viz_batch_data,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        tb_writer=tb_val_writer,
        name='Validation GT',
        max_outputs=FLAGS.tb_img_max_outputs,
    )

    # Model
    backbone_xception = get_xception_backbone(cfg=cfg, freeze=False)
    yolo = YOLO(backbone=backbone_xception, cfg=cfg)

    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.init_lr,
        decay_steps=FLAGS.lr_decay_steps,
        decay_rate=FLAGS.lr_decay_rate,
        staircase=True,
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    # Checkpoint
    VOC2012_PB_PATH = os.path.join(ProjectPath.VOC2012_CKPTS_DIR.value, f'yoloy_epoch_{FLAGS.epochs}.pb')
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=yolo)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory=ProjectPath.VOC2012_CKPTS_DIR.value,
        max_to_keep=5
    )

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=ProjectPath.VOC2012_CKPTS_DIR.value)
    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        print('\n' + '=' * 60)
        print(f'* Load latest checkpoint file [{latest_ckpt}]')
        print('=' * 60 + '\n')
    else:
        print('\n' + '=' * 60)
        print(f'* Training from scratch')
        print('=' * 60 + '\n')

    # Training
    train()

    
def train():
    for epoch in range(1, FLAGS.epochs+1):
        train_ds = voc2012.get_train_ds(shuffle=True, drop_remainder=True)
        steps_per_epoch = len(train_ds)
        train_log_handler = TrainLogHandler(total_epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, optimizer=optimizer, logger=logger)

        for step, batch_data in enumerate(train_ds, 1):
            batch_imgs, batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width)
            losses = train_step(yolo, optimizer, batch_imgs, batch_labels, cfg)
            lr = optimizer.lr.numpy()
            train_log_handler.logging(epoch=epoch, step=step, losses=losses, lr=lr, tb_writer=tb_train_writer)

        if epoch % FLAGS.val_step == 0:
            validation(epoch=epoch)
    
    yolo.save(filepath=VOC2012_PB_PATH, save_format='tf')
    print('\n' + '=' * 60)
    print('* Training Completed and Save pb file [{VOC2012_PB_PATH}]')
    print('=' * 60 + '\n')
    
    
def validation(epoch):
    global mAP_prev
    mAP_prev = 0
    
    val_ds = voc2012.get_val_ds()
    val_log_handler = ValLogHandler(total_epochs=FLAGS.epochs, logger=logger)
    val_losses_raw = {
        'total_loss': tf.keras.metrics.MeanTensor(),
        'coord_loss': tf.keras.metrics.MeanTensor(),
        'obj_loss': tf.keras.metrics.MeanTensor(),
        'noobj_loss': tf.keras.metrics.MeanTensor(),
        'class_loss': tf.keras.metrics.MeanTensor(),
    }

    img_id = 0
    val_preds_all = list()

    for step, batch_data in tqdm.tqdm(enumerate(val_ds, 1), total=len(val_ds), desc='Validation'):
        batch_imgs, batch_labels = prep_voc_data(batch_data, input_height=cfg.input_height, input_width=cfg.input_width)
        yolo_output_raw = yolo(batch_imgs)

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
            output_boxes = box_postp2use(yolo_boxes[i], cfg.nms_iou_thr, cfg.conf_thr)
            if output_boxes.size == 0:
                img_id += 1
                continue
            for output_box in output_boxes:
                *pts, conf, cls_idx = output_box
                cls_name = VOC_CLS_MAP[cls_idx]
                val_preds_all.append([cls_name, conf, *map(round, pts), img_id])
                img_id += 1

    APs = get_ap(preds_all=val_preds_all, gts_all=voc2012_val_gts_all, classes=list(VOC_CLS_MAP.values()), iou_thr=0.5)
    val_losses = dict()
    for loss_name in val_losses_raw:
        val_losses[loss_name] = val_losses_raw[loss_name].result().numpy()
        val_losses_raw[loss_name].reset_states()

    val_log_handler.logging(epoch=epoch, losses=val_losses, APs=APs, tb_writer=tb_val_writer)

    # ====== Tensorboard Image: prediction output visualization ======
    sampled_voc_imgs, _ = prep_voc_data(val_viz_batch_data, input_height=cfg.input_height, input_width=cfg.input_width)
    sampled_voc_preds = yolo(sampled_voc_imgs)
    sampled_voc_output_boxes = yolo_output2boxes(sampled_voc_preds, cfg.input_height, cfg.input_width, cfg.cell_size, cfg.boxes_per_cell)
    sampled_imgs_num = FLAGS.tb_img_max_outputs if len(sampled_voc_imgs) > FLAGS.tb_img_max_outputs else len(sampled_voc_imgs)
    pred_viz_imgs = np.empty([sampled_imgs_num, cfg.input_height, cfg.input_width, 3], dtype=np.uint8)
    for idx in range(sampled_imgs_num):
        img = sampled_voc_imgs[idx].numpy()
        labels = box_postp2use(pred_boxes=sampled_voc_output_boxes[idx], nms_iou_thr=cfg.nms_iou_thr, conf_thr=cfg.conf_thr)
        pred_viz_imgs[idx] = viz_pred(img=img, labels=labels, cls_map=VOC_CLS_MAP)
    tb_write_imgs(tb_val_writer, 'Validation Prediction', imgs=pred_viz_imgs, step=epoch, max_outputs=FLAGS.tb_img_max_outputs)

    # Save Checkpoint
    if APs['mAP'] >= mAP_prev:
        ckpt_manager.save(checkpoint_number=ckpt.step)
        mAP_prev = APs['mAP']
        print('\n' + '=' * 30)
        print('* Save checkpoint file')
        print('=' * 30 + '\n')
    ckpt.step.assign_add(1)


if __name__ == '__main__':
    app.run(main)
