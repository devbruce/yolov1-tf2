import tensorflow as tf
import numpy as np
from datasets.voc_tfds.viz import viz_voc_prep


__all__ = ['tb_write_losses', 'tb_write_lr', 'tb_write_APs', 'tb_write_imgs', 'tb_write_sampled_voc_gt_imgs']


def tb_write_losses(tb_writer, losses, step):
    with tb_writer.as_default():
        tf.summary.scalar('losses/total_loss', losses['total_loss'], step=step)
        tf.summary.scalar('losses/coord_loss', losses['coord_loss'], step=step)
        tf.summary.scalar('losses/obj_loss', losses['obj_loss'], step=step)
        tf.summary.scalar('losses/noobj_loss', losses['noobj_loss'], step=step)
        tf.summary.scalar('losses/class_loss', losses['class_loss'], step=step)


def tb_write_lr(tb_writer, lr, step):
    with tb_writer.as_default():
        tf.summary.scalar('learning_rate', lr, step=step)

    
def tb_write_APs(tb_writer, APs, step, prefix='[Val] '):
    APs = APs.copy()
    mAP = APs.pop('mAP')
    with tb_writer.as_default():
        tf.summary.scalar(prefix + 'APs/mAP', mAP, step=step)
        for cls_name, ap in APs.items():
            tf.summary.scalar(prefix + 'APs/' + f'{cls_name}', ap, step=step)
            

def tb_write_imgs(tb_writer, name, imgs, step, max_outputs):
    with tb_writer.as_default():
        tf.summary.image(name, imgs, step=step, max_outputs=max_outputs)


def tb_write_sampled_voc_gt_imgs(batch_data, input_height, input_width, tb_writer, name, max_outputs, val):
    viz_img_list = list()
    for idx in range(len(batch_data['image'])):
        viz_img = viz_voc_prep(
            batch_data,
            idx,
            input_height=input_height,
            input_width=input_width,
            val=val,
            box_color=(0, 255, 0),
            thickness=1,
            txt_color=(255, 0, 0),
        )
        viz_img_list.append(viz_img)
    viz_img_batch = np.stack(viz_img_list, axis=0)
    tb_write_imgs(tb_writer=tb_writer, name=name, imgs=viz_img_batch, step=0, max_outputs=max_outputs)
