import tensorflow as tf
import numpy as np
from datasets.voc2012_tfds.viz import viz_voc_prep


__all__ = ['tb_write_scalars', 'tb_write_imgs', 'tb_write_sampled_voc_gt_imgs']


def tb_write_scalars(tb_writer, losses, step):
    with tb_writer.as_default():
        tf.summary.scalar('total_loss', losses['total_loss'], step=step)
        tf.summary.scalar('coord_loss', losses['coord_loss'], step=step)
        tf.summary.scalar('obj_loss', losses['obj_loss'], step=step)
        tf.summary.scalar('noobj_loss', losses['noobj_loss'], step=step)
        tf.summary.scalar('class_loss', losses['class_loss'], step=step)


def tb_write_imgs(tb_writer, name, imgs, step, max_outputs):
    with tb_writer.as_default():
        tf.summary.image(name, imgs, step=step, max_outputs=max_outputs)


def tb_write_sampled_voc_gt_imgs(ds, tb_writer, name, max_outputs):
    viz_img_list = list()
    viz_batch_data = next(iter(ds))
    viz_batch_data['imgs'] = viz_batch_data['imgs'][:max_outputs]
    for idx in range(len(viz_batch_data['imgs'])):
        viz_img = viz_voc_prep(
            viz_batch_data,
            idx,
            input_height=cfg.input_height,
            input_width=cfg.input_width,
            box_color=(0, 255, 0),
            thickness=1,
            txt_color=(255, 0, 0),
        )
        viz_img_list.append(viz_img)
    viz_img_batch = np.stack(viz_img_list, axis=0)
    tb_write_imgs(tb_writer=tb_writer, name=name, imgs=viz_img_batch, step=0, max_outputs=max_outputs)
