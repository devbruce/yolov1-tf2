import tensorflow as tf
import numpy as np


__all__ = ['tb_write_scalars', 'tb_write_img']


def tb_write_scalars(tb_writer, losses, step):
    with tb_writer.as_default():
        tf.summary.scalar('total_loss', losses['total_loss'], step=step)
        tf.summary.scalar('coord_loss', losses['coord_loss'], step=step)
        tf.summary.scalar('obj_loss', losses['obj_loss'], step=step)
        tf.summary.scalar('noobj_loss', losses['noobj_loss'], step=step)
        tf.summary.scalar('class_loss', losses['class_loss'], step=step)


def tb_write_img(tb_writer, name, img, step):
    with tb_writer.as_default():
        tf.summary.image(name, img, step=step)
