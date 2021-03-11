import tensorflow as tf


__all__ = ['tb_scalar']


def tb_scalars(tb_writer, losses, step):
    with tb_writer.as_default():
        tf.summary.scalar('total_loss', losses['total_loss'], step=step)
        tf.summary.scalar('coord_loss', losses['coord_loss'], step=step)
        tf.summary.scalar('obj_loss', losses['obj_loss'], step=step)
        tf.summary.scalar('noobj_loss', losses['noobj_loss'], step=step)
        tf.summary.scalar('class_loss', losses['class_loss'], step=step)
