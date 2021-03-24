import random
import tensorflow as tf


__all__ = ['normalize_img', 'flip_lr', 'color_augs']


def normalize_img(data):
    imgs_normalized = tf.cast(data['image'], dtype=tf.float32) / 255.0
    ret = {
        'imgs': imgs_normalized,
        'labels': data['objects']
    }
    return ret


def flip_lr(data):
    if random.choice([True, False]):
        img_lr_flipped = data['imgs'][:, ::-1, :]
        pts = data['labels']['bbox']
        y_min_rel, x_min_rel, y_max_rel, x_max_rel = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        w_rel = x_max_rel - x_min_rel
        x_min_rel_flipped, x_max_rel_flipped = 1. - x_min_rel, 1. - x_max_rel
        pts_lr_flipped = tf.transpose(tf.stack([y_min_rel, x_min_rel_flipped - w_rel, y_max_rel, x_max_rel_flipped + w_rel]))
        data['imgs'] = img_lr_flipped
        data['labels']['bbox'] = pts_lr_flipped
    return data


def color_augs(data):
    imgs = data['imgs']
    imgs = tf.image.random_hue(imgs, 0.08)
    imgs = tf.image.random_saturation(imgs, 0.8, 1.2)
    imgs = tf.image.random_brightness(imgs, 0.2)
    imgs = tf.image.random_contrast(imgs, 0.8, 1.2)
    imgs = tf.clip_by_value(imgs, 0., 1.)
    data['imgs'] = imgs
    return data
