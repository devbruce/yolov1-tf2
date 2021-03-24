import tensorflow as tf


__all__ = ['normalize_img', 'flip_lr']


def normalize_img(data):
    imgs_normalized = tf.cast(data['image'], dtype=tf.float32) / 255.0
    ret = {
        'imgs': imgs_normalized,
        'labels': data['objects']
    }
    return ret


def flip_lr(data):
    img_lr_flipped = tf.image.random_flip_left_right(data['imgs'])
    
    pts = data['labels']['bbox']
    y_min_rel, x_min_rel, y_max_rel, x_max_rel = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    x_min_rel_flipped, x_max_rel_flipped = 1 - x_min_rel, 1 - x_max_rel
    pts_lr_flipped = tf.transpose(tf.stack([y_min_rel, x_min_rel_flipped, y_max_rel, x_max_rel_flipped]))
    
    data['imgs'] = img_lr_flipped
    data['labels']['bbox'] = pts_lr_flipped
    return data
