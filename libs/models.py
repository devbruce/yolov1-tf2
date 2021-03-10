import tensorflow as tf


__all__ = ['YOLO', 'get_xception_backbone']


class YOLO(tf.keras.Model):
    def __init__(self, backbone, cfg):
        super().__init__()
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        x = tf.keras.layers.Dense(cfg.cell_size * cfg.cell_size * ((cfg.boxes_per_cell * 5) + cfg.num_classes), activation=None)(x)
        output = tf.reshape(x, [-1, cfg.cell_size, cfg.cell_size, (cfg.boxes_per_cell * 5) + cfg.num_classes])
        self.model = tf.keras.Model(inputs=backbone.input, outputs=output)

    def call(self, x):
        return self.model(x)


def get_xception_backbone(cfg, freeze=False):
    backbone = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3),
    )
    backbone.trainable = not freeze
    return backbone
