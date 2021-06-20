import tensorflow as tf


__all__ = ['YOLO', 'get_xception_backbone']


class YOLO(tf.keras.Model):
    def __init__(self, backbone, cfg):
        super().__init__()
        # Xception output shape: (14, 14, 2048)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(backbone.output)  # (14, 14, 2048) --> (7, 7, 2048)
        x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1))(x)      # (7, 7, 2048) --> (7, 7, 1024)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=4096)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Dense(units=cfg.cell_size * cfg.cell_size * ((cfg.boxes_per_cell * 5) + cfg.num_classes))(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        output = tf.reshape(x, [-1, cfg.cell_size, cfg.cell_size, (cfg.boxes_per_cell * 5) + cfg.num_classes]) 
        self.model = tf.keras.Model(inputs=backbone.input, outputs=output)

    def call(self, x):
        return self.model(x)


def get_xception_backbone(input_height, input_width, freeze=False):
    backbone = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(input_height, input_width, 3),
    )
    backbone.trainable = not freeze
    return backbone
