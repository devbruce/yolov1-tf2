import tensorflow as tf
import tensorflow_datasets as tfds
from .libs import normalize_img


__all__ = ['GetVoc2012']


# Reference: https://www.tensorflow.org/datasets/catalog/voc
class GetVoc2012:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_train_ds(self, shuffle=True, drop_remainder=True):
        train_ds, ds_info = tfds.load(name='voc/2012', split=['train'], with_info=True)
        train_ds = train_ds[0]
        train_ds = train_ds.map(normalize_img, num_parallel_calls=self.autotune)
        train_ds = train_ds.cache()  # Loaded data first time, it's going to keep track of some of them in memory. It makes faster
        if shuffle:
            train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
        train_ds = train_ds.padded_batch(self.batch_size, drop_remainder=drop_remainder)
        train_ds = train_ds.prefetch(self.autotune)  # While running on gpu, it's going to prefetch number of batch_size examples, so they are ready to be run instantly after the gpu calls are done
        return train_ds

    def get_val_ds(self):
        val_ds = tfds.load(name='voc/2012', split=['validation'], with_info=False)[0]
        val_ds = val_ds.map(normalize_img, num_parallel_calls=self.autotune)
        val_ds = val_ds.padded_batch(self.batch_size)
        val_ds = val_ds.prefetch(self.autotune)
        return val_ds

    def get_test_ds(self):
        test_ds = tfds.load(name='voc/2012', split=['test'], with_info=False)[0]
        test_ds = test_ds.map(normalize_img, num_parallel_calls=self.autotune)
        test_ds = test_ds.padded_batch(self.batch_size)
        test_ds = test_ds.prefetch(self.autotune)
        return test_ds
    