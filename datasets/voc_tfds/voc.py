import tensorflow as tf
import tensorflow_datasets as tfds


__all__ = ['GetVoc']


# Reference: https://www.tensorflow.org/datasets/catalog/voc
class GetVoc:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_train_ds(self, shuffle=True, drop_remainder=True, sample_ratio=1.):
        # Training Dataset (voc2007 trainval + 2012 trainval)
        (voc2007_train, voc2007_val), voc2007_info = tfds.load(name='voc/2007', split=['train', 'validation'], with_info=True)
        (voc2012_train, voc2012_val), voc2012_info = tfds.load(name='voc/2012', split=['train', 'validation'], with_info=True)
        train_ds = voc2007_train.concatenate(voc2007_val).concatenate(voc2012_train).concatenate(voc2012_val)
        train_ds_num_examples = voc2007_info.splits['train'].num_examples + \
            voc2007_info.splits['validation'].num_examples + \
            voc2012_info.splits['train'].num_examples + \
            voc2012_info.splits['validation'].num_examples

        if shuffle:
            train_ds = train_ds.shuffle(train_ds_num_examples)
        if sample_ratio != 1.:
            train_ds = train_ds.take(int(train_ds_num_examples * sample_ratio))
        
        # Loaded data first time, it's going to keep track of some of them in memory. It makes faster
        # train_ds = train_ds.cache()
        train_ds = train_ds.padded_batch(self.batch_size, drop_remainder=drop_remainder)
        # While running on gpu, it's going to prefetch number of batch_size examples, so they are ready to be run instantly after the gpu calls are done
        # train_ds = train_ds.prefetch(self.autotune)  
        return train_ds

    def get_val_ds(self, sample_ratio=1.):
        (val_ds,), ds_info = tfds.load(name='voc/2007', split=['test'], with_info=True)
        if sample_ratio != 1.:
            val_ds = val_ds.take(int(ds_info.splits['test'].num_examples * sample_ratio))
        val_ds = val_ds.padded_batch(self.batch_size)
        # val_ds = val_ds.prefetch(self.autotune)
        return val_ds
    