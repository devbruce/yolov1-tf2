import tensorflow as tf
import tensorflow_datasets as tfds
from .libs import normalize_img


__all__ = ['get_voc2012']


def get_voc2012(batch_size):
    """Reference: https://www.tensorflow.org/datasets/catalog/voc
    Args:
      batch_size (int): Batch size of dataset

    Returns:
      dict: train, val, test
    """
    (train_ds, val_ds, test_ds), ds_info = tfds.load(
        name='voc/2012',
        split=['train', 'validation', 'test'],
        with_info=True,
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()  # Loaded data first time, it's going to keep track of some of them in memory. It makes faster
    train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
    train_ds = train_ds.padded_batch(batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)  # While running on gpu, it's going to prefetch number of batch_size examples, so they are ready to be run instantly after the gpu calls are done

    val_ds = val_ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.padded_batch(batch_size)
    val_ds = val_ds.prefetch(AUTOTUNE)

    test_ds = test_ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.padded_batch(batch_size)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    ret = {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }
    return ret
