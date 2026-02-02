import tensorflow as tf
import tensorflow_datasets as tfds


def load_cats_dogs(batch_size=64, img_size=64, buffer_size=1000):
    def _preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_size, img_size])
        image = (image * 2.0) - 1.0  # scale to [-1,1]
        label = tf.cast(label, tf.int32)
        return image, label

    ds = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
