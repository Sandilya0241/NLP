import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from models import build_generator


def generate_samples(checkpoint_dir='checkpoints', out_dir='generated', noise_dim=100, num_per_class=4):
    gen = build_generator()
    ckpt = tf.train.Checkpoint(generator=gen)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        ckpt.restore(latest).expect_partial()
    os.makedirs(out_dir, exist_ok=True)

    for label in [0,1]:
        noise = tf.random.normal([num_per_class, noise_dim])
        labels = np.full((num_per_class,), label, dtype='int32')
        imgs = gen([noise, labels], training=False)
        imgs = (imgs + 1.0) / 2.0
        for i in range(num_per_class):
            plt.imsave(os.path.join(out_dir, f'label_{label}_{i}.png'), imgs[i])

    print('Saved generated samples to', out_dir)


if __name__ == '__main__':
    generate_samples()
