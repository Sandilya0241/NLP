import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import load_cats_dogs
from models import build_generator, build_discriminator


class CGANTrainer:
    def __init__(self, img_size=64, noise_dim=100, batch_size=64, lr=2e-4):
        self.img_size = img_size
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        self.gen = build_generator(img_size=img_size, noise_dim=noise_dim)
        self.disc = build_discriminator(img_size=img_size)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(generator=self.gen, discriminator=self.disc,
                                              gen_optimizer=self.gen_optimizer, disc_optimizer=self.disc_optimizer)

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, images, labels):
        noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen([noise, labels], training=True)

            real_output = self.disc([images, labels], training=True)
            fake_output = self.disc([generated_images, labels], training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))

        return gen_loss, disc_loss

    def train(self, epochs=1, dataset=None, sample_dir='samples', sample_every=500):
        if dataset is None:
            dataset = load_cats_dogs(batch_size=self.batch_size, img_size=self.img_size)
        os.makedirs(sample_dir, exist_ok=True)

        step = 0
        for epoch in range(epochs):
            for image_batch, label_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch, label_batch)
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Gen loss: {gen_loss:.4f}, Disc loss: {disc_loss:.4f}")
                if step % sample_every == 0:
                    self.generate_and_save_images(step, sample_dir)
                    self.checkpoint.save(os.path.join(self.checkpoint_dir, 'ckpt'))
                step += 1

    def generate_and_save_images(self, step, output_dir, num_examples=8):
        seed_noise = tf.random.normal([num_examples, self.noise_dim])
        seeds = np.concatenate([np.zeros((num_examples//2,1)), np.ones((num_examples - num_examples//2,1))], axis=0)
        seeds = seeds.astype('int32').squeeze()
        gen_imgs = self.gen([seed_noise, seeds], training=False)
        gen_imgs = (gen_imgs + 1.0) / 2.0

        plt.figure(figsize=(12,3))
        for i in range(num_examples):
            plt.subplot(1, num_examples, i+1)
            plt.axis('off')
            plt.imshow(gen_imgs[i])
        out_path = os.path.join(output_dir, f'step_{step}.png')
        plt.savefig(out_path)
        plt.close()


if __name__ == '__main__':
    trainer = CGANTrainer()
    trainer.train(epochs=1, dataset=None)
