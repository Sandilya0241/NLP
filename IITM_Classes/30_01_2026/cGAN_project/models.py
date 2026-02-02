import tensorflow as tf


def build_generator(img_size=64, noise_dim=100, num_classes=2):
    label_input = tf.keras.layers.Input(shape=(), dtype='int32')
    noise_input = tf.keras.layers.Input(shape=(noise_dim,))

    # label embedding and expand
    label_emb = tf.keras.layers.Embedding(num_classes, 50)(label_input)
    label_dense = tf.keras.layers.Dense(8 * 8 * 1, activation='relu')(label_emb)
    label_reshaped = tf.keras.layers.Reshape((8, 8, 1))(label_dense)

    x = tf.keras.layers.Concatenate()([noise_input, tf.keras.layers.Flatten()(label_emb)])
    x = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((8, 8, 256))(x)

    # concatenate label channel-wise
    x = tf.keras.layers.Concatenate()([x, label_reshaped])

    x = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation='tanh')(x)

    model = tf.keras.Model([noise_input, label_input], x, name='generator')
    return model


def build_discriminator(img_size=64, num_classes=2):
    img_input = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    label_input = tf.keras.layers.Input(shape=(), dtype='int32')

    # embed label into a spatial map
    label_emb = tf.keras.layers.Embedding(num_classes, 50)(label_input)
    label_dense = tf.keras.layers.Dense(img_size * img_size * 1)(label_emb)
    label_reshaped = tf.keras.layers.Reshape((img_size, img_size, 1))(label_dense)

    x = tf.keras.layers.Concatenate()([img_input, label_reshaped])

    x = tf.keras.layers.Conv2D(64, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model([img_input, label_input], x, name='discriminator')
    return model
