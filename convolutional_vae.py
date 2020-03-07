from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Conv2DTranspose, Dense, Reshape
import numpy as np
from matplotlib import pyplot as plt
import cv2


class CVAE(tf.keras.Model):

    def __init__(self, img_shape, latent_dim):
        """Create the Convolutional Variational Autoencoder model.

        The model consists of 4 convolutional layers, each one downscaling the image by a factor of 2, the encoded layer
        and 4 transpose convolutional layers that upscale the decoded image to the right shape. Input/output shape
        consistency is ensured as long as it is n x n x k with n power of 2

        Args:
            img_shape (tuple): The shape of input images (height, width, channels)
            latent_dim (int): Size of encoding
        """
        super(CVAE, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        encoder_input = Input(shape=self.img_shape)
        x = Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(encoder_input)
        x = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        conv_shape = x.shape
        x = Flatten()(x)
        x = Dense(latent_dim * 2)(x)
        self.inference_net = tf.keras.Model(encoder_input, x)
        self.inference_net.summary()

        decoder_input = Input(shape=(latent_dim,))
        x = Dense(units=conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(decoder_input)
        x = Reshape(target_shape=(conv_shape[1], conv_shape[2], conv_shape[3]))(x)
        x = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(filters=self.img_shape[2], kernel_size=3, padding='same', activation='sigmoid')(x)
        self.generative_net = tf.keras.Model(decoder_input, x)
        self.generative_net.summary()

    def encode(self, x):
        """Encode the given batch of images in latent space.

        Args:
            x (ndarray): Batch of images of shape N x `self.img_shape`

        Returns:
            mean, logvar: N Tensors with mean and log variance of the latent space distribution of the given image batch

        """
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparametrize(mean, logvar):
        """Sample from a normal distribution with the given means and log variances using the reparametrization trick

        Args:
            mean (Tensor): N mean vectors
            logvar (Tensor): N log-variance vectors

        Returns:
            N random vectors (as Tensors) sampled from a normal distribution with the given means and variances

        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """Decode the given vectors in latent space into an image of shape `self.img_shape`

        Args:
            z (Tensor): N latent vectors (as Tensors)
            apply_sigmoid (:obj:`int`, optional): Whether to apply a sigmoid activation on the decoded images

        Returns:
            N images decoded from the given latent space representations. Pixel values are ensured to be in [0, 1] only
            if `apply_sigmoid` is True

        """
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


@tf.function
def compute_loss(model, x):
    """Compute the VAE loss (reconstruction + KL divergence) for the given model and batch of images

    For a batch of N images, the loss is computed as reconstruction_loss + KL_loss, where
        reconstruction_loss: Mean (over the N images) of the sum of squared errors between images and their
                            reconstructions
        KL_loss: KL divergence between the encoder predicted distribution and N(0, 1)

    Args:
        model (CVAE): A CVAE object for which the loss has to be computed
        x (Tensor): A batch of N images with shape `model.img_shape`

    Returns:
        Loss of the batch as described above

    """
    print(type(x))
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    decoded = model.decode(z)

    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - decoded), axis=[1, 2, 3]))
    kl_loss = - tf.reduce_mean(0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar), axis=1))
    return reconstruction_loss + kl_loss


@tf.function
def compute_apply_gradients(model, x, opt):
    """Compute the gradients of the model variables with respect to the loss for the given batch and update the model

    Args:
        model (CVAE): A CVAE object that we want to update
        x (Tensor): Batch of N images with shape `model.img_shape`
        opt (tf.keras.optimizers.Optimizer): Keras Optimizer used for applying gradients

    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))


def validate(model, validation_data):
    n_validate = 100
    sampled_indices = np.random.choice(range(len(validation_data)), n_validate)
    sampled = validation_data[sampled_indices]
    mean, logvar = model.encode(sampled)
    decoded = model.decode(mean, apply_sigmoid=False)
    return tf.keras.losses.mean_squared_error(sampled, decoded)


def visualize_results(model, test_data, n_images=20):
    images_per_row = 10
    encoded = model.encode(test_data[0: n_images])
    decoded = model.decode(encoded, apply_sigmoid=False)

    true_images = test_images.reshape((-1, test_images.shape[1], test_images.shape[2]))
    predicted_images = decoded.numpy().reshape((-1, test_images.shape[1], test_images.shape[2]))

    f, axes = plt.subplots(int(n_images / images_per_row) * 2, images_per_row)
    ax_idx = 0
    for i in range(n_images):
        if i % images_per_row == 0 and i > 0:
            ax_idx += 2
        axes[ax_idx, i % images_per_row].imshow(true_images[i])
        axes[ax_idx + 1, i % images_per_row].imshow(predicted_images[i])
    plt.show()


if __name__ == "__main__":
    img_dim = 64
    channels = 1
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = np.array([cv2.resize(img, (img_dim, img_dim)) for img in test_images])
    test_images = np.array([cv2.resize(img, (img_dim, img_dim)) for img in test_images])

    train_images = train_images.reshape(train_images.shape[0], img_dim, img_dim, channels).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], img_dim, img_dim, channels).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000

    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder = CVAE(img_shape=(64, 64, 1), latent_dim=10)
    epochs = 5
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        bar = tf.keras.utils.Progbar(target=len(train_images), stateful_metrics=['validation_loss'])
        for i in range(0, len(train_images), BATCH_SIZE):
            compute_apply_gradients(autoencoder, train_images[i: i + BATCH_SIZE], optimizer)
            val_loss = validate(autoencoder, test_images)
            bar.add(BATCH_SIZE, [('validation_loss', val_loss)])

    visualize_results(autoencoder, test_images, 20)
