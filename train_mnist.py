from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from convolutional_vae import CVAE, compute_apply_gradients


def validation_loss(model, validation_data, n_validate=100):
    """Compute the loss on a random batch from the given validation dataset

    The validation loss is computed averaging the sum of squared errors of images and their encode-decode reconstruction
    over a batch of `n_validate` images randomly drawn from `validation_data`

    Args:
        model (CVAE): CVAE object to be evaluated
        validation_data (ndarray): Numpy array of images of shape N x `model.img_shape`
        n_validate (:obj:`int`, optional) Number of samples to draw from `validation_data`. Defaults to 100

    Returns:
        Mean Squared Error validation loss between the random batch and its encode-decode reconstruction

    """
    sampled_indices = np.random.choice(range(len(validation_data)), n_validate)
    sampled = validation_data[sampled_indices]
    mean, logvar = model.encode(sampled)
    decoded = model.decode(mean, apply_sigmoid=False)
    return tf.keras.losses.mean_squared_error(sampled, decoded)


def visualize_results(model, test_data, n_images=20):
    """Visualize a batch of `n_images` randomly sampled from `test_data` and their encode-decode reconstruction using
    the given model.

    Args:
        model (CVAE):
        test_data (ndarray): Numpy array of N images, each with shape `model.img_shape`
        n_images (:obj:`int`, optional): Number of images to display

    """
    images_per_row = 10
    sampled_indices = np.random.choice(range(len(test_data)), n_images)
    sampled = test_data[sampled_indices]
    encoded = model.encode(sampled)
    decoded = model.decode(encoded, apply_sigmoid=False)

    true_images = sampled.reshape((-1, sampled.shape[1], sampled.shape[2]))
    predicted_images = decoded.numpy().reshape((-1, sampled.shape[1], sampled.shape[2]))

    f, axes = plt.subplots(int(n_images / images_per_row) * 2, images_per_row)
    ax_idx = 0
    for img_idx in range(n_images):
        if img_idx % images_per_row == 0 and img_idx > 0:
            ax_idx += 2
        axes[ax_idx, img_idx % images_per_row].imshow(true_images[img_idx])
        axes[ax_idx + 1, img_idx % images_per_row].imshow(predicted_images[img_idx])
    plt.show()


if __name__ == "__main__":
    img_dim = 64
    channels = 1
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = np.array([cv2.resize(img, (img_dim, img_dim)) for img in test_images])
    test_images = np.array([cv2.resize(img, (img_dim, img_dim)) for img in test_images])

    # Reshape images for Tensorflow
    train_images = train_images.reshape(train_images.shape[0], img_dim, img_dim, channels).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], img_dim, img_dim, channels).astype('float32')

    # Normalizing the images to [0., 1.]
    train_images /= 255.
    test_images /= 255.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000

    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder = CVAE(img_shape=(64, 64, 1), latent_dim=10, beta=3)
    epochs = 50
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        bar = tf.keras.utils.Progbar(target=len(train_images), stateful_metrics=['validation_loss'])
        for i in range(0, len(train_images), BATCH_SIZE):
            compute_apply_gradients(autoencoder, train_images[i: i + BATCH_SIZE], optimizer)
            val_loss = validation_loss(autoencoder, test_images)
            bar.add(BATCH_SIZE, [('validation_loss', val_loss)])

    visualize_results(autoencoder, test_images, 20)
