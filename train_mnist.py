from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from convolutional_vae import CVAE
import os


@tf.function
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

    sampled_indices = tf.random.uniform(shape=[n_validate], minval=0, maxval=validation_data.shape[0], dtype=tf.int64)
    sampled = tf.gather(validation_data, sampled_indices)
    mean, logvar = model.encode(sampled)
    decoded = model.decode(mean, apply_sigmoid=False)
    return tf.reduce_mean(tf.reduce_sum(tf.square(sampled - decoded), axis=[1, 2, 3]))


def visualize_results(model, test_data, n_images=20, images_per_row=10):
    """Visualize a batch of `n_images` randomly sampled from `test_data` and their encode-decode reconstruction using
    the given model.

    Args:
        model (CVAE):
        test_data (ndarray): Numpy array of N images, each with shape `model.img_shape`
        n_images (:obj:`int`, optional): Number of images to display
        images_per_row (:obj:`int`, optional): Number of images per row. Must be a submultiple of `n_images`

    """
    sampled_indices = tf.random.uniform(shape=[n_images], minval=0, maxval=test_data.shape[0], dtype=tf.int64)
    sampled = tf.gather(test_data, sampled_indices)
    encoded = model.encode(sampled)
    decoded = model.decode(encoded, apply_sigmoid=True)

    true_images = tf.reshape(sampled, (-1, sampled.shape[1], sampled.shape[2]))
    predicted_images = tf.reshape(decoded, (-1, sampled.shape[1], sampled.shape[2]))

    f, axes = plt.subplots(int(n_images / images_per_row) * 2, images_per_row)
    ax_idx = 0
    for img_idx in range(n_images):
        if img_idx % images_per_row == 0 and img_idx > 0:
            ax_idx += 2
        axes[ax_idx, img_idx % images_per_row].imshow(true_images[img_idx])
        axes[ax_idx + 1, img_idx % images_per_row].imshow(predicted_images[img_idx])
    plt.show()


def load_mnist_dataset(img_size, normalize=True):
    """Load MNIST dataset returning train and test dataset of normalized images of the desired shape

    Args:
        img_size (tuple): Size (width, height) to which images are resized.
        normalize (:obj:`bool`, optional)

    Returns:
        x_train, x_test: Train and test dataset respectively resized to the given size, reshaped for tensorflow
                         (width, height, 1), normalized in [0., 1.] if requested.

    """
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Upsample images to desired shape
    x_train = np.array([cv2.resize(img, img_size) for img in x_train]).astype('float32')
    x_test = np.array([cv2.resize(img, img_size) for img in x_test]).astype('float32')

    # Reshape images for Tensorflow
    x_train = tf.reshape(x_train, (-1, img_size[0], img_size[1], 1))
    x_test = tf.reshape(x_test, (-1, img_size[0], img_size[1], 1))

    if normalize:
        x_train /= 255.
        x_test /= 255.

    return x_train, x_test


def train(model, train_data, optimizer, validation_data=None, epochs=50, batch_size=100,
          save_checkpoint=True, checkpoint_dir='checkpoints', checkpoint_save_freq=1):
    """

    Args:
        model (CVAE): CVAE model to be trained
        train_data (Tensor or ndarray): Images for training, of shape N x `model.img_shape`
        optimizer (tf.keras.optimizers.Optimizer): Any implementation of keras Optimizer class
        validation_data (Tensor or ndarray): Images for training, of shape N x `model.img_shape`. If this field is
                                            present, validation will be performed every 20 batches and reported
        epochs (:obj:`int`, optional): Number of training iterations over the entire dataset. Defaults to 50
        batch_size (:obj:`int`, optional): Size of batch (should be >= 100). Defaults to 100
        save_checkpoint (:obj:`bool`, optional): If true, a checkpoint of the model will be saved in the given dir
        checkpoint_dir (:obj:`str`, optional): Directory in which to save checkpoints. Defaults to `checkpoints/`
        checkpoint_save_freq (:obj:`int`, optional): Frequency (in epochs) of checkpoint saving


    """
    # Metrics
    train_rec_metr = tf.keras.metrics.Mean()
    train_kl_metr = tf.keras.metrics.Mean()
    val_metr = tf.keras.metrics.Mean()
    validation_freq = 20  # Validate every 20 batches
    stateful_metrics = ['train_reconstruction_loss', 'train_kl_loss']
    if validation_data is not None:
        stateful_metrics.append('validation_reconstruction_loss')

    # Checkpoint
    if save_checkpoint:
        checkpoint_saver = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Train
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        bar = tf.keras.utils.Progbar(target=len(train_data), stateful_metrics=stateful_metrics)
        for i in range(0, len(train_data), batch_size):
            rc_loss, kl_loss, loss = model.compute_apply_gradients(train_data[i: i + batch_size], optimizer)
            train_rec_metr.update_state(rc_loss)
            train_kl_metr.update_state(kl_loss)
            metrics = [('train_reconstruction_loss', train_rec_metr.result().numpy()),
                       ('train_kl_loss', train_kl_metr.result().numpy())]
            if validation_data is not None and i % validation_freq == 0:
                val_loss = validation_loss(model, validation_data)
                val_metr.update_state(val_loss)
                metrics.append(('validation_reconstruction_loss', val_metr.result().numpy()))
            bar.add(batch_size, metrics)
        if save_checkpoint and i % checkpoint_save_freq == 0:
            checkpoint_saver.save(file_prefix=os.path.join(checkpoint_dir, 'cpk'))
        train_rec_metr.reset_states()
        train_kl_metr.reset_states()
        val_metr.reset_states()


if __name__ == "__main__":
    # Load data
    img_shape = (32, 32)
    train_images, test_images = load_mnist_dataset(img_size=img_shape)

    # Create model
    autoencoder = CVAE(img_shape=train_images.shape[1:], latent_dim=32, beta=3.0)
    adam = tf.keras.optimizers.Adam()

    checkpoint_directory = 'checkpoints/'
    checkpoint = tf.train.Checkpoint(optimizer=adam, model=autoencoder)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # train(model=autoencoder, train_data=train_images, optimizer=adam, validation_data=test_images)
    visualize_results(autoencoder, test_images, n_images=20, images_per_row=10)
