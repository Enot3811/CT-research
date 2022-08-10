"""Callback saving original, masked and recovered images during mae train."""

import io
from typing import Dict
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ..models import MaskedAutoencoder


class MAEInputsSaver(Callback):
    """
    Callback for MAE-training process which saves input & output data.

    This callback runs model epoch-end state on some fixed number of cases.
    Comparison image contains 4 blocks:
        * original & masked input
        * original & masked output
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        ds: tf.data.Dataset,
        out: Path,
        num_cases: int = 1,
        shuffle_buffer: int = 100
    ):
        """
        Init callback saver.

        This callback has link to the model and dataset from which
        images for monitoring will be gotten.
        Moreover it has parameter for saving images and number of them.

        Note: images will be selected at the init and stayed the same for
        the whole training process.

        Parameters
        ----------
        model : MaskedAutoencoder
            Masked-auto-encoder model.
        ds : tf.data.Dataset
            TF dataset that will be used for monitoring images.
            This dataset contains not less than `num_cases` cases and no
            other additional information.
        out : Path
            Path to the folder where images will be saved.
        num_cases : int, optional
            Number of saved images per epoch.
        shuffle_buffer: int, optional
            Size of the shuffling buffer to select random `num_cases` cases.
        """
        super(MAEInputsSaver, self).__init__()
        self.model = model
        self.ds = ds.unbatch().shuffle(shuffle_buffer).take(
            num_cases
        ).batch(1)
        self.num_cases = num_cases
        self.writer = tf.summary.create_file_writer(str(out))

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Functionality that will be run after each epoch of training.

        It prepares images from original, masked and predicted and saves it.

        Parameters
        ----------
        epoch : int
            Number of training epoch.
        logs : Dict, optional
            Logs from the training. Not used here because no train has run.
        """
        for step, input_data in enumerate(self.ds):
            patches, predictions, mask_indices = self.model(
                input_data
            )
            cur_patches = patches[0].numpy()
            cur_pred = predictions[0].numpy()
            cur_mask_ids = mask_indices[0].numpy()
            fig = self.model.get_mae_plot(cur_patches, cur_pred, cur_mask_ids)
            title = f'epoch #{epoch + 1}'
            with self.writer.as_default():
                tf.summary.image(title, self.__plot_to_image(fig),
                                 step=(step + 1))

    @staticmethod
    def __plot_to_image(figure: plt.figure) -> tf.Tensor:
        """Convert matplotlib figure into tf.Tensor to show."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
