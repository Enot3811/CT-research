"""SWIN patch merger layer."""

import tensorflow as tf
from tensorflow.keras import layers


class SwinPatchMerger(layers.Layer):
    """SWIN layer performing patch merging."""

    def __init__(
        self,
        downsampling: int,
        num_channels: int,
        dimensionality: int,
        **kwargs
    ):
        """
        Initialize SwinPatchMerger layer.

        Parameters
        ----------
        downsampling : int
            Resolution downsampling coefficient. If patch side equals n and
            downsampling equals 2 then size of patch side will decrease to n/2.
        num_channels : int
            Size of channels dimension for projecting after merging.
        dimensionality : int
            Dimensionality of layer.

        Raises
        ------
        ValueError
            Dimensionality of data can be either 2 or 3.
        """
        super().__init__(**kwargs)

        # save all input arguments in some layer config
        self.layer_config = {
            'downsampling': downsampling,
            'num_channels': num_channels,
            'dimensionality': dimensionality,
            **kwargs
        }
        
        if dimensionality in {2, 3}:
            self.dimensionality = dimensionality
        else:
            raise ValueError(
                'Dimensionality of data can be either 2 or 3 but gotten '
                f'{dimensionality}.')

        self.downsampling = downsampling
        self.downsamp_proj = layers.Dense(num_channels)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Merge patches and project them.

        This layer merges patches to produce a hierarchical representation
        in SWIN transformer.
        It's very similar to the patch cutter, but with projecting dense layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Patches input tensor, shape should be `[B, H, W, C]` for 2D
            and `[B, H, W, D, C]` for 3D data.
        args : List
            Not used.
        kwargs : Dict
            Not used.

        Returns
        -------
        tf.Tensor
            Merged and projected patches, shape is
            `[B, H/down_s, W/down_s, new_C]` for 2D and
            `[B, H/down_s, W/down_s, D/down_s, new_C]` for 3D,
            where down_s is `downsampling` and new_C is `num_channels`.
        """
        if self.dimensionality == 2:
            merged_patches = tf.image.extract_patches(
                images=inputs,
                sizes=[1, self.downsampling, self.downsampling, 1],
                strides=[1, self.downsampling, self.downsampling, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
        elif self.dimensionality == 3:
            merged_patches = tf.extract_volume_patches(
                input=inputs,
                ksizes=[
                    1, self.downsampling, self.downsampling,
                    self.downsampling, 1],
                strides=[
                    1, self.downsampling, self.downsampling,
                    self.downsampling, 1],
                padding='VALID'
            )
        else:
            raise ValueError()
        merged_patches = self.downsamp_proj(merged_patches)

        return merged_patches

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self.layer_config

    @classmethod
    def from_config(cls, config):
        """Create layer from the config."""
        return cls(**config)
