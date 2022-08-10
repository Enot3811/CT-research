"""SWIN layer containing patch projecting and position embedding."""

import tensorflow as tf
from tensorflow.keras import layers


class SwinInputProjector(layers.Layer):
    """
    SWIN layer performing patches projecting and position embedding adding.
    
    This layer projects input patches from patch_dim to project_dim
    and adds position embedding to them.
    """

    def __init__(
        self,
        project_dim: int,
        patch_count: int,
        dimensionality: int,
        **kwargs
    ):
        """
        Initialize SwinInputProjector layer.

        Parameters
        ----------
        project_dim : int
            Size of dimension for projecting.
        patch_count : int
            Number of patches for creating position embedding.
        dimensionality : int
            Dimensionality of layer.

        Raises
        ------
        ValueError
            Dimensionality of data can be either 2 or 3.
        """
        super().__init__(**kwargs)

        # save all input arguments in some layer config
        self._config = {
            'project_dim': project_dim,
            'patch_count': patch_count,
            'dimensionality': dimensionality,
            **kwargs
        }

        if dimensionality in {2, 3}:
            self.dimensionality = dimensionality
        else:
            raise ValueError(
                'Dimensionality of data can be either 2 or 3 but gotten '
                f'{dimensionality}.')

        self.projector = layers.Dense(units=project_dim)
        self.position_embedding = layers.Embedding(patch_count, project_dim)
        self.positions = tf.range(0, patch_count)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Project patches and add position embedding.

        This layer projects input patches from patch_dim
        to specified project_dim and adds position embedding to them.

        Parameters
        ----------
        inputs : tf.Tensor
            Patches input tensor, shape is [batch_size, num_patches, patch_dim]
        args : List
            Not used.
        kwargs : Dict
            Not used.

        Returns
        -------
        tf.Tensor
            Projected patches with position embedding,
            shape is [batch_size, h_patches, w_patches, project_dim] for 2D and
            [batch_size, h_patches, w_patches, d_patches, project_dim] for 3D
        """
        projections = self.projector(inputs)
        projections = projections + self.position_embedding(self.positions)

        # Reshape from [B x num_patches x C] to [B x H x W (x D) x C]
        shape = tf.shape(projections)
        n = tf.cast(tf.math.round(
            tf.math.pow(
                tf.cast(shape[1], tf.float32), 1 / self.dimensionality)),
            tf.int32)
        new_shape = (shape[0], *(n,) * self.dimensionality, shape[-1])
        projections = tf.reshape(projections, new_shape)
        return projections

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self._config

    @classmethod
    def from_config(cls, config):
        """Create layer from the config."""
        return cls(**config)
