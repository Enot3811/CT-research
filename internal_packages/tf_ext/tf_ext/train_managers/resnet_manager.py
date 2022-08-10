"""Train manager for ResNet model."""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from pynative_ext.os import get_repo_info

from .classifier_manager import ClassifierManager
from ..models import ResNet


class ResNetManager(ClassifierManager):
    """
    Train manager for ResNet model.

    Mostly all complicated logic is implemented inside ResNet-model class,
    manager need to define just optimization process & metrics.
    """

    @staticmethod
    def load_model(out_path: Path, epoch: int) -> ResNet:
        """
        Load ResNet-model from serialized output information.

        Parameters
        ----------
        out_path : Path
            Out-path of the ResNet training process. It's expected that this
            folder contains at least ``checkpoints`` sub-folder & model config
            as ``JSON`` file.
        epoch : int
            Number of epoch to load.

        Returns
        -------
        ResNet:
            ResNet-model in keras-format with weights from specified `epoch`.
        """
        model_config_path = out_path.joinpath('model_config.json')
        ckpts_path = out_path.joinpath('checkpoints')
        if not model_config_path.exists():
            raise RuntimeError(f'No "model_config.json" at: {out_path}.')
        if not ckpts_path.exists():
            raise RuntimeError(f'No "checkpoints" folder at: {out_path}.')
        all_ckpts = list(ckpts_path.glob('*'))
        if len(all_ckpts) == 0:
            raise RuntimeError(f'No checkpoints at: {ckpts_path}.')

        logger.info('Creating ResNet model...')
        with open(model_config_path, 'r') as fd:
            model_config = json.load(fd)
        resnet_model = ResNet.from_config(model_config)
        logger.info('Check call...')
        random_data = np.random.uniform(
            size=resnet_model.get_config()['input_shape'])
        resnet_model(random_data[None, ...])
        logger.info('OK.')

        logger.info('Checking commit information of the weights & current...')
        cur_repo_info = get_repo_info()
        loaded_repo_info = model_config['commit']
        if cur_repo_info != loaded_repo_info:
            logger.warning(
                'Loading checkpoint & created ResNet architecture were '
                'generated on different repo versions.\n'
                f'Checkpoint:\n"{loaded_repo_info}". '
                f'Created model:\n"{cur_repo_info}"')
        logger.info('OK.')

        logger.info('Searching requested checkpoint...')
        starts_pattern = f'weights.{epoch:04d}-'
        suitable_ckpt_file = list(filter(
            lambda ckpt: ckpt.name.startswith(starts_pattern), all_ckpts
        ))
        if len(suitable_ckpt_file) != 1:
            raise RuntimeError(
                f'Count of weights that satisfy to requested {epoch=}) '
                f'is {len(suitable_ckpt_file)} (need only one).')
        else:
            suitable_ckpt_path = suitable_ckpt_file[0]
        logger.info(f'Checkpoint path is found: "{suitable_ckpt_path}".')
        logger.info(f'Loading weights from "{suitable_ckpt_path}"...')
        resnet_model.load_weights(suitable_ckpt_path)
        logger.info('OK.')

        return resnet_model
