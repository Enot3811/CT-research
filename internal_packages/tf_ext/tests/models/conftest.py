"""Configure pytest cases with necessary data for `models` sub-package."""

from functools import partial

import numpy as np

import tf_ext.models
from tf_ext.models.resnet import ResNetFactory


class ModelDataFactory:
    """Factory of the model & its random test data."""

    @staticmethod
    def create_resnet34_2d():
        """Create ResNet34 model & its random 2D test data."""
        input_shape = (224, 224, 3)
        batch_size = 2
        test_data = np.random.uniform(
            size=(batch_size, *input_shape)).astype('float32')
        factory = ResNetFactory()
        model = factory.create_resnet('ResNet34', dimensionality=2)
        return model, test_data

    @staticmethod
    def create_resnet34_3d():
        """Create ResNet34 model & its random 3D test data."""
        input_shape = (96, 96, 96, 1)
        batch_size = 2
        test_data = np.random.uniform(
            size=(batch_size, *input_shape)).astype('float32')
        factory = ResNetFactory()
        model = factory.create_resnet('ResNet34', dimensionality=3)
        return model, test_data

    @staticmethod
    def create_resnet50_2d():
        """Create ResNet50 model & its random 2D test data."""
        input_shape = (224, 224, 3)
        batch_size = 2
        test_data = np.random.uniform(
            size=(batch_size, *input_shape)).astype('float32')
        factory = ResNetFactory()
        model = factory.create_resnet('ResNet50', dimensionality=2)
        return model, test_data

    @staticmethod
    def create_resnet50_3d():
        """Create ResNet50 model & its random 3D test data."""
        input_shape = (96, 96, 96, 1)
        batch_size = 2
        test_data = np.random.uniform(
            size=(batch_size, *input_shape)).astype('float32')
        factory = ResNetFactory()
        model = factory.create_resnet('ResNet50', dimensionality=3)
        return model, test_data

    def create(self, model_name):
        """Create custom-model & its random test data from `model_name`."""
        if model_name == 'ResNet34_2D':
            return self.create_resnet34_2d()
        elif model_name == 'ResNet34_3D':
            return self.create_resnet34_3d()
        elif model_name == 'ResNet50_2D':
            return self.create_resnet50_2d()
        elif model_name == 'ResNet50_3D':
            return self.create_resnet50_3d()
        else:
            raise NotImplementedError(f'{model_name=} has no test-data '
                                      f'creation method.')


def pytest_generate_tests(metafunc):
    """
    Parametrize tests with input/output images.
    """
    if 'model_data_fn' in metafunc.fixturenames:
        factory = ModelDataFactory()
        data = []
        ids = []
        for model_name, _ in tf_ext.models.CUSTOM_OBJECTS.items():
            if model_name == 'ResNet':
                resnet_names = (
                    'ResNet34_2D',
                    'ResNet50_2D',
                    'ResNet34_3D',
                    'ResNet50_3D'
                )
                for resnet_name in resnet_names:
                    fn = partial(factory.create, model_name=resnet_name)
                    data.append(fn)
                    ids.append(resnet_name)
            else:
                fn = partial(factory.create, model_name=model_name)
                data.append(fn)
                ids.append(model_name)
        metafunc.parametrize('model_data_fn', data, ids=ids)
