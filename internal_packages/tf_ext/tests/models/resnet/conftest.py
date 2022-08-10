"""Configure pytest cases with necessary data for ResNet block tests."""

from functools import partial

from tf_ext.models.resnet import ResNetFactory


def pytest_generate_tests(metafunc):
    """
    Parametrize tests with models.
    """
    model_names = [
        'ResNet18',
        'ResNet34',
        'ResNet50'
    ]
    factory = ResNetFactory()

    if 'resnet2d_fe_fn' in metafunc.fixturenames:
        models = [partial(factory.create_resnet, model_name=name,
                          dimensionality=2)
                  for name in model_names]
        metafunc.parametrize('resnet2d_fe_fn', models, ids=model_names)
