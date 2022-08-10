"""
Debug script for inspecting proportional_resize function.
"""


import argparse

import numpy as np
import matplotlib.pyplot as plt

from ndarray_ext.ops import resize


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--origin-shape',
        help='origin shape for test image.',
        nargs=2,
        default=(512, 512),
        type=int)
    parser.add_argument(
        '--resize-shape',
        help='Shape for image resizing.',
        nargs=2,
        default=(128, 64),
        type=int)
    parser.add_argument(
        '--aspect-ratio',
        help='Whether resize function maintains aspect ratio.',
        action='store_true')
    parser.add_argument(
        '--num-tests',
        help='Number of test images for resizing.',
        default=1,
        type=int)

    args = parser.parse_args()

    return args


def main():
    """Application entry point."""
    args = parse_args()
    test_images = [np.random.uniform(0.0, 1.0, args.origin_shape)
                   for _ in range(args.num_tests)]
    resized_images = resize(*test_images,
                            resize_shape=args.resize_shape,
                            aspect_ratio=args.aspect_ratio)
    for tst, res in zip(test_images, resized_images):
        fig = plt.figure(figsize=(5 * 2, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(tst, cmap='gray')
        plt.title('origin')
        plt.subplot(1, 2, 2)
        plt.imshow(res, cmap='gray')
        plt.title('resized')
        fig.show()
        plt.show()


if __name__ == '__main__':
    main()
