import numpy as np
from typing import Optional, Tuple
from scipy.signal import correlate2d


class SensoryFrontend:
    """
    Layer 0 (The Eye) - Sensory Transduction.

    Converts high-dimensional raw data (Images) into low-dimensional
    stimulus vectors for the Mem4ristor.

    Mechanism:
    - Feature Extraction: Uses localized filters (Edge/Blob detection)
      to extract salient features from the image.
    - Projection: Maps these features to the N neurons of the Mem4ristor.
    """
    def __init__(self, output_dim: int, input_shape: Tuple[int, int] = (64, 64), seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.output_dim = output_dim
        self.input_shape = input_shape

        # 1. Initialize Random Filters
        self.num_filters = 16
        self.filter_size = 5
        self.filters = self.rng.normal(0, 1, (self.num_filters, self.filter_size, self.filter_size))

        # 2. Calculate Feature Dimension Dynamically
        # 'valid' correlation reduces dims by filter_size - 1
        conv_h = input_shape[0] - self.filter_size + 1
        conv_w = input_shape[1] - self.filter_size + 1

        # Pooling reduces by stride (4)
        pool_h = conv_h // 4
        pool_w = conv_w // 4

        feature_dim = self.num_filters * pool_h * pool_w

        # Sparse random projection for efficiency
        self.projection = self.rng.normal(0, 0.05, (feature_dim, self.output_dim))

    def _convolve(self, image: np.ndarray, filters: np.ndarray) -> np.ndarray:
        """2D correlation using scipy (replaces pure-Python triple loop).

        Uses scipy.signal.correlate2d with mode='valid', which is
        equivalent to the original sliding-window dot product but runs
        ~100x faster via compiled C code.
        """
        n_filters = filters.shape[0]
        out_h = image.shape[0] - filters.shape[1] + 1
        out_w = image.shape[1] - filters.shape[2] + 1

        output = np.empty((n_filters, out_h, out_w))
        for k in range(n_filters):
            output[k] = correlate2d(image, filters[k], mode='valid')

        return output

    def _pool(self, feature_map: np.ndarray, stride: int = 4) -> np.ndarray:
        """Max pooling via reshape (no Python loops).

        Reshapes each feature map into (out_h, stride, out_w, stride)
        blocks and takes the max over the two stride axes.
        """
        n, h, w = feature_map.shape
        out_h = h // stride
        out_w = w // stride

        # Crop to exact multiple of stride
        cropped = feature_map[:, :out_h * stride, :out_w * stride]
        # Reshape into blocks and take max
        blocks = cropped.reshape(n, out_h, stride, out_w, stride)
        output = blocks.max(axis=(2, 4))

        return output

    def perceive(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image and return the Mem4ristor stimulus.

        Args:
            image (np.ndarray): Grayscale image, shape (H, W). Values [0, 1].

        Returns:
            np.ndarray: Stimulus vector I_eff, shape (output_dim,).
        """
        if image.shape != self.input_shape:
            raise ValueError(f"Image shape {image.shape} != expected {self.input_shape}")

        # 1. Retina -> V1 (Convolution)
        feat_maps = self._convolve(image, self.filters)
        feat_maps = np.maximum(0, feat_maps)  # ReLU

        # 2. V1 -> Pooling
        pooled = self._pool(feat_maps, stride=4)

        # 3. Flatten
        flat_features = pooled.flatten()

        # Ensure projection matches actual dimension
        if flat_features.shape[0] != self.projection.shape[0]:
            raise ValueError(
                f"Feature dim {flat_features.shape[0]} != Projection dim {self.projection.shape[0]}"
            )

        # 4. V1 -> Mem4ristor (Projection)
        stimulus = flat_features @ self.projection

        # Normalize to [-2.0, 2.0] range suited for Mem4ristor
        stimulus = np.tanh(stimulus) * 2.0

        return stimulus

    def generate_test_pattern(self, pattern_type: str) -> np.ndarray:
        """Generates simple geometric shapes for testing."""
        img = np.zeros(self.input_shape)
        h, w = self.input_shape

        if pattern_type == "circle":
            Y, X = np.ogrid[:h, :w]
            center = (h // 2, w // 2)
            dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            img[dist <= h // 3] = 1.0

        elif pattern_type == "square":
            start = h // 4
            end = 3 * h // 4
            img[start:end, start:end] = 1.0

        elif pattern_type == "noise":
            img = self.rng.rand(*self.input_shape)

        return img
