import haiku as hk
import jax.numpy as jnp
from jax import lax
from typing import Optional, Union, Sequence, Tuple
import numpy as np



gains = {
    'linear': 1,
    'relu': (0.5 * (1 - 1/np.pi)) ** 0.5,   # 0.58 (sq=0.34)
    'relu_maxpool': 1 / 1.5,  # found empirically
    'sigmoid': 1 / 4.818253,  # found empirically
}


def get_gain(gain):
    if gain is None:
        return 0

    if isinstance(gain, float):
        return gain

    if isinstance(gain, str):
        if gain[0] == '~':
            return 1 / get_gain(gain[1:])

        return gains[gain]**2

    assert isinstance(gain, list) or isinstance(gain, tuple), \
        f"Unsupported format of 'gain' ({type(gain)})."

    return sum([get_gain(g) for g in gain])

def ___get_gain(gain):
    """
    Parameter skip indicates the variance of a skip connection
    for which we need to correct to achieve unit variance. Depending
    on the skip connection this may be made up of multiples rather
    than a single skip connection.
    """
    if gain is None:
        return 0

    if isinstance(gain, str):
        if gain[0] == '~':
            return 1/gains[gain[1:]]**2

        return gains[gain]**2

    assert isinstance(gain, list) or isinstance(gain, tuple), \
        f"Unsupported format of 'gain' ({type(gain)})."

    return sum([(1/gains[s[1:]]**2 if s[0]=='~' else gains[s]**2) for s in gain])



class LinearSW(hk.Linear):
    """
    Linear layer with Scaled Weight Standardization.

    Linear layer which adjusts its parameters to correct for changes
    in mean and variance of input signal caused by preceeding
    non-linearity.
    """
    # mpr: custom __init__
    def __init__(self, *args, gain='linear', skip=None, **kwargs):
        """ Parameter 'gain' refers to the preceeding non-linearity. """
        super().__init__(*args, **kwargs)
        self._gain = (get_gain(gain) + get_gain(skip))**0.5

    # mpr: copied __call__ from hk.Linear
    def __call__(
            self,
            inputs: jnp.ndarray,
            *,
            precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
          raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
          stddev = 1. / np.sqrt(self.input_size)
          w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        # mpr: normalizing weights w
        # -------------------------------------------------------------------------
        fan_in = w.shape[0]
        mean = jnp.mean(w, axis=[0], keepdims=True)
        var = jnp.var(w, axis=[0], keepdims=True)
        w = (w - mean) / (var * fan_in + 1e-4)**0.5
        w = w / self._gain
        # -------------------------------------------------------------------------

        out = jnp.dot(inputs, w, precision=precision)

        if self.with_bias:
          b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
          b = jnp.broadcast_to(b, out.shape)
          out = out + b

        return out


class ConvNDSW(hk.ConvND):
    """ ConvND with Scaled Weight Standardization. """
    def __init__(self, *args, gain='linear', skip=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gain = (get_gain(gain) + get_gain(skip))**0.5

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Connects ``ConvND`` layer.
        Args:
          inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
            or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
          precision: Optional :class:`jax.lax.Precision` to pass to
            :func:`jax.lax.conv_general_dilated`.
        Returns:
          An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
            unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
            and rank-N+2 if batched.
        """
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
          raise ValueError(f"Input to ConvND needs to have rank in {allowed_ranks},"
                           f" but input has shape {inputs.shape}.")

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
          inputs = jnp.expand_dims(inputs, axis=0)

        if inputs.shape[self.channel_index] % self.feature_group_count != 0:
          raise ValueError(f"Inputs channels {inputs.shape[self.channel_index]} "
                           f"should be a multiple of feature_group_count "
                           f"{self.feature_group_count}")
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels)

        if self.mask is not None and self.mask.shape != w_shape:
          raise ValueError("Mask needs to have the same shape as weights. "
                           f"Shapes are: {self.mask.shape}, {w_shape}")

        w_init = self.w_init
        if w_init is None:
          fan_in_shape = np.prod(w_shape[:-1])
          stddev = 1. / np.sqrt(fan_in_shape)
          w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        if self.mask is not None:
          w *= self.mask

        # mpr: normalizing weights w
        # -------------------------------------------------------------------------
        fan_in = np.prod(w.shape[:-1])
        mean = jnp.mean(w, axis=[0, 1, 2], keepdims=True)  # w have format hwco
        var = jnp.var(w, axis=[0, 1, 2], keepdims=True)
        w = (w - mean) / (var * fan_in + 1e-5)**0.5
        w = w / self._gain
        # -------------------------------------------------------------------------

        out = lax.conv_general_dilated(inputs,
                                       w,
                                       window_strides=self.stride,
                                       padding=self.padding,
                                       lhs_dilation=self.lhs_dilation,
                                       rhs_dilation=self.kernel_dilation,
                                       dimension_numbers=self.dimension_numbers,
                                       feature_group_count=self.feature_group_count,
                                       precision=precision)

        if self.with_bias:
          if self.channel_index == -1:
            bias_shape = (self.output_channels,)
          else:
            bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
          b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
          b = jnp.broadcast_to(b, out.shape)
          out = out + b

        if unbatched:
          out = jnp.squeeze(out, axis=0)
        return out


class Conv2DSW(ConvNDSW):
    """Two dimensional convolution."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_spatial_dims=2, **kwargs) 
