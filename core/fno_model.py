import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class SpectralConv1d(nn.Module):
    modes: int
    width: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_grid, width)
        batch, n_grid, width = x.shape
        
        # Fourier Transform
        x_ft = jnp.fft.rfft(x, axis=1)
        
        # Initialize weights for spectral convolution
        # Shape: (in_channel, out_channel, modes) - Complex
        kernel_shape = (width, width, self.modes)
        weights_real = self.param('weights_real', nn.initializers.xavier_uniform(), kernel_shape)
        weights_imag = self.param('weights_imag', nn.initializers.xavier_uniform(), kernel_shape)
        weights = weights_real + 1j * weights_imag
        
        # Apply multiplication
        # x_ft: (batch, k_max, in_channel) where k_max = n_grid/2 + 1
        # relevant_ft: (batch, modes, in_channel)
        relevant_ft = x_ft[:, :self.modes, :]
        
        # Correct complex multiplication:
        # (batch, modes, in) x (in, out, modes) -> (batch, modes, out)
        # We want to multiply each mode by its own in-out matrix
        # Einsum: b:batch, m:modes, i:in, o:out. Weights are (i, o, m)
        multiplied = jnp.einsum('bmi,iom->bmo', relevant_ft, weights)
        
        # Build output frequency tensor
        out_ft = jnp.zeros_like(x_ft)
        out_ft = out_ft.at[:, :self.modes, :].set(multiplied)
        
        # Inverse Fourier Transform
        x = jnp.fft.irfft(out_ft, n=n_grid, axis=1)
        return x

class MLP(nn.Module):
    width: int
    dropout_rate: float = 0.05

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(self.width)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.width)(x)
        return x

class FNO1d(nn.Module):
    modes: int
    width: int
    out_channels: int
    padding: int = 20
    dropout_rate: float = 0.05

    @nn.compact
    def __call__(self, x, r_grid, train: bool = False):
        # x: (batch, n_grid, channels)
        # r_grid: (batch, n_grid, 1)
        
        # Concat coordinate feature
        x_feat = jnp.concatenate([x, r_grid], axis=-1)
        
        # 1. Project to width
        x = nn.Dense(self.width)(x_feat)
        
        # 2. Zero Padding (Implement Phase 3.5 reinforcement)
        if self.padding > 0:
            x = jnp.pad(x, ((0,0), (self.padding, self.padding), (0,0)), mode='constant')

        # 3. Fourier Layers
        for _ in range(4):
            res = x
            x_spectral = SpectralConv1d(self.modes, self.width)(x)
            x_spectral = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x_spectral)
            x_dense = nn.Dense(self.width)(x)
            x = nn.gelu(x_spectral + x_dense)
        
        # 4. Remove Padding
        if self.padding > 0:
            x = x[:, self.padding:-self.padding, :]
            
        # 5. Project back
        x = MLP(self.width, self.dropout_rate)(x, train=train)
        x = nn.Dense(self.out_channels)(x)
        return x

def create_fno_model(modes=64, width=128, out_channels=6, padding=20, dropout_rate=0.05):
    return FNO1d(modes=modes, width=width, out_channels=out_channels, padding=padding, dropout_rate=dropout_rate)
