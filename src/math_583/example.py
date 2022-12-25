"""Example module with a simple function."""
import numpy.fft
import numpy as np

__all__ = ['differentiate_fft']


def differentiate_fft(f, dx=1.0):
    """Return the derivative of `f` using the FFT.

    Arguments
    ---------
    f : array-like
        1D array with the function to be differentiated.
    dx : float
        Lattice spacing.
    """
    k = 2*np.pi * np.fft.fftfreq(len(f), dx)
    df = np.fft.ifft(1j*k * np.fft.fft(f))
    if np.isrealobj(f):
        df = df.real
    return df
