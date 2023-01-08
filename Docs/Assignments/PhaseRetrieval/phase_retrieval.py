import numpy as np
from skimage.restoration import unwrap_phase


class Base:
    """Base class to set attributes."""

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        """Overload to compute stuff on initialization."""


class Problem(Base):
    x = np.linspace(-10, 10, 256)
    b = 2 * np.sin(x)
    a = 10 / (x**2 + 2)
    phi = x / 2 - 5 * np.exp(-(x**2))
    rng = np.random.default_rng(seed=2)

    def get_I(self, theta, eta=0, rng=None):
        if rng is None:
            rng = self.rng
        if not np.isscalar(theta):
            theta = np.asarray(theta)[:, np.newaxis]
        I = self.b + self.a * np.cos(self.phi + theta)
        I += eta * rng.normal(size=I.shape)
        return I.T  # Transpose so that columns are the images


class RetrievePhase(Base):
    Np = 10
    
    def get_phase(self, problem, eta=0, rng=None):
        """Return (phase, λb[2]/λb[1])"""
        thetas = 2 * np.pi * np.arange(self.Np) / self.Np
        I = problem.get_I(thetas, eta=eta, rng=rng)
        Nx = len(problem.x)
        assert I.shape == (Nx, self.Np)
        b = I.mean(axis=1)
        Ib = I - b[:, np.newaxis]
        Ub, λb, Vtb = np.linalg.svd(Ib, full_matrices=False)
        ud = Ub[:, :2] * λb[np.newaxis, :2]
        a = np.sqrt((ud**2).sum(axis=1))
        cs = ud / a[:, np.newaxis]
        phase = np.arctan2(*ud.T)  # Assume we know phi(0) = 0
        phase = unwrap_phase(phase)
        
        phase += problem.phi[0] - phase[0]  # Assume we know this!
        return phase, λb[2]/λb[1]
