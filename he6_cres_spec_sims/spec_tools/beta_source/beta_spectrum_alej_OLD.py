import numpy as np
from scipy import special
from mpmath import fp
import matplotlib.pyplot as plt
from scipy import integrate

# Physical constants.
ALPHA = 1.0 / 137.0360
HBARC = 197.3269804  # (Drew) What is this unit? eV * mm ?
MPROT = 1836.15267389  # in units of ME
ME = 5.10998950e5  # Electron rest mass (eV).

# Math constants.
PI = np.pi

# ---------- Simple functions used in the Beta Spectrum class ---------------


def p(W):
    return np.sqrt(W**2 - 1)


def beta(W):
    return p(W) / W


def eta(W, Z):
    return ALPHA * Z * W / p(W)


def Gamma(Z):
    return np.sqrt(1 - (ALPHA * Z) ** 2)


def R(A):
    return 1.2 * A ** (1.0 / 3.0) * 1e-6 * ME / HBARC


# Vectorize the ploylog function
polylog = np.vectorize(fp.polylog)


# ---------- Beta Spectrum class used in he6-cres-spec-sims ---------------


class BetaSpectrum:
    """A class used to produce and interact with the pdf of a beta spectrum.

    Args:
        isotope_info (dict): contains the isotope information necessary
            to build the beta spectrum. Examples that can be used to create
            spectra for Ne19 and He6 are:
            ne19_info = {"W0": 5.337377690802349, "Z": 8, "A": 19, "decay_type": "+", "b": 0}
            he6_info =  {"W0": 7.859114, "Z": 3, "A": 6, "decay_type": "-", "b": 0}
            Note: Z and A refer to the decaying isotope, NOT the daughter nucleus.
    """

    def __init__(self, isotope_info: dict) -> None:

        self.dNdE_norm = None
        self.Wmax = isotope_info["W0"]

        self.A = isotope_info["A"]
        self.decay_type = isotope_info["decay_type"]
        self.Z = (
            isotope_info["Z"] + 1
            if self.decay_type == "-"
            else -(isotope_info["Z"] - 1)
        )

        if "b" in isotope_info:
            self.b = isotope_info["b"]
        else:
            self.b = 0

    def dNdE(self, W):
        """
        Normalized beta spectrum. If self.dNdE_norm is None, then the
        normalization is calculated and saved to self.dNdE_norm.
        """
        if self.dNdE_norm is None:

            norm, norm_err = integrate.quad(
                self.dNdE_unnormed,
                1.0,
                self.Wmax,
                args=(self.Wmax, self.Z, self.A, self.b),
            )
            self.dNdE_norm = norm

        return self.dNdE_unnormed(W, self.Wmax, self.Z, self.A, self.b) / self.dNdE_norm

    def dNdE_unnormed(self, W, Wmax, Z, A, b):
        """
        Unnormalized beta spectrum.
        """
        return (
            self.dNdE0(W, Wmax)  # Phase space.
            * self.F(W, Z, A)  # Naive phase space term plus Fermi function.
            * (1.0 + self.Delta(W, Z, A))  # Radiative effects.
            * (1.0 + ALPHA / (2.0 * PI) * self.g(W, Wmax))  # Radiative effects.
            * (1 + (b / W))  # Fierz interference.
        )

    def dNdE0(self, W, Wmax):
        """
        Phase space.
        """
        return p(W) * W * (Wmax - W) ** 2

    def g(self, W, Wmax):
        """
        Term in radiative correction.
        """

        const = 3.0 * np.log(MPROT) - (3.0 / 4.0)
        deltaW = (
            4
            * (np.arctanh(beta(W)) / beta(W) - 1.0)
            * ((Wmax - W) / (3 * W) - 3 / 2 + np.log(2 * (Wmax - W)))
        )
        polybeta = -4.0 / beta(W) * polylog(2, (2 * beta(W)) / (1 + beta(W)))
        deltaW2 = (
            1.0
            / beta(W)
            * np.arctanh(beta(W))
            * (2.0 * (1.0 + (beta(W)) ** 2) + (Wmax - W) ** 2 / (6 * W**2))
        )
        return const + deltaW + polybeta + deltaW2

    def F(self, W, Z, A):
        """
        Fermi function.
        """

        return (
            2
            * (1 + Gamma(Z))
            * (2 * p(W) * R(A)) ** (2 * (Gamma(Z) - 1))
            * np.exp(PI * eta(W, Z))
            * np.abs(
                special.gamma(Gamma(Z) + 1j * eta(W, Z))
                / special.gamma(2 * Gamma(Z) + 1)
            )
            ** 2
        )

    def Delta(self, W, Z, A):
        """
        Recoil order.
        """
        return (2 * self.L0(W, Z, A)) / (1 + Gamma(Z)) - 1.0

    def L0(self, W, Z, A):
        """
        Recoil order.
        """
        r = R(A)
        # breaking it up by powers of ALPHAZ
        az1 = (
            ALPHA
            * Z
            * (
                28.0 / 15.0 * W * r
                - 8.0 / 15.0 * r / W
                + ((W * r) ** 3) * (1.0 + 9.0 / 2.0 * ALPHA * Z)
            )
        )
        az2 = 7.0 / 20.0 * (ALPHA * Z) ** 2
        az3 = -0.5 * W * r * (ALPHA * Z) ** 3
        az4 = -3.0 / 8.0 * W * r * (ALPHA * Z) ** 4
        az6 = (ALPHA * Z) ** 6 * (1.0 / 20.0 + 10 * (W * r) ** 2)
        az8 = -3.0 / 8.0 * (ALPHA * Z) ** 8
        return 1.0 + az1 + az2 + az3 + az4 + az6 + az8 - 1.0 / 3.0 * (p(W) * r) ** 2

    def energy_samples(self, n, E_start, E_stop, rand_seed):
        """Produce n random samples from dNdE(E) between E_start and E_stop.
        Also return the fraction of the entire spectrum this accounts for."""

        rng = np.random.default_rng(rand_seed)

        fraction_of_spectrum = None

        W_start = (E_start + ME) / ME
        W_stop = (E_stop + ME) / ME

        if fraction_of_spectrum == None:
            fraction_of_spectrum, norm_err = integrate.quad(
                self.dNdE,
                W_start,
                W_stop,
            )

        # Providing the pdf exactly 1 or Wmax leads to errors.
        epsilon = 10**-10
        energy_intervals = 10**5
        Es = np.linspace(E_start + epsilon, E_stop - epsilon, energy_intervals)

        Ws = (Es + ME) / ME
        pdf = self.dNdE(Ws)

        # get cumulative distribution from 0 to 1
        cumpdf = np.cumsum(pdf)
        cumpdf *= 1 / cumpdf[-1]

        # input random values
        randv = rng.uniform(size=n)

        # find where random values would go
        idx1 = np.searchsorted(cumpdf, randv)
        # get previous value, avoiding division by zero below
        idx0 = np.where(idx1 == 0, 0, idx1 - 1)
        idx1[idx0 == 0] = 1

        # do linear interpolation in x
        frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
        energy_samples = Es[idx0] * (1 - frac1) + Es[idx1] * frac1

        return energy_samples, fraction_of_spectrum


isotopes = {
    "Ne19": {
        "Wmax": 5.339539,
        "Z": 10,
        "A": 19,
        "decay_type": "+",
        "transition_type": "Gamow-Teller",
        "mixing_ratio": 2.22,
        "R": 2.9,
        "bAc": 5,
        "dAc": 2,
        "Lambda": 5,
    },
    "He6": {
        "Wmax": 5.339539,
        "Z": 10,
        "A": 19,
        "decay_type": "+",
        "transition_type": "Gamow-Teller",
        "mixing_ratio": 2.22,
        "radius": 2.9,
        "bAc": 5,
        "dAc": 2,
        "Lambda": 5,
    },
}
