# Imports.
import sys
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from he6_cres_spec_sims.constants import ME

# ---------- Beta Spectrum class used in he6-cres-spec-sims ---------------


class BetaSpectrum:
    """A class used to produce and interact with the pdf of a beta
    spectrum. Note that the pickled beta spectra are for b = 0. The
    distortion due to b and the necessary renormalization is done here.
    Unnormalized beta spectrum (dNdE_unnormed_SM) based on interpolating
    the cobs_bs look-up table (df). Drew (March 2023) found that 10^4 vs
    10^3 pkled points was a 10^-6 effect and quadratic vs cubic was a
    10^-8 effect. So he went with 10^3 pts and a quadratic interpolation
    (for performance).

    Args:
        isotope (str): Isotope to build BetaSpectrum class for. Currently
            the two options are: "Ne19", "He6". More spectra can be pickled
            based on thecobs (Leendert Hayen's beta spectrum library)
            and put in the 'pickled_spectra' directory. Instructions for
            how to pickle the spectra in the correct format can be found
            ____. TODO: WHERE??? FILL IN LATER..
        b (float): Value for the Fierz Inerference coefficient (little b).
    """

    def __init__(self, isotope, b=0) -> None:

        self.isotope = isotope

        self.load_beta_spectrum()

        self.W0 = self.isotope_info["W0"]
        self.A = self.isotope_info["A"]
        self.Z = self.isotope_info["Z"]
        self.decay_type = self.isotope_info["decay_type"]
        self.b = b

        self.dNdE_norm = None

        # SM spectrum as calculated by thecobs. The cobs_bs df contains
        # all of the corrections included. Note that using fill_value
        # ='extrapolate' will lead to bad behaviour outside of the physical
        # gamma values (1, W0).
        self.dNdE_unnormed_SM = interp1d(
            self.cobs_bs["W"],
            self.cobs_bs["Spectrum"],
            kind="quadratic",
            bounds_error=False,
            fill_value=0,
        )

    def load_beta_spectrum(self):

        # Key for relative paths from executing file is to use __file__
        pkl_dir = Path(__file__).parent.resolve() / Path("pickled_spectra")
        self.pkl_spectra_path = pkl_dir / Path(f"{self.isotope}.json")

        with open(self.pkl_spectra_path) as json_file:
            isotope_info = json.load(json_file)

        self.cobs_bs = pd.DataFrame.from_dict(isotope_info.pop("cobs_df"))

        self.isotope_info = isotope_info

        return None

    def dNdE(self, W):
        """
        Normalized beta spectrum. If self.dNdE_norm is None, then the
        normalization is calculated and saved to self.dNdE_norm.
        """

        if self.dNdE_norm is None:

            norm, norm_err = integrate.quad(
                self.dNdE_unnormed, 1.0, self.W0, epsrel=1e-6
            )
            self.dNdE_norm = norm

        return np.clip(self.dNdE_unnormed(W) / self.dNdE_norm, 0, np.inf)

    def dNdE_unnormed(self, W):
        """
        Add in the littleb distortion before you renormalize the spectrum.
        """
        return self.dNdE_unnormed_SM(W) * (1 + (self.b / W))


    def energy_samples(self, n, E_start, E_stop, rand_seed):

            """Produce n random samples from dNdE(E) between E_start and E_stop assuming constant spacing in Ws.
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
            epsilon = 10**-6
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