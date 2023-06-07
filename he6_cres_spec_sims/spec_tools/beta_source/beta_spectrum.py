# Imports.
import sys
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

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
