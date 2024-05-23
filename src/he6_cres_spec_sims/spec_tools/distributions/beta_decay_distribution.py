import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc

class BetaDecayDistribution:
    """A class used to produce and interact with the pdf of a beta
    spectrum. Note that the pickled beta spectra are for b = 0. The
    distortion due to b and the necessary renormalization is done here.
    Unnormalized beta spectrum (dNdE_unnormed_SM) based on interpolating
    the cobs_bs look-up table (df). Drew (March 2023) found that 10^4 vs
    10^3 pkled points was a 10^-6 effect and quadratic vs cubic was a
    10^-8 effect. So he went with 10^3 pts and a quadratic interpolation
    (for performance).
    N. Buzinsky after D. Byron (main author)

    Args:
        isotope (str): Isotope to build BetaSpectrum class for. Currently
            the two options are: "Ne19", "He6". More spectra can be pickled
            based on thecobs (Leendert Hayen's beta spectrum library)
            and put in the 'pickled_spectra' directory. Instructions for
            how to pickle the spectra in the correct format can be found
            ____. TODO: WHERE??? FILL IN LATER..
        b (float): Value for the Fierz Inerference coefficient (little b).
    """

    def __init__(self, isotope="He6", b=0):
        # Include all possible isotopes here.
        self.allowed_isotopes = {
            "Ne19": {"Wmax": 5.337377690802349, "Z": 9, "A": 19},
            "He6": {"Wmax": 7.864593361688904, "Z": 2, "A": 6},
        }

    def set_parameters(self, yaml_block):
        # if present, assign from config file
        if "isotope" in yaml_block:
            self.isotope = yaml_block["isotope"]
        if "b" in yaml_block:
            self.b = yaml_block["b"]

        self.load_beta_spectrum()

        # SM spectrum (with all corrections) from thecobs
        # 'extrapolate' will lead to bad behaviour outside of the physical gamma values (1, W0).

        self.dNdE_unnormed_SM = interp1d(
            self.cobs_bs["W"],
            self.cobs_bs["Spectrum"],
            kind="quadratic",
            bounds_error=False,
            fill_value=0,
        )

        main_field = self.config.eventbuilder.main_field

        freq_acceptance_high = self.config.physics.freq_acceptance_high
        freq_acceptance_low = self.config.physics.freq_acceptance_low

        E_start = sc.freq_to_energy( freq_acceptance_high, main_field) + epsilon
        E_stop = sc.freq_to_energy( freq_acceptance_low, main_field) - epsilon

        # Providing the pdf exactly 1 or Wmax leads to errors.
        epsilon = 10**-6
        W_start = (E_start + ME) / ME
        W_stop = (E_stop + ME) / ME
 
        nEnergyIntervals = 10**5
        # Using known PDF, use base class helper for inverse transform sampling inverse CDF
        self.beta_decay_inv_cdf = self.inverse_cdf_helper(self.dNdE, W_start, W_stop, nEnergyIntervals)

    def load_beta_spectrum(self):
        # Key for relative paths from executing file is to use __file__
        pkl_dir = Path(__file__).parent.resolve() / Path("Data")
        self.pkl_spectra_path = pkl_dir / Path(f"{self.isotope}.json")

        with open(self.pkl_spectra_path) as json_file:
            isotope_info = json.load(json_file)
            #print(isotope_info)

        self.cobs_bs = pd.DataFrame.from_dict(isotope_info.pop("cobs_df"))


    def dNdE(self, W):
        """
        Add in the little b distortion. Unnormalized PDF
        """
        return np.clip(self.dNdE_unnormed_SM(W) * (1 + (self.b / W)), 0, np.inf)

    #def fraction_of_spectrum(self):
    #    fraction_of_spectrum, norm_err = integrate.quad( self.dNdE, self.W_start, self.W_stop,)
    #    return fraction_of_spectrum

    def generate(self, size=None):
        """Generate N random samples from dNdE(E) between E_start and E_stop
        """
        # Inverse transform sampling, get N random values between 0 to 1
        rand_values = self.rng.uniform(size=size)
        return self.beta_decay_inv_cdf(rand_values)
