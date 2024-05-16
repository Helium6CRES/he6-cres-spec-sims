import numpy as np
import scipy.special as sp
import math

import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc

from he6_cres_spec_sims.constants import *

@np.vectorize
def power_calc(center_x, center_y, frequency, field, trap_radius):

    """Calculates the average cyclotron radiation power (in one direction) in Watts in the
    TE11 mode of an electron undergoing cyclotron motion in the
    cylindrical waveguide around the point (center_x,center_y) with
    frequency in Hz and field in Tesla. 
    See https://arxiv.org/abs/2405.06847 Eqs 18-20 (n,m,h=1)
    """
    
    center_rho = np.sqrt(center_x**2 + center_y**2)

    kc = p11prime / trap_radius
    Rcycl = sc.cyc_radius(sc.freq_to_energy(frequency, field), field, 90)
    
    if Rcycl > trap_radius:
        print("Warning: cyclotron radius greater than trap radius")

    # values in power equation
    omega = 2 * math.pi * frequency
    k = omega / C
    v_perp = Rcycl * omega

    if k >= kc:
        beta = math.sqrt(pow(k, 2) - pow(kc, 2))
    else:
        print("Warning: frequency below TE11 cutoff")
        return 0

    P_lambda = math.pi * beta / (2*kc**2 *mu0 * omega) * (p11prime**2 - 1) * sp.jv(1,p11prime)**2
    power = (Q*v_perp/2.) **2 / P_lambda * (sp.jv(0, kc*center_rho)**2 + sp.jv(2, kc*center_rho)**2) * sp.jvp(1, kc*Rcycl)**2

    return power
