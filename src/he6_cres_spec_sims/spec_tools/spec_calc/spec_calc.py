"""spec_calc

This set of calculators takes cres electron properties (such as energy, pitch angle), as
well as information about the trap (instance of a trap_profile object) and outputs other
cres electron properties such as axial frequency, z_turn, average cyclotron frequency

---
Units for all module functions' inputs/outputs:

Energy   : eV [NOTE: THIS IS KINETIC ONLY!]
B-field  : T
Time     : s
Angle    : degrees
Distance : m
Frequency: Hz
Power    : W
Velocity : m/s
Momentum : eV
Magnetic Moment : eV/T
---
"""
import numpy as np

import scipy.integrate as integrate
from scipy.fft import fft
from scipy.optimize import minimize_scalar

from he6_cres_spec_sims.constants import *

import matplotlib.pyplot as plt

def central_diff(f, x, dx=1e-6):
    #Central-difference 1st derivative to replace deprecated scipy.misc.derivative
    return (f(x + dx) - f(x - dx)) / (2 * dx)

# Simple special relativity functions.
# energy is kinetic!

def gamma(energy):
    gamma = (energy + ME) / ME
    return gamma

def instantaneous_gamma(rho, z, hamiltonian, voltage):
    gamma = (hamiltonian - voltage(rho,z)) / ME
    return gamma

def magnetic_moment(energy, pitch_angle, magnetic_field):
    p_perp = momentum(energy) * np.sin(pitch_angle / RAD_TO_DEG)
    return p_perp**2 / (2*ME*magnetic_field)

def hamiltonian(energy, rho, z, voltage):
    H = energy + ME + voltage(rho, z)
    return H

def energy(gamma):
    energy = ME*(gamma - 1)
    return energy

def momentum(energy):
    momentum = np.sqrt(((energy + ME)** 2 - ME**2))
    return momentum

def velocity(energy):
    velocity = momentum(energy) * C / (energy + ME)
    return velocity

# CRES functions.
def energy_to_freq(energy, field):
    """Converts kinetic energy (eV) to cyclotron frequency (Hz)."""
    cycl_freq = Q * field / (2 * PI * gamma(energy) * M)
    return cycl_freq

def freq_to_energy(frequency, field):
    """Calculates energy of beta particle in eV given cyclotron
    frequency in Hz, magnetic field in Tesla, and pitch angle
    at 90 degrees.
    """
    gamma = Q * field / (2 * PI * frequency * M)
    if np.any(gamma < 1):
        gamma = 1
        max_freq = Q * field / (2 * PI * M)
        warning = "Warning: {} higher than maximum cyclotron frequency {}".format(
            frequency, max_freq
        )
        print(warning)
    return gamma * ME - ME

def energy_and_freq_to_field(energy, freq):
    """Converts kinetic energy to cyclotron frequency."""
    field = (2 * PI * gamma(energy) * M * freq) / Q
    return field

def cyc_radius(magnetic_moment, field):
    #Calculates the instantaneous cyclotron radius of a beta electron based on field
    #More convenient to use the conserved quantity μ
    return np.sqrt(2*ME * magnetic_moment / field) / C

def max_radius(magnetic_moment, rho, zTurningPoints, trap_profile):

    """Calculates the maximum cyclotron radius of a beta electron given
    the magnetic_moment, trap_profile, and turning_points
    """

    Bz = lambda z: trap_profile.Bz(rho, z)
    min_field = minimize_scalar(Bz, bracket=zTurningPoints, bounds=zTurningPoints, method="bounded").fun

    max_radius = cyc_radius(magnetic_moment, min_field)
    return max_radius

def min_theta(magnetic_moment, H, rho, zTurningPoints, trap_profile):

    """Calculates the maximum cyclotron radius of a beta electron given
    the magnetic_moment, trap_profile, and turning_points
    """

    Bz = lambda z: trap_profile.Bz(rho, z)
    V = lambda z: trap_profile.voltage(rho, z)

    min_sin2theta = minimize_scalar(lambda z: 2*ME*magnetic_moment*Bz(z) / ((H - V(z))**2 - ME**2), bracket=zTurningPoints, bounds=zTurningPoints, method="bounded").fun

    #XXXX Edge case to fix: I can get sin2theta ~ 1 + 1e-6. It seems numerical & not mathematical. Unsure...
    min_sin2theta = np.clip(min_sin2theta, a_min = 0, a_max = 1.)

    theta_min = np.arcsin(np.sqrt(min_sin2theta))

    #return as degrees
    return theta_min * RAD_TO_DEG

def ode_terminal_events(zKapton):

    turning_event = lambda t, y: y[1]
    turning_event.terminal = 3 #require a full axial cycle (so 3 turning points) before terminating

    kapton_eventL = lambda t, y: y[0] - zKapton[0]
    kapton_eventL.terminal = True

    kapton_eventR = lambda t, y: y[0] - zKapton[1]
    kapton_eventR.terminal = True

    events = [turning_event, kapton_eventL, kapton_eventR]

    return events

def axial_trajectory(H, mu, rho, z_birth, pz_birth, trap_profile, zKapton, TMax=1e-6):
    """ Computes the time series of the beta axial motion over a single
    found by integrating the relevant ODE. Returns [z(t), pz(t)].
    """
    ##### define fields
    Bz = lambda z: trap_profile.Bz(rho,z)
    dBdz = lambda z: central_diff(Bz, z, dx=1e-6)

    V = lambda z: trap_profile.voltage(rho,z)
    dVdz = lambda z: central_diff(V, z, dx=1e-6)
    #########

    events = ode_terminal_events(zKapton)

    ODEgamma = lambda z: (H - V(z)) / ME

    ### Coupled ODE for z-motion: z = y[0], pz = y[1]. z'=pz / M γ. pz' = -mu * B'(z) / M γ + qV'(z)
    #z [m], pz [eV], ME [eV]
    ode = lambda t, y: [y[1] * C / (ME * ODEgamma(y[0])), -(mu / ODEgamma(y[0]) * dBdz(y[0]) + dVdz(y[0])) * C]

    result = None

    try:
        result = integrate.solve_ivp( ode, (0, TMax), [z_birth, pz_birth], events=events, dense_output=True, rtol=1e-9, atol=1e-12)
    except ValueError as e:
        print("FAILED axial trajectory")
        print(f"TMax       = {TMax}")
        print(f"H    = {H}")
        print(f"rho    = {rho}")
        print(f"z_birth    = {z_birth}")
        print(f"pz_birth   = {pz_birth}")
        print("This beta is probably fine! scipy means throwing out! A!")
        return False

        #if we see an event in the "hit kapton" check, ignore the beta
    if len(result.t_events[1]) or len(result.t_events[2]):
        return False

    return result

def get_turning_points_axial_period(ode_result):

    turning_times = ode_result.t_events[0]
    turning_positions = []
    period = None

    for t in turning_times:
        z, _ = ode_result.sol(t)
        turning_positions.append(z)

    turning_positions = np.array(turning_positions)
    turning_positions.sort()

    if len(ode_result.t_events[0]) >= 3:
        period = (ode_result.t_events[0][2] - ode_result.t_events[0][0])
    elif not len(ode_result.t_events[0]):
        #if it doesn't hit the kapti, but doesn't complete a full period in 1 μs - send a warning, but kill the event
        print(f"Axial event integration time TMax not sufficient!")
        return [None, None]
    else:
        return [None, None]

    return [turning_positions, period]

def get_avg_cyc_freq(sol, hamiltonian, rho, axial_period, trap_profile, nSamples = 1000):
    t = np.linspace(0,axial_period/2.,nSamples)
    B = lambda z: trap_profile.Bz(rho,z)
    #returns z(t)
    z_t = sol.sol(t)[0]

    gamma = instantaneous_gamma(rho, z_t, hamiltonian, trap_profile.voltage)
    mean_cycl_freq = np.mean(Q * B(z_t) / (2 * PI * gamma * M))

    return  mean_cycl_freq

def get_b_avg(sol, rho, axial_period, trap_profile, nSamples = 1000):
    t = np.linspace(0,axial_period/2.,nSamples)
    B = lambda z: trap_profile.Bz(rho,z)
    #returns z(t)
    z_t = sol.sol(t)[0]
    return np.mean(B(z_t))

def waveguide_beta(omega, waveguide_radius):
    """  Computes the (waveguide definition) of beta (propagation constant for TE11 mode)
    """
    # fixed experiment parameters
    kc = P11_PRIME / waveguide_radius

    # calculated parameters
    k_wave = omega / C
    beta = np.sqrt(k_wave**2 - kc**2)
    return beta

def sideband_calc(hamiltonian, mu, rho, z_birth, pz_birth, axial_freq, trap_profile, zKapton, waveguide_radius, num_sidebands=7, nHarmonics=128):

    ### Compute particle trajectory over single period
    result = axial_trajectory(hamiltonian, mu, rho, z_birth, pz_birth, trap_profile, zKapton)
    sol = get_sideband_trajectory_samples(result, axial_freq, nHarmonics = 128)

    z = sol[0]
    pz = sol[1]

    sidebands =  FFT_sideband_amplitudes(hamiltonian, rho, axial_freq, pz, z, trap_profile, waveguide_radius, nHarmonics)

    #convention is that num_sidebands doesn't include the mainband
    return sidebands[:num_sidebands + 1]

def get_sideband_trajectory_samples(ode_sol, axial_freq, nHarmonics = 128):
    ###For nHarmonics // 2 sidebands, we need to sample the axial period like the below linspace

    Ta = 1. / axial_freq
    dt = Ta / nHarmonics
    tFFT = np.arange(0,Ta,dt)

    return ode_sol.sol(tFFT)

def instantaneous_frequency(hamiltonian, rho, vz, z, trap_profile):
    """ Computes the instantaneous (angular) frequency as a function of time
        Doppler-free! Useful for computing the mean frequency! (Does this make sense!?)
        I claim the Doppler shifting does not affect the mean cyclotron frequency: Every +vz has a -vz when after turning around
    """

    Bz = lambda z: trap_profile.Bz(rho, z)
    return Q * Bz(z) / (M * instantaneous_gamma(rho, z, hamiltonian, trap_profile.voltage))

def doppler_correction(vz, mean_omega, waveguide_radius):
    beta = waveguide_beta(mean_omega, waveguide_radius)
    phase_vel = mean_omega / beta
    return  1. + vz / phase_vel

def FFT_sideband_amplitudes(hamiltonian, rho, axial_freq, pz, z, trap_profile, waveguide_radius, nHarmonics=128):
    """  Computes sideband amplitudes as a function of axial trajectory, magnetic field profile. Returns list with sidebands
    """
    gamma_z = instantaneous_gamma(rho, z, hamiltonian, trap_profile.voltage)
    #pz, ME in [eV]. vz in [m/s], z in [m]
    vz = pz / (ME * gamma_z) * C

    #Note the pz -> vz argument here
    omega_c = instantaneous_frequency( hamiltonian, rho, vz, z, trap_profile)
    mean_omega_c = np.mean(omega_c)
    omega_c *= doppler_correction(vz, mean_omega_c, waveguide_radius)

    omega_c -= mean_omega_c

    #Euler integration (approximate) of inst. frequency -> CRES phase
    dt = 1./ (nHarmonics * axial_freq)
    Phi = np.cumsum(omega_c) * dt
    expPhi = np.exp(1j * Phi)

    #multiply by v_perp / vtot = sqrt(vtot^2 - vz^2) / vtot
    vTot = C * np.sqrt(1 - 1./gamma_z**2)
    expPhi *= np.sqrt( 1. - (vz/vTot)**2)

    yf = np.abs(fft(expPhi,norm="forward"))
    yf = yf[:nHarmonics//2]
    return yf

def power_from_slope(energy, slope, field):
    """Converts slope, energy, field into the associated cres power."""
    energy_Joules = (energy + ME) / J_TO_EV
    power = slope * (2 * PI) * ((energy_Joules) ** 2) / (Q * field * C**2)

    return power

def df_dt(energy, field, power):
    """Calculates cyclotron frequency rate of change of electron with
    given kinetic energy at field in T radiating energy at rate power.
    """
    energy_Joules = (energy + ME) / J_TO_EV

    slope = (Q * field * C**2) / (2 * PI) * (power) / (energy_Joules) ** 2

    return slope

def power_larmor(field, frequency):

    energy = freq_to_energy(frequency, field)

    return power_larmor_e(field, energy)

def power_larmor_e(field, energy):
    """Takes energy instead of frequency as input. """

    r_c = cyc_radius(magnetic_moment(energy, 90., field), field)
    beta = velocity(energy) / C

    power_larmor = (2 / 3 * Q**2 * C * beta**4 * gamma(energy) ** 4) / ( 4 * PI * EPS_0 * r_c**2)

    return power_larmor
