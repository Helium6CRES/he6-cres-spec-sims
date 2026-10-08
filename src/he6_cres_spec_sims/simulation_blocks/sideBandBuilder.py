import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import pandas as pd
from he6_cres_spec_sims.simulation_blocks.trackBuilder import *

class SideBandBuilder:
    """ Constructs list of sidebands and powers from main bands made in trackbuilder
    """

    def __init__(self, config):

        self.config = config

    def run(self, tracks_df, bands):

        print("~~~~~~~~~~~~SideBandBuilder Block~~~~~~~~~~~~~~\n")
        sideband_num = self.config.sidebandbuilder.sideband_num

        frac_total_track_power_cut = self.config.sidebandbuilder.frac_total_track_power_cut
        decay_cell_radius = self.config.eventbuilder.decay_cell_radius
        zKapton =  self.config.eventbuilder.kapton_zs

        out_bands = []

        for tracks_index, row in tracks_df.iterrows():
            sideband_amplitudes = sc.sideband_calc(
                row["hamiltonian"],
                row["magnetic_moment"],
                row["start_guiding_center_rho"],
                row["start_z"],
                row["start_pz"],
                row["axial_freq"],
                self.config.trap_profile,
                zKapton,
                decay_cell_radius,
                num_sidebands=sideband_num,
            )

            sidebands = []

            for band_num in range(-sideband_num, sideband_num + 1):
                if sideband_amplitudes[abs(band_num)] > frac_total_track_power_cut:
                    # fill in new avg_cycl_freq, band_power, band_num
                    start_freq = row["start_freq"] + band_num * row["axial_freq"]
                    # Note that the sideband amplitudes need to be squared to give power.
                    band_power = sideband_amplitudes[abs(band_num)]** 2 * row.track_power

                    freq_shift = start_freq - row["start_freq"]
                    new_track = bands[int(row["event_num"])][int(row["track_num"])].copy()
                    new_track.shift_frequency(freq_shift)
                    new_track.set_band(band_num)
                    new_track.set_power(band_power)
                    sidebands.append(new_track)

            #event_num is used as an index to loop over events
            if int(row["event_num"]) >= len(out_bands):
                out_bands.append([])

            out_bands[int(row["event_num"])] += sidebands

        return out_bands
