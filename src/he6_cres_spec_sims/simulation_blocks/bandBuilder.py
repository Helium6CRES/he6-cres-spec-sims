import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import pandas as pd

class BandBuilder:
    """ Constructs list of sidebands and powers for trapped "segments", between scatters
    """

    def __init__(self, config):

        self.config = config

    def run(self, segments_df):

        print("~~~~~~~~~~~~BandBuilder Block~~~~~~~~~~~~~~\n")
        sideband_num = self.config.bandbuilder.sideband_num
        magnetic_modulation = self.config.bandbuilder.magnetic_modulation
        harmonic_sidebands = self.config.bandbuilder.harmonic_sidebands

        frac_total_segment_power_cut = (
            self.config.bandbuilder.frac_total_segment_power_cut
        )
        total_band_num = sideband_num * 2 + 1

        band_list = []

        for segment_index, row in segments_df.iterrows():

            if harmonic_sidebands:
                sideband_amplitudes = sc.sideband_calc(
                    row["energy"],
                    row["rho_center"],
                    row["avg_cycl_freq"],
                    row["axial_freq"],
                    row["zmax"],
                    self.config.trap_profile,
                    magnetic_modulation=magnetic_modulation,
                    num_sidebands=sideband_num,
                )[0]
            else:
                sideband_amplitudes = sc.anharmonic_sideband_calc(
                    row["energy"],
                    row["center_theta"],
                    row["rho_center"],
                    row["avg_cycl_freq"],
                    row["axial_freq"],
                    row["zmax"],
                    self.config.trap_profile,
                    magnetic_modulation=magnetic_modulation,
                    num_sidebands=sideband_num,
                )[0]

            for i, band_num in enumerate(range(-sideband_num, sideband_num + 1)):

                if sideband_amplitudes[i][1] < frac_total_segment_power_cut:
                    continue
                else:
                    # copy segment in order to fill in band specific values
                    row_copy = row.copy()

                    # fill in new avg_cycl_freq, band_power, band_num
                    # TODO: properly determine band power stop.
                    row_copy["avg_cycl_freq"] = sideband_amplitudes[i][0]
                    # Note that the sideband amplitudes need to be squared to give power.
                    row_copy["band_power_start"] = (
                        sideband_amplitudes[i][1] ** 2 * row.segment_power
                    )
                    row_copy["band_power_stop"] = row_copy["band_power_start"]
                    row_copy["band_num"] = band_num

                    # append to band_list, as it's better to grow a list than a df
                    band_list.append(row_copy.tolist())

        bands_df = pd.DataFrame(band_list, columns=segments_df.columns)

        return bands_df
