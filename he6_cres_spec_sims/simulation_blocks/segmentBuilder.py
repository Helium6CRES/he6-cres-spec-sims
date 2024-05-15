from .eventBuilder import *
import he6_cres_spec_sims.spec_tools.spec_calc.power_calc as pc

class SegmentBuilder:
    """ Constructs a list of tracks/ segments (interrupted by scatters) making up the trapped event
    """

    def __init__(self, config):

        self.config = config
        self.eventbuilder = EventBuilder(config)

    def run(self, trapped_event_df):
        """TODO: DOCUMENT"""
        print("~~~~~~~~~~~~SegmentBuilder Block~~~~~~~~~~~~~~\n")
        # Empty list to be filled with scattered segments.
        scattered_segments_list = []

        for event_index, event in trapped_event_df.iterrows():
            if event_index % 25 == 0:
                print("\nScattering Event :", event_index)

            # Assign segment 0 of event with a segment_length.
            event["segment_length"] = self.segment_length()

            # Fill the event with computationally intensive properties.
            event = self.fill_in_properties(event)

            # Extract position and center theta from event.
            center_x, center_y = event["center_x"], event["center_y"]
            rho_pos = event["initial_rho_pos"]
            phi_pos = event["initial_phi_pos"]
            zpos = 0
            center_theta = event["center_theta"]
            phi_dir = event["initial_phi_dir"]

            # Extract necessary parameters from event.
            # TODO(byron): Note that it is slightly incorrect to assume the power doesn't change as time passes.
            energy = event["energy"]
            energy_stop = event["energy_stop"]
            event_num = event["event_num"]
            beta_num = event["beta_num"]

            segment_radiated_power = event["segment_power"] * 2

            # Append segment 0 to scattered_segments_list because segment 0 is trapped by default.
            scattered_segments_list.append(event.values.tolist())

            # Begin with trapped beta (segment 0 of event).
            is_trapped = True
            jump_num = 0

            # The loop breaks when the trap condition is False or the jump_num exceeds self.jump_num_max.
            # This forces us to check to see that the current beta is trapped even when we don't want any scattering.
            # TODO: Improve the above issue. This will greatly improve run times.
            while True:

                if jump_num >= self.config.segmentbuilder.jump_num_max:
                    # print(
                    #     "Event reached jump_num_max : {}".format(
                    #         self.config.segmentbuilder.jump_num_max
                    #     )
                    # )
                    break

                print("Jump: {jump_num}".format(jump_num=jump_num))
                scattered_segment = event.copy()

                # Physics happens. TODO: This could maybe be wrapped into a different method.

                # Jump Size: Sampled from normal dist.
                mu = self.config.segmentbuilder.jump_size_eV
                sigma = self.config.segmentbuilder.jump_std_eV
                jump_size_eV = np.random.normal(mu, sigma)

                # Delta Pitch Angle: Sampled from normal dist.
                mu, sigma = 0, self.config.segmentbuilder.pitch_angle_costheta_std
                rand_float = np.random.normal(
                    mu, sigma
                )  # Necessary to properly distribute angles on a sphere.
                delta_center_theta = (np.arccos(rand_float) - PI / 2) * RAD_TO_DEG

                # Second, calculate new pitch angle and energy.
                # New Pitch Angle:
                center_theta = center_theta + delta_center_theta

                # Solving an issue caused by pitch angles larger than 90.
                if center_theta > 90:
                    center_theta = 180 - center_theta

                # New energy:
                energy = energy_stop - jump_size_eV

                # New position and direction. Only center_theta is changing right now.
                beta_position, beta_direction = (
                    [rho_pos, phi_pos, zpos],
                    [center_theta, phi_dir],
                )

                # Third, construct a scattered, meaning potentially not-trapped, segment df
                scattered_segment_df = self.eventbuilder.construct_untrapped_segment_df(
                    beta_position, beta_direction, energy, event_num, beta_num
                )

                # Fourth, check to see if the scattered beta is trapped.
                is_trapped = self.eventbuilder.trap_condition(scattered_segment_df)

                jump_num += 1

                # If the event is not trapped or the max number of jumps has been reached,
                # we do not want to write the df to the scattered_segments_list.
                if not is_trapped:
                    print("Event no longer trapped.")
                    break
                # if jump_num > self.config.segmentbuilder.jump_num_max:
                #     print(
                #         "Event reached jump_num_max : {}".format(
                #             self.config.segmentbuilder.jump_num_max
                #         )
                #     )
                #     break

                scattered_segment_df["segment_num"] = jump_num
                scattered_segment_df["segment_length"] = self.segment_length()
                scattered_segment_df = self.fill_in_properties(scattered_segment_df)

                scattered_segments_list.append(
                    scattered_segment_df.iloc[0].values.tolist()
                )

                # reset energy_stop, so that the next segment can be scattered based on this energy.
                energy_stop = scattered_segment_df["energy_stop"]

        scattered_df = pd.DataFrame(
            scattered_segments_list, columns=trapped_event_df.columns
        )

        return scattered_df

    def fill_in_properties(self, incomplete_scattered_segments_df):

        """DOCUMENT LATER"""

        df = incomplete_scattered_segments_df.copy()
        trap_profile = self.config.trap_profile
        main_field = self.config.eventbuilder.main_field
        decay_cell_radius = self.config.eventbuilder.decay_cell_radius

        # Calculate all relevant segment parameters. Order matters here.
        axial_freq = sc.axial_freq(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )

        # TODO: Make this more accurate as per discussion with RJ.
        b_avg = sc.b_avg(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )
        avg_cycl_freq = sc.energy_to_freq(df["energy"], b_avg)

        zmax = sc.max_zpos(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )
        mod_index = sc.mod_index(avg_cycl_freq, zmax)
        segment_radiated_power_te11 = (
            pc.power_calc(
                df["center_x"],
                df["center_y"],
                avg_cycl_freq,
                main_field,
                decay_cell_radius,
            )
            * 2
        )

        segment_radiated_power_tot = sc.power_larmor(main_field, avg_cycl_freq)

        # slope = sc.df_dt(
        #     df["energy"], self.config.eventbuilder.main_field, segment_radiated_power
        # )

        energy_stop = (
            df["energy"] - segment_radiated_power_tot * df["segment_length"] * J_TO_EV
        )

        # Replace negative energies if energy_stop is a float or pandas series
        if isinstance(energy_stop, pd.core.series.Series):
            energy_stop[energy_stop < 0]  = 1e-10
        elif energy_stop < 0:
            energy_stop = 1e-10

        freq_stop = sc.avg_cycl_freq(
            energy_stop, df["center_theta"], df["rho_center"], trap_profile
        )
        slope = (freq_stop - avg_cycl_freq) / df["segment_length"]

        segment_power = segment_radiated_power_te11 / 2

        df["axial_freq"] = axial_freq
        df["avg_cycl_freq"] = avg_cycl_freq
        df["b_avg"] = b_avg
        df["freq_stop"] = freq_stop
        df["energy_stop"] = energy_stop
        df["zmax"] = zmax
        df["mod_index"] = mod_index
        df["slope"] = slope
        df["segment_power"] = segment_power

        return df

    def segment_length(self):
        """TODO: DOCUMENT"""
        mu = self.config.segmentbuilder.mean_track_length
        segment_length = np.random.exponential(mu)

        return segment_length