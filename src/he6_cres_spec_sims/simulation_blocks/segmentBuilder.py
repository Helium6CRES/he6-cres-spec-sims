from .eventBuilder import *
import he6_cres_spec_sims.spec_tools.spec_calc.power_calc as pc

class SegmentBuilder:
    """ Constructs a list of tracks/ segments (interrupted by scatters) making up the trapped event
    """

    def __init__(self, config):

        self.config = config
        self.eventbuilder = EventBuilder(config)

        # distribution of energy losses [eV]
        self.jump_distribution = config.dist_interface.get_distribution(self.config.segmentbuilder.energy_loss)

        # distribution of track durations [s]
        self.segment_length_distribution = config.dist_interface.get_distribution(self.config.segmentbuilder.segment_length)

        # distribution of scattering angles [degrees]
        self.scattering_angle_distribution = config.dist_interface.get_distribution(self.config.segmentbuilder.scattering_angle)

        # distribution of start times [s]
        self.start_time_distribution = config.dist_interface.get_distribution(self.config.segmentbuilder.start_time)

    def run_old(self, trapped_event_df):
        """TODO: DOCUMENT"""
        print("~~~~~~~~~~~~SegmentBuilder Block~~~~~~~~~~~~~~\n")
        # Empty list to be filled with scattered segments.
        scattered_segments_list = []

        for event_index, event in trapped_event_df.iterrows():
            if event_index % 25 == 0:
                print("\nScattering Event :", event_index)

            # Assign segment 0 of event with a segment_length.
            event["segment_length"] = self.segment_length_distribution.generate()

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
            # TODO: Note that it is slightly incorrect to assume the power doesn't change as time passes.
            energy = event["energy"]
            #energy_stop = event["energy_stop"]
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
                    break

                print("Jump: {jump_num}".format(jump_num=jump_num))
                scattered_segment = event.copy()

                # Create new scattered segment then check if its trapped
                scattered_segment_df, center_theta = self.scatter_segment(center_theta, energy_stop, rho_pos, phi_pos, zpos, phi_dir,event_num, beta_num)

                # Fourth, check to see if the scattered beta is trapped.
                is_trapped = self.eventbuilder.trap_condition(scattered_segment_df)

                jump_num += 1

                # If the event is not trapped or the max number of jumps has been reached,
                # we do not want to write the df to the scattered_segments_list.
                if not is_trapped:
                    print("Event no longer trapped.")
                    break

                scattered_segment_df["segment_num"] = jump_num
                scattered_segment_df["segment_length"] = self.segment_length_distribution.generate()
                scattered_segment_df = self.fill_in_properties(scattered_segment_df)

                scattered_segments_list.append(scattered_segment_df.iloc[0].values.tolist())

                # reset energy_stop, so that the next segment can be scattered based on this energy.
                energy_stop = scattered_segment_df["energy_stop"]

        scattered_df = pd.DataFrame(scattered_segments_list, columns=trapped_event_df.columns)

        return scattered_df
    
    def run(self, trapped_event_df):
        """
        Builds scattered tracks for each event. Each track is composed of many segments that describe the shape 
        (freq vs time)
        """
        print("~~~~~~~~~~~~SegmentBuilder Block~~~~~~~~~~~~~~\n")
        # Empty list to be filled with segments.
        tracks_list = []
        segments = []
        #create segments for every event
        for event_index, event in trapped_event_df.iterrows():
            if event_index % 25 == 0:
                print("\nBuilding Event :", event_index)

            # Assign segment 0 of event with a segment_length.
            scatter_time = self.segment_length_distribution.generate()

            # Fill the event with computationally intensive properties.
            event = self.fill_in_properties(event)

            event["time_start"] = self.start_time_distribution.generate()
            event["freq_start"] = event["avg_cycl_freq"]

            # Begin with trapped beta (segment 0 of event).
            tracks = [event]
            is_trapped = True
            jump_num = 0
            max_freq = self.config.physics.freq_acceptance_high

            #TODO maybe this should be a different varibale in the config... or maybe just renamed
            trap_on_time = self.config.daq.spec_length if self.config.daq.spec_length else float('inf')
            end_time = (trap_on_time, scatter_time) [scatter_time>trap_on_time]
            print(end_time)
            

            while is_trapped and jump_num<=self.config.segmentbuilder.jump_num_max:
                print(f"Jump number: {jump_num}")
                track_segments = []
                t, freq, field = tracks[-1]["time_start"], tracks[-1]["freq_start"], tracks[-1]["b_avg"]
                segment_radiated_power_tot = sc.power_larmor(field, freq)
                while (t < end_time) and (freq < max_freq):
                    segment = self.create_segment(t, freq, segment_radiated_power_tot, end_time, max_freq, event_index,
                                                   jump_num, field)
                    t, freq = segment.end_time, segment.end_freq
                    track_segments.append(segment)

                tracks[-1]["freq_stop"] = freq
                tracks[-1]["time_stop"] = t
                tracks[-1]["energy_stop"] = sc.freq_to_energy(freq, tracks[-1]["b_avg"])
                #TODO update all segment numbers to track numbers
                tracks[-1]["segment_num"] = jump_num

                segments.append(track_segments)
                tracks_list.append(tracks[-1].values.tolist())

                # break out of loop if this track reached end of trap on time
                if t >= trap_on_time: break 

                new_track = self.scatter(tracks[-1])

                if self.eventbuilder.trap_condition(new_track) == True:
                    #TODO there is almost certainly a better way to pass/grab the new track info
                    new_track = next(self.fill_in_properties(new_track).iterrows())[1]
                    new_track["time_start"] = t
                    new_track["freq_start"] = freq
                    tracks.append(new_track)
                    jump_num += 1
                else: 
                    is_trapped=False
        print(tracks_list[-1])
        # TODO there may be a more elegant way to update the columns... but this works for now       
        columns = np.append(trapped_event_df.columns.to_numpy(), ["time_start","freq_start","time_stop"])
        scattered_df = pd.DataFrame(tracks_list, columns=columns)

        return scattered_df, segments
    
    def scatter(self, event):
        """Creates Scattered track from initial event conditions.
        """

        center_x, center_y = event["center_x"], event["center_y"]
        rho_pos = event["initial_rho_pos"]
        phi_pos = event["initial_phi_pos"]
        zpos = 0
        center_theta = event["center_theta"]
        phi_dir = event["initial_phi_dir"]
        energy_stop = event["energy_stop"]
        event_num = event["event_num"]
        beta_num = event["beta_num"]

        # Jump Size
        jump_size_eV = self.jump_distribution.generate()

        # Delta Pitch Angle: Sampled from normal dist.
        scattering_angle = self.scattering_angle_distribution.generate()

        # Original beta direction vector in cartesian coordinates
        theta_dir = center_theta # tmp see TODO below
        v_vec_old = np.array([np.sin(theta_dir/RAD_TO_DEG) *np.cos(phi_dir/RAD_TO_DEG),
                    np.sin(theta_dir/RAD_TO_DEG)*np.sin(phi_dir/RAD_TO_DEG), np.cos(theta_dir/RAD_TO_DEG)])

        # This always produces a vector lying on cone with angle = scattering_angle with respect to initial velocity vector
        tmp_theta = theta_dir + scattering_angle
        tmp_v_vec = np.array([np.sin(tmp_theta/RAD_TO_DEG) *np.cos(phi_dir/RAD_TO_DEG),
                    np.sin(tmp_theta/RAD_TO_DEG)*np.sin(phi_dir/RAD_TO_DEG), np.cos(tmp_theta/RAD_TO_DEG)])

        # Using Rodrigues' Rotation Formula (rotate tmp_velocity_vec around velocity_vec_old)
        # Depending on dPhi, get random vector along that cone
        dPhi = 2*PI*self.config.dist_interface.rng.uniform()
        vNew = tmp_v_vec * np.cos(dPhi) + np.cross(v_vec_old,tmp_v_vec) * np.sin(dPhi) + v_vec_old * np.dot(v_vec_old,tmp_v_vec) * (1-np.cos(dPhi))
        vNew /= np.sqrt(np.dot(vNew, vNew)) # Probably unnecessary, don't want this to drift too much from floating point errors

        # Second, calculate new pitch angle and energy.
        theta_new = np.arccos( vNew[2] ) * RAD_TO_DEG
        phi_dir = np.arctan2(vNew[1],vNew[0]) * RAD_TO_DEG

        # TODO: We should 1) randomly choose z's to scatter at (PDF proportional to v_z^-1, which means saving z-motion, inverse transform sampling)
        # Then 2, convert local scattered pitch angle to the new center_theta
        # Since we scatter only at z=0, don't bother propagating scattering change in instantaneous theta_dir to center_theta
        center_theta = theta_new

        # Ensure that pitch angle is defined to be smaller than 90.
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
        return self.eventbuilder.construct_untrapped_segment_df(beta_position, beta_direction, energy, event_num, beta_num)


    def scatter_segment(self, center_theta, energy_stop, rho_pos, phi_pos, zpos, phi_dir,event_num, beta_num):
        """Creates Scattered segment from initial event conditions.
        TODO find more elegant solution to dealing with center_theta, I don't like that its being passed around so much.
        """
        # Jump Size
        jump_size_eV = self.jump_distribution.generate()

        # Delta Pitch Angle: Sampled from normal dist.
        scattering_angle = self.scattering_angle_distribution.generate()

        # Original beta direction vector in cartesian coordinates
        theta_dir = center_theta # tmp see TODO below
        v_vec_old = np.array([np.sin(theta_dir/RAD_TO_DEG) *np.cos(phi_dir/RAD_TO_DEG),
                    np.sin(theta_dir/RAD_TO_DEG)*np.sin(phi_dir/RAD_TO_DEG), np.cos(theta_dir/RAD_TO_DEG)])

        # This always produces a vector lying on cone with angle = scattering_angle with respect to initial velocity vector
        tmp_theta = theta_dir + scattering_angle
        tmp_v_vec = np.array([np.sin(tmp_theta/RAD_TO_DEG) *np.cos(phi_dir/RAD_TO_DEG),
                    np.sin(tmp_theta/RAD_TO_DEG)*np.sin(phi_dir/RAD_TO_DEG), np.cos(tmp_theta/RAD_TO_DEG)])

        # Using Rodrigues' Rotation Formula (rotate tmp_velocity_vec around velocity_vec_old)
        # Depending on dPhi, get random vector along that cone
        dPhi = 2*PI*self.config.dist_interface.rng.uniform()
        vNew = tmp_v_vec * np.cos(dPhi) + np.cross(v_vec_old,tmp_v_vec) * np.sin(dPhi) + v_vec_old * np.dot(v_vec_old,tmp_v_vec) * (1-np.cos(dPhi))
        vNew /= np.sqrt(np.dot(vNew, vNew)) # Probably unnecessary, don't want this to drift too much from floating point errors

        # Second, calculate new pitch angle and energy.
        theta_new = np.arccos( vNew[2] ) * RAD_TO_DEG
        phi_dir = np.arctan2(vNew[1],vNew[0]) * RAD_TO_DEG

        # TODO: We should 1) randomly choose z's to scatter at (PDF proportional to v_z^-1, which means saving z-motion, inverse transform sampling)
        # Then 2, convert local scattered pitch angle to the new center_theta
        # Since we scatter only at z=0, don't bother propagating scattering change in instantaneous theta_dir to center_theta
        center_theta = theta_new

        # Ensure that pitch angle is defined to be smaller than 90.
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
        return self.eventbuilder.construct_untrapped_segment_df(beta_position, beta_direction, energy, event_num, beta_num), center_theta

    def fill_in_properties(self, incomplete_scattered_segments_df):
        """ Assigns calculated properties (e.g. axial frequency, power, slope, etc.)
            to beta with given (E, theta, rho) in the magnetic field profile
        """

        df = incomplete_scattered_segments_df.copy()
        trap_profile = self.config.trap_profile
        main_field = self.config.eventbuilder.main_field
        decay_cell_radius = self.config.eventbuilder.decay_cell_radius

        # Calculate all relevant segment parameters. Order matters here.
        axial_freq = sc.axial_freq( df["energy"], df["center_theta"], df["rho_center"], trap_profile)

        # TODO: Make this more accurate as per discussion with RJ.
        b_avg = sc.b_avg( df["energy"], df["center_theta"], df["rho_center"], trap_profile)
        avg_cycl_freq = sc.energy_to_freq(df["energy"], b_avg)
        zmax = sc.max_zpos( df["energy"], df["center_theta"], df["rho_center"], trap_profile)
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

        # slope = sc.df_dt( df["energy"], self.config.eventbuilder.main_field, segment_radiated_power)

        energy_stop = ( df["energy"] - segment_radiated_power_tot * df["segment_length"] * J_TO_EV)

        # Replace negative energies if energy_stop is a float or pandas series
        if isinstance(energy_stop, pd.core.series.Series):
            energy_stop[energy_stop < 0]  = 1e-10
        elif energy_stop < 0:
            energy_stop = 1e-10

        freq_stop = sc.avg_cycl_freq( energy_stop, df["center_theta"], df["rho_center"], trap_profile)
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
    
    def create_segment(self, time, freq, power, max_time, max_freq, event, track, b_avg):
        '''
        This function creates a track object for a given time and frequency. Currently only check that we are within
        segment length.
        TODO add more track options
        '''

        # set different ranges of frequencies where different things can happen, the largest range is normal linear
        # segments, but there can be cutoff regions, field slewing regions, etc where shape and slope changes
        
        # TODO currently code is creating events outside of physics frequency range... this buffer is a bandaid on
        # this problem which I believe is from the physics part of the code, bandaiding so I can continue debugging
        # this code...
        linear_range = [self.config.physics.freq_acceptance_low-0.1e9, self.config.physics.freq_acceptance_high]
        print(freq)
        
        if linear_range[0] <= freq < linear_range[1]:
            segment = LinearSegment(time, freq, power, event, track, max_time, max_freq, b_avg)

        return segment

class Segment:
    '''
    A class for different types of track segments, the most common being a linear segment. All tracks have a start
    and end frequency and time and power, but  have different shapes and integrals.
    '''

    def __init__(self, start_time, start_freq, start_power, event, track):
        self.start_freq = start_freq
        self.start_time = start_time
        self.start_power = start_power
        self.event = event
        self.track = track

    
class LinearSegment(Segment):
     def __init__(self, start_time, start_freq, start_power, event, track, max_time, max_freq, field):
        super().__init__(start_time, start_freq, start_power, event, track)
        self.track_type = "Linear"

        start_energy = sc.freq_to_energy(start_freq, field)
        self.slope = sc.df_dt( start_energy, field, start_power)

        if self.slope*max_time < max_freq:
            self.end_freq = self.slope*max_time + start_freq
            self.end_time = max_time
        else:
            self.end_freq = max_freq
            self.end_time = (max_freq-start_freq)/self.slope

        print(f"Slope: {self.slope} \n t0: {start_time}, tf: {self.end_time} \n f0: {start_freq}, ff: {self.end_freq}")