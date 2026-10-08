from .eventBuilder import *
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.spec_calc.power_calc as pc
import he6_cres_spec_sims.spec_tools.spec_calc.exb as exb
from .Band import *

class TrackBuilder:
    """ Constructs a list of tracks (interrupted by scatters) making up the trapped event
    """

    def __init__(self, config):

        self.config = config
        self.eventbuilder = EventBuilder(config)

        # distribution of energy losses [eV]
        self.jump_distribution = config.dist_interface.get_distribution(self.config.trackbuilder.energy_loss)

        # distribution of track durations [s]
        self.track_length_distribution = config.dist_interface.get_distribution(self.config.trackbuilder.track_length)

        # distribution of scattering angles [degrees]
        self.scattering_angle_distribution = config.dist_interface.get_distribution(self.config.trackbuilder.scattering_angle)

        # distribution of start times [s]
        self.start_time_distribution = config.dist_interface.get_distribution(self.config.trackbuilder.start_time)

        self.ExB = exb.ExB(self.config.trackbuilder.voltage_off_time_ms/1000., self.config.trackbuilder.voltage_on_time_ms/1000., self.config.trackbuilder.voltage_fractional_offset)

        self.verbosity = self.config.trackbuilder.verbose
        print(self.config.trackbuilder.verbose)

    def run(self, trapped_event_df):
        """
        Builds scattered tracks for each event.
        """
        print("~~~~~~~~~~~~TrackBuilder Block~~~~~~~~~~~~~~\n")
        # Empty list to be filled with tracks.
        bands = []
        tracks_list = []

        #RF BW for cutting off bands appropriately to remove aliasing
        min_freq = self.config.downmixer.mixer_freq
        max_freq = min_freq + self.config.daq.freq_bw

        #create tracks for every event
        for event_index, event in trapped_event_df.iterrows():
            if event_index % 2500 == 0:
                print("\nBuilding Event :", event_index)

            # Assign track 0 of event with a birth time.
            event["start_time"] = self.start_time_distribution.generate()

            tracks = [event]

            #list of band objects (to be added to bands list)
            event_main_bands = []

            #Randomly distribute events among N acquisitions as an integer between [0,N-1] (to be assigned to all tracks in event)
            #https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
            acq_num = self.config.dist_interface.rng.integers(0, self.config.daq.n_acquisitions, dtype=int, endpoint=False)
            #print("Acquisition Number: ", acq_num)

            #Time at which the trap next empties or the current file acquisition ends
            #Corresponds to the max end time of the event, unless it scatters out or leaves bandwidth
            time_next_exb_sweep = min(self.ExB.next_empty(event["start_time"]), self.config.daq.acq_length)

            # number of max tracks per event
            nMaxTracks = self.config.trackbuilder.jump_num_max + 1

            for track_num in range(nMaxTracks):
                #tracks[track_num] is final "known" trapped track in the event, currently (before next are computed)
                tracks[track_num]["track_num"] = track_num
                tracks[track_num]["acq_num"] = acq_num
                tracks[track_num]["trap_acq_num"] = self.ExB.trap_cycle_index(tracks[track_num]["start_time"])
                tracks[track_num]["start_time_in_trap_acq"] = self.ExB.time_in_trap_acq(tracks[track_num]["start_time"])

                tracks[track_num]["track_length"] = self.track_length_distribution.generate()
                tracks[track_num]["end_time"] =  tracks[track_num]["start_time"] + tracks[track_num]["track_length"]
                tracks[track_num]["end_time_in_trap_acq"] = self.ExB.time_in_trap_acq(tracks[track_num]["end_time"])

                #Determine whether scatter time or exb happens sooner, adjust end_time and track_length as necessary
                #Recompute track_length for tracks that are cleared out by ExB as now track_length != scatter_time
                final_track = (tracks[track_num]["end_time"] >= time_next_exb_sweep)
                tracks[track_num]["end_time"] =  np.clip(tracks[track_num]["end_time"], None, time_next_exb_sweep)
                tracks[track_num]["track_length"] = tracks[track_num]["end_time"] - tracks[track_num]["start_time"]

                #Estimate the slope from the Larmor power
                track_radiated_power_tot = sc.power_larmor(tracks[track_num]["b_avg"], tracks[track_num]["start_freq"])
                start_energy = sc.freq_to_energy(tracks[track_num]["start_freq"], tracks[track_num]["b_avg"])
                slope = sc.df_dt( start_energy, tracks[track_num]["b_avg"], track_radiated_power_tot)

                band = LinearBand(acq_num, tracks[track_num]["start_time"], tracks[track_num]["start_freq"],  tracks[track_num]["end_time"], min_freq, max_freq, event_index, track_num, 0, slope)
                event_main_bands.append(band)

                #Modify track end properties based on integration
                tracks[track_num]["end_freq"] = band.end_freq
                tracks[track_num]["end_time"] = band.end_time
                tracks[track_num]["end_energy"] = sc.freq_to_energy(band.end_freq, tracks[track_num]["b_avg"])

                #More efficient to add new tracks to list instead of directly to DataFrame
                tracks_list.append(tracks[track_num].values.tolist())

                if final_track:
                    break
                else:
                    new_track = self.scatter(tracks[track_num]) # if we are not cleared by the ExB, compute next scatter
                    #if we roll low, scatter is elastic, assumed to instantly eject beta
                    if self.config.dist_interface.rng.uniform(0,1) < self.config.trackbuilder.frac_elastic:
                        break
                    if not self.eventbuilder.trap_condition(new_track): #confirm that next scatter is still magnetically trapped
                        break

                    #fill in the "expensive" properties of the next track (b_avg, axial_freq, etc.) if it is trapped
                    filled_new_track = self.eventbuilder.fill_in_start_properties(new_track)

                    #Assign start_time of next time as end_time of previous track
                    #Note we do not need to do this for frequencies, which are computed by b_avg, given pitch angle. Start energy handled in scatter(). Do not override!
                    filled_new_track["start_time"] = band.end_time

                    filled_new_track = pd.Series(filled_new_track.squeeze()) # converts new_track from a pandas DataFrame to a pandas Series (table vs. single row)
                    tracks.append(filled_new_track)

            bands.append(event_main_bands)

        tracks_df = pd.DataFrame(tracks_list, columns=trapped_event_df.columns)

        return tracks_df, bands

    def scatter(self, event):
        """Creates Scattered track from initial event conditions.
        """

        center_x, center_y = event["start_guiding_center_x"], event["start_guiding_center_y"]
        rho_pos = event["start_rho"]
        phi_pos = event["start_phi"]
        zpos = 0
        start_theta = event["start_theta"]
        phi_dir = event["start_phi_dir"]
        energy_stop = event["end_energy"]
        beta_num = event["beta_num"]
        event_num = event["event_num"]

        # Jump Size
        jump_size_eV = self.jump_distribution.generate()

        # Delta Pitch Angle: Sampled from normal dist.
        scattering_angle = self.scattering_angle_distribution.generate()

        # Original beta direction vector in cartesian coordinates
        theta_dir = start_theta # tmp see TODO below
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
        start_theta = theta_new

        # Ensure that pitch angle is defined to be smaller than 90.
        if start_theta > 90:
            start_theta = 180 - start_theta

        # New energy: (do not lose so much energy it becomes negative)
        energy = np.clip(energy_stop - jump_size_eV, 1e-10, None)

        # New position and direction. Only center_theta is changing right now.
        beta_position, beta_direction = (
            [rho_pos, phi_pos, zpos],
            [start_theta, phi_dir],
        )

        # Third, construct a scattered, meaning potentially not-trapped, segment df
        return self.eventbuilder.construct_untrapped_beta_df(beta_position, beta_direction, energy, beta_num, event_num)

