import numpy as np

class TrackBuilder:
    """ Assigns track truth parameters (e.g. start/ end times, frequencies) to created tracks
    """

    def __init__(self, config):

        self.config = config

    def run(self, bands_df):

        print("~~~~~~~~~~~~TrackBuilder Block~~~~~~~~~~~~~~\n")
        run_length = self.config.trackbuilder.run_length
        # events_to_simulate = self.config.physics.events_to_simulate
        events_simulated = int(bands_df["event_num"].max() + 1)
        print("events simulated: ", events_simulated)
        # TODO: Event timing is not currently physical. 
        # RJ: working on making this more physical, currently assuming beta monitor rate is equivalent to decay cell rate
        # Add time/freq start/stop.
        tracks_df = bands_df.copy()
        tracks_df["time_start"] = np.nan
        tracks_df["time_stop"] = np.nan
        tracks_df["file_in_acq"] = np.nan

        tracks_df["freq_start"] = bands_df["avg_cycl_freq"]
        tracks_df["freq_stop"] = bands_df["slope"] * bands_df["segment_length"] + bands_df["avg_cycl_freq"]

        # Assigning event times by uniformly placing them in time window
        window = self.config.daq.n_files*self.config.daq.spec_length
        trapped_event_start_times = self.config.dist_interface.rng.uniform(0, window, events_simulated)

        # iterate through the segment zeros and fill in start times.
        for index, row in bands_df[bands_df["segment_num"] == 0.0].iterrows():
            event_num = int(tracks_df["event_num"][index])
            file_num = int(trapped_event_start_times[event_num] // self.config.daq.spec_length)
            # updating some pandas behavior that will be deprecated in pandas 3.0
            # tracks_df["time_start"][index] = trapped_event_start_times[event_num] - self.config.daq.spec_length*file_num
            tracks_df.loc[index, "time_start"] = trapped_event_start_times[event_num] - self.config.daq.spec_length*file_num
            # tracks_df["file_in_acq"][index] = file_num
            tracks_df.loc[index, "file_in_acq"] = file_num

        for event in range(0, events_simulated):
            # find max segment_num for each event
            segment_num_max = int( bands_df[bands_df["event_num"] == event]["segment_num"].max())

            for segment in range(1, segment_num_max + 1):
                fill_condition = (tracks_df["event_num"] == float(event)) & ( tracks_df["segment_num"] == segment)
                previous_time_condition = (
                    (tracks_df["event_num"] == event)
                    & (tracks_df["segment_num"] == segment - 1)
                    & (tracks_df["band_num"] == 0.0)
                )
                
                if not np.any(previous_time_condition):
                    breakpoint()
                previous_segment_time_start = tracks_df[previous_time_condition][ "time_start" ].iloc[0]
                previous_segment_length = tracks_df[previous_time_condition][ "segment_length" ].iloc[0]

                file_num = tracks_df[(tracks_df["event_num"] == event) & (tracks_df["segment_num"] == 0.0)]["file_in_acq"].item()

                for index, row in tracks_df[fill_condition].iterrows():
                    # updating some pandas behavior that will be deprecated in pandas 3.0
                    # tracks_df["time_start"][index] = previous_segment_time_start + previous_segment_length
                    tracks_df.loc[index, "time_start"] = previous_segment_time_start + previous_segment_length
                    #inherit file_in_acq from parent event
                    # tracks_df["file_in_acq"][index] = file_num
                    tracks_df.loc[index, "file_in_acq"] = file_num

        tracks_df["time_stop"] = tracks_df["time_start"] + tracks_df["segment_length"]

        return tracks_df
