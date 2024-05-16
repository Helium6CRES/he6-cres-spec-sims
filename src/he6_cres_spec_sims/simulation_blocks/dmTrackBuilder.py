import numpy as np

class DMTrackBuilder:

    """ Downmixes freq_start and freq_stop of simulated tracks to observed frequency band out of DAQ
    """

    def __init__(self, config):

        self.config = config

    def run(self, tracks_df):
        """TODO:Document"""

        print("~~~~~~~~~~~~DMTrackBuilder Block~~~~~~~~~~~~~~\n")
        print(
            "DownMixing the cyclotron frequency with a {} GHz signal".format(
                np.around(self.config.downmixer.mixer_freq * 1e-9, 4)
            )
        )
        mixer_freq = self.config.downmixer.mixer_freq

        downmixed_tracks_df = tracks_df.copy()
        downmixed_tracks_df["freq_start"] = (
            downmixed_tracks_df["freq_start"] - mixer_freq
        )
        downmixed_tracks_df["freq_stop"] = downmixed_tracks_df["freq_stop"] - mixer_freq

        return downmixed_tracks_df
