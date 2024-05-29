from scipy import interpolate
import pandas as pd
import numpy as np
from time import process_time

from he6_cres_spec_sims.constants import *

class DAQ:
    """  Optionally passes through list of produced downmixed tracks through DAQ, producing fake .spec(k) files
         These can be passed through Katydid, identically to data
         Converts tracks to time-domain signals, s(t). FFTs give S(f), for each slice
         Data = |S(f) + N(f)|**2, converted to uint8, and written to .spec(k) files
    """

    def __init__(self, config):

        self.config = config

        # DAQ parameters derived from the config parameters.
        self.delta_f = config.daq.freq_bw / config.daq.freq_bins # frequency bin size
        self.delta_t = 1 / self.delta_f # time bin size (before averaging/ tossing)
        self.slice_time = self.delta_t * self.config.daq.roach_avg # time bin size (after averaging/ tossing)
        self.pts_per_fft = config.daq.freq_bins * 2
        self.freq_axis = np.linspace( 0, self.config.daq.freq_bw, self.config.daq.freq_bins)

        self.antenna_z = 50  # Ohms

        self.slices_in_spec = int( config.daq.spec_length / self.delta_t / self.config.daq.roach_avg)

        # This block size is used to create chunks of spec file that don't overwhelm the ram.
        self.slice_block = int(50 * 32768 / config.daq.freq_bins)

        # Get date for building out spec file paths.
        self.date = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S")

        # Grab the gain_noise csv. TODO: Document what this needs to look like.
        self.gain_noise = pd.read_csv(self.config.daq.gain_noise_csv_path)

        # Divide the noise_mean_func by the roach_avg.
        # Need to add in U vs I side here.
        self.noise_mean_func = interpolate.interp1d( self.gain_noise.freq, self.gain_noise.noise_mean)
        self.gain_func = interpolate.interp1d( self.gain_noise.freq, self.gain_noise.gain)

    def run(self, downmixed_tracks_df):
        """
        This function is responsible for building out the spec files and calling the below methods.
        """
        self.tracks = downmixed_tracks_df
        self.create_results_dir()
        self.n_spec_files = downmixed_tracks_df.file_in_acq.nunique()
        self.spec_file_paths = self.build_file_paths(self.n_spec_files, self.spec_files_dir, "spec")
        self.write_empty_files(self.spec_file_paths)

        spec_array = np.zeros(shape=(self.slice_block, self.config.daq.freq_bins))

        for file_in_acq in range(self.n_spec_files):
            print( f"Building spec file {file_in_acq}. {self.config.daq.spec_length} s, {self.slices_in_spec} slices.")
            build_file_start = process_time()
            # Iterate by the slice_block until you hit the end of the spec file.
            for start_slice in np.arange(0, self.slices_in_spec, self.slice_block):
                stop_slice = min(start_slice + self.slice_block, self.slices_in_spec)

                num_slices = stop_slice - start_slice

                spec_array = self.get_noise_array(num_slices)
                spec_array += self.get_signal_array( file_in_acq, start_slice, stop_slice)

                # Computer Fourier power (magnitude_squared)
                spec_array = np.abs(spec_array)**2

                # LNA gain of 67dB
                #TODO: make this a function of freq
                #spec_array *= self.gain_func(self.freq_axis) * self.requant_gain_scaling
                spec_array = self.roach_slice_avg(spec_array)

                # Write chunk to spec file.
                if self.config.daq.spec_suffix == "spec":
                    self.write_to_spec(spec_array, self.spec_file_paths[file_in_acq])
                elif self.config.daq.spec_suffix == "speck":
                    self.write_to_speck(spec_array, self.spec_file_paths[file_in_acq])
                else:
                    raise ValueError('Invalid spec_suffix: spec || speck')

            build_file_stop = process_time()
            print( f"Time to build file {file_in_acq}: {build_file_stop- build_file_start:.3f} s \n")

        print("Done building {} files. ".format(self.config.daq.spec_suffix))

    def get_signal_time_series(self, file_in_acq, start_slice, stop_slice):
        """
        Build a time-domain array of signal (Dimensions = N_FFT Bins x num_slices)
        Later, this will be converted to the frequency domain S(f) via FFT, with the same dimensions
        """
        print(f"file = {file_in_acq}, slices = [{start_slice}:{stop_slice}]")
        slice_start_time = start_slice * self.delta_t 
        slice_stop_time = (stop_slice + 1) * self.delta_t 
        num_slices = stop_slice - start_slice

        t = np.linspace(slice_start_time, slice_stop_time, self.pts_per_fft * num_slices )
        signal_time_series = np.zeros(shape=self.pts_per_fft * num_slices )
        track_phase = np.zeros(shape=self.pts_per_fft * num_slices )

        # TODO: It is on the previous blocks to make sure that the end times are calculated correctly
        # so that the LPF is imposed. Maybe throw a warning?

        # shape of signal_alive_condition: num_tracks
        signal_alive_condition = (
            (self.tracks["file_in_acq"] == file_in_acq)
            & (self.tracks["time_start"] <= slice_stop_time)
            & (self.tracks["time_stop"] >= slice_start_time)
        )

        eligible_tracks = self.tracks[signal_alive_condition]

        # Sum all signals in bandwidth to get total (CRES) time-series, to be FFT'ed
        # The factor of 2 is needed because the instantaneous frequency is the derivative of the phase
        # The band_phase is a random phase assigned to each band.

        for track_index, track in eligible_tracks.iterrows():
            # TODO: Put back in time-dependence of amplitudes. Want to add in frequency-dependence too
            band_power = track["band_power_start"]

            # Slice object - selects the time indices in which the track is active
            track_mask = slice(int(track["time_start"] / self.delta_t), int(track["time_stop"] / self.delta_t), 1)
            voltage = np.sqrt(band_power * self.antenna_z)

            track_phase[track_mask] = 2 * PI * track["freq_start"] * (t[track_mask]-track["time_start"])
            track_phase[track_mask] += 2 * PI * track["slope"] / 2 * (t[track_mask]-track["time_start"])**2

            # Define a random phase for each band
            # TODO: This is technically (actually) incorrect, there is an overall random phase that arises from
            # initial particle position. Different bands in the same event have correlated phases depending on z0
            track_phase[track_mask] += self.config.dist_interface.rng.uniform(0, 2*PI)

            signal_time_series[track_mask] += voltage * np.sin( track_phase[track_mask])
            track_phase[track_mask] = 0

        ### check that I don't need, to e.g. transpose, flip_ud, etc.
        return signal_time_series.reshape((self.pts_per_fft, num_slices))

    def get_signal_array(self, file_in_acq, start_slice, stop_slice):
        """
        Build a frequency-domain array of signal (Dimensions = N_FFT Bins x self.slice_block slices)
        Given signal time-series s(t), convert to frequency domain S(f) via FFT
        Returns frequency-domain (with phase)
        """
        signal_time_series = self.get_signal_time_series(file_in_acq, start_slice, stop_slice)

        # shape of signal_time_series: (pts_per_fft, num_slices). Conduct a 1d FFT along axis = 0 (the time axis).
        Y_fft = np.fft.fft(signal_time_series, axis=0, norm="ortho")[:self.pts_per_fft // 2]
        return Y_fft.T


    def get_noise_array(self, num_slices):
        """
        Build a frequency-domain array of noise (Dimensions = N_FFT Bins x self.slice_block slices)
        For additive white Gaussian noise n, N(f) = FFT(n) is also Gaussian distributed
        For colored noise, multiply by appropriate frequency-dependent factor
        Returns frequency-domain (with phase)
        """

        self.freq_axis = np.linspace( 0, self.config.daq.freq_bw, self.config.daq.freq_bins)
        delta_f_12 = 2.4e9 / 2**13
        # TODO, what is the correct scaling bro?
        noise_scaling = self.delta_f / delta_f_12
        #requant_gain_scaling = (2**self.config.daq.requant_gain) / (2**self.config.daq.noise_file_gain)

        array_size = (num_slices, self.config.daq.freq_bins)

        # Additive white Gaussian noise has FFT which is complex Gaussian
        # Real time-series only imply symmetry between positive/negative frequencies. FFT still complex
        # Imaginary first gets array typing to complex128, avoid recast
        noise_array = 1j * self.config.dist_interface.rng.normal(scale=1./np.sqrt(2), size=array_size)
        noise_array += self.config.dist_interface.rng.normal(scale = 1./np.sqrt(2), size=array_size)

        noise_array *= self.noise_mean_func(self.freq_axis)
        # Scale by noise power.
        noise_array *= noise_scaling

        return noise_array

    def roach_slice_avg(self, signal_array):
        N = int(self.config.daq.roach_avg)
        if self.config.daq.roach_inverted_flag == True:
            result = signal_array[::N]
        else:
            if signal_array.shape[0] % 2 == 0:
                result = signal_array[1::2] + signal_array[::2]
            else:
                result = signal_array[1::2] + signal_array[:-1:2]

        return result

    #################### File Writing Utilities ####################

    def build_file_paths(self, n_files, files_dir, file_label):
        file_paths = []
        for idx in range(n_files):
            file_path = files_dir / "{}_{}_{}.{}".format( self.config.daq.spec_prefix, file_label, idx, self.config.daq.spec_suffix)
            file_paths.append(file_path)
        return file_paths

    def safe_mkdir(self, new_dir):
        # If new_dir doesn't exist, then create it.
        if not new_dir.is_dir():
            new_dir.mkdir()
            print("created directory : ", new_dir)

    def create_results_dir(self):
        # First make a results_dir with the same name as the config.
        config_name = self.config.config_path.stem
        parent_dir = self.config.config_path.parents[0]

        self.results_dir = parent_dir / config_name
        self.safe_mkdir(self.results_dir)

        self.spec_files_dir = self.results_dir / "spec_files"
        self.safe_mkdir(self.spec_files_dir)

    def write_empty_files(self, files):
        """
        Create empty files to be filled with data (to be appended later)
        """
        for file_path in files:
            open(file_path, "wb")

    def write_to_spec(self, spec_array, spec_file_path):
        """
        Append to an existing spec file. This is necessary because the spec arrays get too large for 1s
        worth of data.
        """
        # Make spec file:
        slices_in_spec, freq_bins_in_spec = spec_array.shape

        zero_hdrs = np.zeros((slices_in_spec, 32))

        # Append empty (zero) headers to the spec array.
        spec_array_hdrs = np.hstack((zero_hdrs, spec_array))

        data = spec_array_hdrs.flatten().astype("uint8")

        # Pass "ab" to append to a binary file
        with open(spec_file_path, "ab") as spec_file:
            # Write data to spec_file.
            data.tofile(spec_file)

        return None

    def add_high_power_point(self, frequency_bin):
        # We want to write (bin number, power) to zero-suppressed file
        # aIndex (0 - 4095) is a 12-bit number, does not fit in a byte.
        # Fit in 2 bytes via: index = 2^8 a[0] + a[1]

        aOnes = frequency_bin % 256
        aTens = (frequency_bin - aOnes) // 256

        return [aTens, aOnes]

    def write_to_speck(self, spec_array, speck_file_path):
        """
        Append to an existing speck file. This is necessary because the raw spec arrays get too large for 1s
        worth of data.
        """
        slices_in_spec, freq_bins_in_spec = spec_array.shape

        # Append empty (zero) packet header to data 
        header = np.zeros(32)

        # Append empty (zero) footer. 3 zeros signals end of spectrogram slice
        footer = np.zeros(3)

        if self.config.daq.threshold_factor is None or self.config.daq.threshold_factor < 0:
                raise ValueError('Invalid DAQ::threshold_factor. Set to non-negative real value!')

        #thresholds = np.mean(self.noise_array, axis=0) * self.config.daq.threshold_factor
        # TODO: What should I do to fix this?
        thresholds = np.ones(shape = config.daq.freq_bins ) * self.config.daq.threshold_factor
        data = np.array([])

        # Pass "ab" to append to a binary file
        with open(speck_file_path, "ab") as speck_file:
            for s in range(slices_in_spec):
                data = np.append(data, header)
                for j in range(freq_bins_in_spec):
                    if int(spec_array[s][j]) > thresholds[j]:
                        data = np.append(data, self.add_high_power_point(j))
                        data = np.append(data, spec_array[s][j])
                data = np.append(data, footer)

            data = data.flatten().astype("uint8")
            data.tofile(speck_file)

        return None
