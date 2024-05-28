from scipy import interpolate
import pandas as pd
import numpy as np
from time import process_time

from he6_cres_spec_sims.constants import *

class DAQ:
    """  If desired, passes through list of produced downmixed tracks through DAQ, producing fake .spec(k) files
         These can be passed through Katydid, identically to data
    """

    def __init__(self, config):

        self.config = config

        # DAQ parameters derived from the config parameters.
        self.delta_f = config.daq.freq_bw / config.daq.freq_bins
        self.delta_t = 1 / self.delta_f
        self.slice_time = self.delta_t * self.config.daq.roach_avg
        self.pts_per_fft = config.daq.freq_bins * 2
        self.freq_axis = np.linspace( 0, self.config.daq.freq_bw, self.config.daq.freq_bins)

        self.antenna_z = 50  # Ohms

        self.slices_in_spec = int( config.daq.spec_length / self.delta_t / self.config.daq.roach_avg)

        # This block size is used to create chunks of spec file that don't overhwelm the ram.
        self.slice_block = int(50 * 32768 / config.daq.freq_bins)

        # Get date for building out spec file paths.
        self.date = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S")

        # Grab the gain_noise csv. TODO: Document what this needs to look like.
        self.gain_noise = pd.read_csv(self.config.daq.gain_noise_csv_path)

        # Divide the noise_mean_func by the roach_avg.
        # Need to add in U vs I side here.
        self.noise_mean_func = interpolate.interp1d( self.gain_noise.freq, self.gain_noise.noise_mean)
        self.gain_func = interpolate.interp1d( self.gain_noise.freq, self.gain_noise.gain)

        self.noise_array = self.build_noise_floor_array()

    def run(self, downmixed_tracks_df):
        """
        This function is responsible for building out the spec files and calling the below methods.
        """
        self.tracks = downmixed_tracks_df
        self.build_results_dir()
        self.n_spec_files = downmixed_tracks_df.file_in_acq.nunique()
        self.spec_file_paths = self.build_file_paths(self.n_spec_files, self.spec_files_dir, "spec")
        self.build_empty_files(self.spec_file_paths)

        # Define a random phase for each band
        # TODO: This is technically (actually) incorrect, there is an overall random phase that arises from initial particle position
        # in cyclotron motion. Inter-band phases are correlated depending on z0.
        self.phase = self.config.dist_interface.rng.uniform(0, 2*PI, size=len(self.tracks))

        for file_in_acq in range(self.n_spec_files):
            print( f"Building spec file {file_in_acq}. {self.config.daq.spec_length} s, {self.slices_in_spec} slices.")
            build_file_start = process_time()
            for i, start_slice in enumerate( np.arange(0, self.slices_in_spec, self.slice_block)):
                # Iterate by the slice_block until you hit the end of the spec file.
                stop_slice = min(start_slice + self.slice_block, self.slices_in_spec)
                # This is the number of slices before averaging roach+avg slices together.
                num_slices = (stop_slice - start_slice) * self.config.daq.roach_avg

                noise_array = self.noise_array.copy()
                self.config.dist_interface.rng.shuffle(noise_array)
                noise_array = noise_array[: num_slices]

                signal_array = self.build_signal_chunk( file_in_acq, start_slice, stop_slice)

                requant_gain_scaling = 2**self.config.daq.requant_gain

                spec_array = (
                    signal_array
                    * requant_gain_scaling
                    # LNA gain of 67dB
                    #TODO: make this a function of freq
                    *5e6
                    * self.gain_func(self.freq_axis)
                    + noise_array
                )

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
        return None

    def build_signal_chunk(self, file_in_acq, start_slice, stop_slice):

        print(f"file = {file_in_acq}, slices = [{start_slice}:{stop_slice}]")
        ith_slice = np.arange(
            start_slice * self.config.daq.roach_avg,
            stop_slice * self.config.daq.roach_avg,
        )
        num_slices = (stop_slice - start_slice) * self.config.daq.roach_avg

        slice_start = ith_slice * self.delta_t
        slice_stop = (ith_slice + 1) * self.delta_t

        # shape of t: (pts_per_fft, 1, num_slices). axis = 1 will be broadcast into the num_tracks.
        t = np.linspace(slice_start, slice_stop, self.pts_per_fft)
        t = np.expand_dims(t, axis=1)

        # shape of track info arrays: (num_tracks, num_slices)
        time_start = np.repeat(
            np.expand_dims(self.tracks.time_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        time_stop = np.repeat(
            np.expand_dims(self.tracks.time_stop.to_numpy(), axis=1), num_slices, axis=1
        )
        band_power_start = np.repeat(
            np.expand_dims(self.tracks.band_power_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        band_power_stop = np.repeat(
            np.expand_dims(self.tracks.band_power_stop.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        freq_start = np.repeat(
            np.expand_dims(self.tracks.freq_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        slope = np.repeat(
            np.expand_dims(self.tracks.slope.to_numpy(), axis=1), num_slices, axis=1
        )
        file_in_acq_array = np.repeat(
            np.expand_dims(self.tracks.file_in_acq.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )

        # Reshape the random phase assigned to each band.
        band_phase = np.repeat(
            np.expand_dims(self.phase, axis=1),
            num_slices,
            axis=1,
        )

        # shape of slice_start/stop: (1, num_slices)
        slice_start = np.expand_dims(slice_start, axis=0)
        slice_stop = np.expand_dims(slice_stop, axis=0)

        # Note that you will get a division by zero warning if the time_stop and time_start are the same.
        band_powers = band_power_start + (band_power_stop - band_power_start) / (
            time_stop - time_start
        ) * (slice_start - time_start)

        # Caluculate max frequency within the signal in order to impose the LPF.
        freq_curr = freq_start + slope * (slice_stop - time_start)

        # shape of signal_alive_condition: (num_tracks, num_slices).
        # This condition is what imposes the LPF at the top of the freq bandwidth.
        signal_alive_condition = (
            (file_in_acq_array == file_in_acq)
            & (time_start <= slice_start)
            & (time_stop >= slice_stop)
            & (freq_curr <= self.config.daq.freq_bw)
        )

        # print(freq_curr[signal_alive_condition])
        # Setting all "dead" tracks to zero power.
        band_powers[~signal_alive_condition] = 0

        # Calculate voltage of signal.
        voltage = np.sqrt(band_powers * self.antenna_z)

        # The following condition selects only signals that are alive at some point during the time block.
        condition = band_powers.any(axis=1)

        # shape of signal_time_series: (pts_per_fft, num_tracks_alive_in_block, num_slices).
        # The below is the most time consuming operation and the array is very memory intensive.
        # What is happening is that the time domain signals for all slices in this block for each alive signal are made simultaneously
        # and then the signals are summed along axis =1 (track axis).
        # The factor of 2 is needed because the instantaeous frequency is the derivative
        # of the phase. The band_phase is a random phase assigned to each band.
        signal_time_series = voltage[condition, :] * np.sin(
            (
                freq_start[condition, :]
                + slope[condition, :] / 2 * (t - time_start[condition, :])
            )
            * (2 * np.pi * ((t - time_start[condition, :])))
            + band_phase[condition, :]
        )

        # shape of signal_time_series: (pts_per_fft, num_slices).
        signal_time_series = signal_time_series.sum(axis=1)

        # shape of signal_time_series: (pts_per_fft, num_slices). Conduct a 1d FFT along axis = 0 (the time axis).
        fft = np.fft.fft(signal_time_series, axis=0, norm="ortho")[:self.pts_per_fft // 2]

        signal_array = np.real(fft)**2

        #can't average until we've added noise and gotten the powers. 

        return signal_array.T

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
        #aIndex (0 - 4095) is a 12-bit number, does not fit in a byte.
        #Fit in 2 bytes via: index = 2^8 a[0] + a[1]
        #Fourier power then takes up 1 more byte, we know it can't be 0 (given threshold)

        aOnes = frequency_bin % 256
        aTens = (frequency_bin - aOnes) // 256

        return [aTens, aOnes]

    def write_to_speck(self, spec_array, speck_file_path):
        """
        Append to an existing speck file. This is necessary because the raw spec arrays get too large for 1s
        worth of data.
        """
        # Make spec file:
        slices_in_spec, freq_bins_in_spec = spec_array.shape

        # Append empty (zero) header to the spec array.
        header = np.zeros(32)

        # Append empty (zero) header to the spec array.
        footer = np.zeros(3)

        if self.config.daq.threshold_factor is None or self.config.daq.threshold_factor < 0:
                raise ValueError('Invalid DAQ::threshold_factor. Set to non-negative real value!')

        thresholds = np.mean(self.noise_array, axis=0) * self.config.daq.threshold_factor
        data = np.array([])

        # Pass "ab" to append to a binary file
        counter = 0
        with open(speck_file_path, "ab") as speck_file:
            for s in range(slices_in_spec):
                data = np.append(data, header)
                for j in range(freq_bins_in_spec):
                    if int(spec_array[s][j]) > thresholds[j]:
                        data = np.append(data, self.add_high_power_point(j))
                        data = np.append(data, spec_array[s][j])
                        counter +=1
                data = np.append(data, footer)

            data = data.flatten().astype("uint8")
            data.tofile(speck_file)

        #print("counter: "+str(counter))

        return None


    def build_noise_floor_array(self):
        """
        Build a noise floor array with self.slice_block slices.
        Note that this only works for roach avg = 2 at this point.
        """

        self.freq_axis = np.linspace( 0, self.config.daq.freq_bw, self.config.daq.freq_bins)

        delta_f_12 = 2.4e9 / 2**13

        noise_power_scaling = self.delta_f / delta_f_12
        requant_gain_scaling = (2**self.config.daq.requant_gain) / (2**self.config.daq.noise_file_gain)
        noise_scaling = noise_power_scaling * requant_gain_scaling

        # Chisquared noise:
        noise_array = self.config.dist_interface.rng.chisquare(
            df=2, size=(self.slice_block * self.config.daq.roach_avg, self.config.daq.freq_bins)
        )
        noise_array *= self.noise_mean_func(self.freq_axis) / noise_array.mean(axis=0)

        # Scale by noise power.
        noise_array *= noise_scaling
        noise_array = np.around(noise_array).astype("uint8")

        return noise_array

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

    def build_results_dir(self):
        # First make a results_dir with the same name as the config.
        config_name = self.config.config_path.stem
        parent_dir = self.config.config_path.parents[0]

        self.results_dir = parent_dir / config_name
        self.safe_mkdir(self.results_dir)

        self.spec_files_dir = self.results_dir / "spec_files"
        self.safe_mkdir(self.spec_files_dir)

    def build_empty_files(self, files):
        """
        Build empty files to be filled with data
        """
        # Pass "wb" to write a binary file. But here we just build the files.
        for idx, file_path in enumerate(files):
            with open(file_path, "wb") as file:
                pass

        return None

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
