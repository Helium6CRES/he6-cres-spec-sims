from PyQt6 import QtWidgets, uic, QtCore, QtGui
import qdarktheme
from pathlib import Path
import yaml
import sys
import ast
import subprocess as sp

dr = Path(__file__).parent

distribution_index = {
        "aseev": 0,
        "beta_decay": 1,
        "cauchy": 2, "lorentz": 2,
        "dirac": 3, "fixed": 3, "delta": 3,
        "exponential": 4,
        "normal": 5, "gaussian": 5,
        "rudd": 6,
        "uniform": 7,
        "uniform_annulus": 8,
    }

def load_yaml(path):
    with open(path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        uic.loadUi(dr / 'main_window.ui', self)
        self.init_settings()
        self.init_physics()
        self.init_event_builder()
        self.init_track_builder()
        self.init_dmtrack_builder()
        self.init_daq()
        self.init_buttons()
        
        self.default_yaml = dr / "config.yaml"
        if self.default_yaml.is_file():
            self.load_yaml_to_ui(path = self.default_yaml)

# ~~~~~~~~~~~~~~~~~~~~ init functions ~~~~~~~~~~~~~~~~~~~~
    def init_settings(self):
        self.simDAQCheckBox.toggled.connect(self.sim_daq_toggled)
        self.useHardwareEntropyCheckBox.toggled.connect(self.hardware_entropy_toggled)

    def init_physics(self):
        self.freqAcceptanceLowLineEdit.setValidator(
            QtGui.QDoubleValidator(0.0, 22e9, 6,
            notation = QtGui.QDoubleValidator.Notation.ScientificNotation,)
        )
        self.freqAcceptanceHighLineEdit.setValidator(
            QtGui.QDoubleValidator(0.0, 22e9, 6,
            notation = QtGui.QDoubleValidator.Notation.ScientificNotation,)
        )
        self.energyWidget.distributionCombo.setCurrentIndex(distribution_index["beta_decay"])
        self.rhoWidget.distributionCombo.setCurrentIndex(distribution_index["uniform_annulus"])
        self.zWidget.distributionCombo.setCurrentIndex(distribution_index["uniform"])
    
    def init_event_builder(self):
        self.decayCellRadiusLineEdit.setValidator(
            QtGui.QDoubleValidator(0.0, 1.0, 6,
            notation = QtGui.QDoubleValidator.Notation.ScientificNotation,)
        )

    def init_track_builder(self):
        self.startTimeWidget.distributionCombo.setCurrentIndex(distribution_index["uniform"])
        self.trackLengthWidget.distributionCombo.setCurrentIndex(distribution_index["uniform"])
        self.energyLossWidget.distributionCombo.setCurrentIndex(distribution_index["aseev"])
        self.scatteringAngleWidget.distributionCombo.setCurrentIndex(distribution_index["fixed"])

    def init_dmtrack_builder(self):
        self.mixerFreqLineEdit.setValidator(
            QtGui.QDoubleValidator(0.0, 22e9, 6,
            notation = QtGui.QDoubleValidator.Notation.ScientificNotation,)
        )

    def init_daq(self):
        self.freqBandwidthLineEdit.setValidator(
            QtGui.QDoubleValidator(0.0, 10e9, 6,
            notation = QtGui.QDoubleValidator.Notation.ScientificNotation,)
        )

    def init_buttons(self):
        self.loadButton.clicked.connect(self.load_yaml_to_ui)
        self.saveButton.clicked.connect(self.save_ui_to_yaml)
        self.runButton.clicked.connect(self.run_spec_sims)

# ~~~~~~~~~~~~~~~~~~~~ Get settings from UI ~~~~~~~~~~~~~~~~~~~~
    def build_dict(self):
        config_dict = {
            "Settings": self.build_settings_dict(),
            "Physics": self.build_physics_dict(),
            "EventBuilder": self.build_eventbuilder_dict(),
            "TrackBuilder": self.build_trackbuilder_dict(),
            "SideBandBuilder": self.build_sideband_dict(),
            "DMTrackBuilder": self.build_dmtrack_dict(),
            "DAQ": self.build_daq_dict(),
            }
        print(config_dict)
        return config_dict

    def build_distribution_dict(self, distribution_widget):
        # see he6_cres_spec_sims/spec_tools/distributions/distribution_interface.py for allowed names
        distribution_names = ["aseev", "beta_decay", "cauchy", "dirac", "exponential", "normal", "rudd", "uniform", "uniform_annulus"]

        name = distribution_names[distribution_widget.distributionCombo.currentIndex()]
        distribution_dict = {'distribution': name}
 
        if name == "aseev":
            distribution_dict["isotope"] = "H2" # hardcoded because I think this is the only thing that's actually used
            distribution_dict["x_c"] = distribution_widget.x_cDoubleSpinBox.value()
            distribution_dict["mu_1"] = distribution_widget.mu_1DoubleSpinBox.value()
            distribution_dict["sigma_1"] = distribution_widget.sigma_1DoubleSpinBox.value()
            distribution_dict["mu_2"] = distribution_widget.mu_2DoubleSpinBox.value()
            distribution_dict["sigma_2"] = distribution_widget.sigma_2DoubleSpinBox.value()
        elif name == "beta_decay":
            pass
        elif name == "cauchy" or name=="lorentz":
            distribution_dict["mean"] = distribution_widget.meanDoubleSpinBox.value()
            distribution_dict["gamma"] = distribution_widget.gammaDoubleSpinBox.value()
        elif name == "dirac" or name == "fixed":
            distribution_dict["value"] = distribution_widget.valueDoubleSpinBox.value()
        elif name == "exponential":
            distribution_dict["tau"] = distribution_widget.tauDoubleSpinBox.value()
        elif name == "normal" or name=="gaussian":
            distribution_dict["mean"] = distribution_widget.muDoubleSpinBox.value()
            distribution_dict["sigma"] = distribution_widget.sigmaDoubleSpinBox.value()
        elif name == "rudd":
            distribution_dict["alpha"] = distribution_widget.alphaDoubleSpinBox.value()
        elif name == "uniform":
            distribution_dict["low"] = distribution_widget.minDoubleSpinBox.value()
            distribution_dict["high"] = distribution_widget.maxDoubleSpinBox.value()
        elif name == "uniform_annulus":
            distribution_dict["rho_min"] = distribution_widget.rhoMinDoubleSpinBox.value()
            distribution_dict["rho_max"] = distribution_widget.rhoMaxDoubleSpinBox.value()
        else:
            raise ValueError(f"Loading parameters for distribution {name} from UI not yet implemented")
 
        return distribution_dict

    def literal_eval_float(self, line_edit):
        text = line_edit.text()
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (int, float)):
            float_values = [float(parsed)]
        else:
            # interprets "[1,2,3]" as list and "1,2,3" as tuple
            float_values = [float(x) for x in parsed]
        return float_values

    def build_settings_dict(self):
        seed = self.randSeedLineEdit.text() if self.useHardwareEntropyCheckBox.isChecked() else None
        sim_daq = self.simDAQCheckBox.isChecked()
        settings_dict = {
            "rand_seed": seed,
            "sim_daq": sim_daq,
            }
        return settings_dict

    def build_physics_dict(self):
        events_to_simulate = self.eventsToSimulateSpinBox.value()
        betas_to_simulate = self.betasToSimulateSpinBox.value()
        freq_acceptance_low = float(self.freqAcceptanceLowLineEdit.text())
        freq_acceptance_high = float(self.freqAcceptanceHighLineEdit.text())
        energy = self.build_distribution_dict(self.energyWidget)
        rho = self.build_distribution_dict(self.rhoWidget)
        z = self.build_distribution_dict(self.zWidget)
        min_theta = self.minThetaDoubleSpinBox.value()
        max_theta = self.maxThetaDoubleSpinBox.value()

        physics_dict = {
            "events_to_simulate": events_to_simulate,
            "betas_to_simulate": betas_to_simulate,
            "freq_acceptance_low": freq_acceptance_low,
            "freq_acceptance_high": freq_acceptance_high,
            "energy": energy,
            "rho": rho,
            "z": z,
            "min_theta": min_theta,
            "max_theta": max_theta,
            }
        return physics_dict

    def build_eventbuilder_dict(self):
        main_field = self.literal_eval_float(self.mainFieldLineEdit)
        trap_current = self.literal_eval_float(self.trapCurrentLineEdit)
        decay_cell_radius = float(self.decayCellRadiusLineEdit.text())
        # only write first value to yaml, pass the full array of fields/traps when running
        self.main_field_array = main_field
        self.trap_current_array = trap_current
        eventbuilder_dict = {
            "main_field": main_field[0], 
            "trap_current": trap_current[0],
            "decay_cell_radius": decay_cell_radius,
            }
        return eventbuilder_dict

    def build_trackbuilder_dict(self):
        start_time = self.build_distribution_dict(self.startTimeWidget)
        track_length = self.build_distribution_dict(self.trackLengthWidget)
        energy_loss = self.build_distribution_dict(self.energyLossWidget)
        scattering_angle = self.build_distribution_dict(self.scatteringAngleWidget)
        frac_elastic = self.fracElasticDoubleSpinBox.value()
        jump_num_max = self.maxNumberOfJumpsSpinBox.value()
        voltage_on_time = self.voltageOnTimeDoubleSpinBox.value()
        voltage_off_time = self.voltageOffTimeDoubleSpinBox.value()
        voltage_fractional_offset = self.voltageFractionalOffsetDoubleSpinBox.value()

        trackbuilder_dict = {
            "start_time": start_time,
            "track_length": track_length,
            "energy_loss": energy_loss,
            "scattering_angle": scattering_angle,
            "frac_elastic": frac_elastic,
            "jump_num_max": jump_num_max,
            "voltage_on_time_ms": voltage_on_time,
            "voltage_off_time_ms": voltage_off_time,
            "voltage_fractional_offset": voltage_fractional_offset,
            }
        return trackbuilder_dict
    
    def build_sideband_dict(self):
        sideband_num = self.sidebandNumberSpinBox.value()
        frac_total_track_power_cut = self.fracTotalTrackPowerCutDoubleSpinBox.value()
        harmonic_sidebands = self.harmonicSidebandsCheckBox.isChecked()
        magnetic_modulation = self.magneticModulationCheckBox.isChecked()

        sideband_dict = {
            "sideband_num": sideband_num,
            "frac_total_track_power_cut": frac_total_track_power_cut,
            "harmonic_sidebands": harmonic_sidebands,
            "magnetic_modulation": magnetic_modulation,
            }
        return sideband_dict

    def build_dmtrack_dict(self):
        mixer_freq = float(self.mixerFreqLineEdit.text())
        dmtrack_dict = {"mixer_freq": mixer_freq}
        return dmtrack_dict

    def build_daq_dict(self):
        acq_length = self.acqLengthDoubleSpinBox.value()
        n_acquisitions = self.nAcquisitionsSpinBox.value()
        freq_bw = float(self.freqBandwidthLineEdit.text())
        freq_bins = int(self.freqBinsComboBox.currentText())
        n_channels = int(self.nChannelsComboBox.currentText())
        roach_avg = self.roachAvgSpinBox.value()
        roach_inverted_flag = self.roachInvertedCheckBox.isChecked()
        build_labels = self.buildLabelsCheckBox.isChecked()
        noise_path1 = self.noisePath1Widget.pathLineEdit.text()
        noise_path2 = self.noisePath2Widget.pathLineEdit.text()
        if not Path(noise_path1).is_file():
            noise_path1 = ""
        if not Path(noise_path2).is_file():
            noise_path2 = ""
        noise_paths = [noise_path1, noise_path2]
        noise_temperature = self.noiseTemperatureDoubleSpinBox.value()
        spec_prefix = self.specPrefixLineEdit.text()
        spec_suffix = self.specSuffixLineEdit.text()
        threshold_factor = self.thresholdFactorDoubleSpinBox.value()
        trap_off_bin = self.trapOffBinDoubleSpinBox.value()
        rigol_voltage_mv = self.rigolVoltageDoubleSpinBox.value()

        daq_dict = {
            "acq_length": acq_length,
            "n_acquisitions": n_acquisitions,
            "freq_bw": freq_bw,
            "freq_bins": freq_bins,
            "n_channels": n_channels,
            "roach_avg": roach_avg,
            "roach_inverted_flag": roach_inverted_flag,
            "build_labels": build_labels,
            "noise_paths": noise_paths,
            "noise_temperature": noise_temperature,
            "spec_prefix": spec_prefix,
            "spec_suffix": spec_suffix,
            "threshold_factor": threshold_factor,
            "trap_off_bin": trap_off_bin,
            "rigol_voltage_mv": rigol_voltage_mv,
            }
        return daq_dict

# ~~~~~~~~~~~~~~~~~~~~ Write settings to UI ~~~~~~~~~~~~~~~~~~~~
    def display_dict(self, config_dict):
        self.display_settings_dict(config_dict),
        self.display_physics_dict(config_dict),
        self.display_eventbuilder_dict(config_dict),
        self.display_trackbuilder_dict(config_dict),
        self.display_sideband_dict(config_dict),
        self.display_dmtrack_dict(config_dict),
        self.display_daq_dict(config_dict),

    def display_distribution_dict(self, distribution_dict, distribution_widget):
        # see he6_cres_spec_sims/spec_tools/distributions/distribution_interface.py for allowed names
        distribution_names = ["aseev", "beta_decay", "cauchy", "dirac", "exponential", "normal", "rudd", "uniform", "uniform_annulus"]

        name = distribution_dict["distribution"]
        name_idx = distribution_index[name]
 
        if name == "aseev":
            # distribution_dict["isotope"] = "H2"
            try:
                distribution_widget.x_cDoubleSpinBox.setValue(distribution_dict["x_c"])
                distribution_widget.mu_1DoubleSpinBox.setValue(distribution_dict["mu_1"])
                distribution_widget.sigma_1DoubleSpinBox.setValue(distribution_dict["sigma_1"])
                distribution_widget.mu_2DoubleSpinBox.setValue(distribution_dict["mu_2"])
                distribution_widget.sigma_2DoubleSpinBox.setValue(distribution_dict["sigma_2"])
            except:
                pass
        elif name == "beta_decay":
            pass
        elif name == "cauchy" or name=="lorentz":
            distribution_widget.meanDoubleSpinBox.setValue(distribution_dict['mean'])
            distribution_widget.gammaDoubleSpinBox.setValue(distribution_dict['gamma'])
        elif name == "dirac" or name == "fixed":
            distribution_widget.valueDoubleSpinBox.setValue(distribution_dict['value'])
        elif name == "exponential":
            distribution_widget.tauDoubleSpinBox.setValue(distribution_dict['tau'])
        elif name == "normal" or name=="gaussian":
            distribution_widget.muDoubleSpinBox.setValue(distribution_dict['mu'])
            distribution_widget.sigmaDoubleSpinBox.setValue(distribution_dict['sigma'])
        elif name == "rudd":
            distribution_widget.alphaDoubleSpinBox.setValue(distribution_dict['alpha'])
        elif name == "uniform":
            distribution_widget.minDoubleSpinBox.setValue(distribution_dict['low'])
            distribution_widget.maxDoubleSpinBox.setValue(distribution_dict['high'])
        elif name == "uniform_annulus":
            distribution_widget.rhoMinDoubleSpinBox.setValue(distribution_dict['rho_min'])
            distribution_widget.rhoMaxDoubleSpinBox.setValue(distribution_dict['rho_max'])
        else:
            raise ValueError(f"Displaying parameters for distribution {name} from UI not yet implemented")
 
        return distribution_dict

    def display_settings_dict(self, config_dict):
        settings = config_dict["Settings"]
        seed = settings["rand_seed"]
        sim_daq = settings["sim_daq"]
        use_hardware_entropy = (seed is None) or (not seed)
        self.useHardwareEntropyCheckBox.setChecked(use_hardware_entropy)
        if not use_hardware_entropy:
            self.randSeedLineEdit.setText(str(seed))
        self.randSeedLineEdit.setEnabled(not use_hardware_entropy)
        self.simDAQCheckBox.setChecked(sim_daq)
        self.daq_tab.setEnabled(sim_daq)

    def display_physics_dict(self, config_dict):
        physics = config_dict["Physics"]
        events_to_simulate = physics["events_to_simulate"]
        betas_to_simulate = physics["betas_to_simulate"]
        freq_acceptance_low = physics["freq_acceptance_low"]
        freq_acceptance_high = physics["freq_acceptance_high"]
        energy = physics["energy"]
        rho = physics["rho"]
        z = physics["z"]
        min_theta = physics["min_theta"]
        max_theta = physics["max_theta"]

        self.eventsToSimulateSpinBox.setValue(events_to_simulate)
        self.betasToSimulateSpinBox.setValue(betas_to_simulate)
        self.freqAcceptanceLowLineEdit.setText(str(freq_acceptance_low))
        self.freqAcceptanceHighLineEdit.setText(str(freq_acceptance_high))
        self.display_distribution_dict(energy, self.energyWidget)
        self.display_distribution_dict(rho, self.rhoWidget)
        self.display_distribution_dict(z, self.zWidget)
        self.minThetaDoubleSpinBox.setValue(min_theta)
        self.maxThetaDoubleSpinBox.setValue(max_theta)

    def display_eventbuilder_dict(self, config_dict):
        eb = config_dict["EventBuilder"]
        main_field = eb["main_field"]
        trap_current = eb["trap_current"]
        decay_cell_radius = eb["decay_cell_radius"]

        self.mainFieldLineEdit.setText(str(main_field))
        self.trapCurrentLineEdit.setText(str(trap_current))
        self.decayCellRadiusLineEdit.setText(str(decay_cell_radius))

    def display_trackbuilder_dict(self, config_dict):
        tb = config_dict["TrackBuilder"]
        start_time = tb["start_time"]
        track_length = tb["track_length"]
        energy_loss = tb["energy_loss"]
        scattering_angle = tb["scattering_angle"]
        frac_elastic = tb["frac_elastic"]
        jump_num_max = tb["jump_num_max"]
        voltage_on_time = tb["voltage_on_time_ms"]
        voltage_off_time = tb["voltage_off_time_ms"]
        voltage_fractional_offset = tb["voltage_fractional_offset"]

        self.display_distribution_dict(start_time, self.startTimeWidget)
        self.display_distribution_dict(track_length, self.trackLengthWidget)
        self.display_distribution_dict(energy_loss, self.energyLossWidget)
        self.display_distribution_dict(scattering_angle, self.scatteringAngleWidget)
        self.fracElasticDoubleSpinBox.setValue(frac_elastic)
        self.maxNumberOfJumpsSpinBox.setValue(jump_num_max)
        self.voltageOnTimeDoubleSpinBox.setValue(voltage_on_time)
        self.voltageOffTimeDoubleSpinBox.setValue(voltage_off_time)
        self.voltageFractionalOffsetDoubleSpinBox.setValue(voltage_fractional_offset)
    
    def display_sideband_dict(self, config_dict):
        sb = config_dict["SideBandBuilder"]
        sideband_num = sb["sideband_num"]
        frac_total_track_power_cut = sb["frac_total_track_power_cut"]
        harmonic_sidebands = sb["harmonic_sidebands"]
        magnetic_modulation = sb["magnetic_modulation"]

        self.sidebandNumberSpinBox.setValue(sideband_num)
        self.fracTotalTrackPowerCutDoubleSpinBox.setValue(frac_total_track_power_cut)
        self.harmonicSidebandsCheckBox.setChecked(harmonic_sidebands)
        self.magneticModulationCheckBox.setChecked(magnetic_modulation)

    def display_dmtrack_dict(self, config_dict):
        dmtracks = config_dict["DMTrackBuilder"]
        mixer_freq = dmtracks["mixer_freq"]
        self.mixerFreqLineEdit.setText(str(mixer_freq))

    def display_daq_dict(self, config_dict):
        daq = config_dict["DAQ"]
        acq_length = daq["acq_length"]
        n_acquisitions = daq["n_acquisitions"]
        freq_bw = daq["freq_bw"]
        freq_bins = daq["freq_bins"]
        n_channels = daq["n_channels"]
        roach_avg = daq["roach_avg"]
        roach_inverted_flag = daq["roach_inverted_flag"]
        build_labels = daq["build_labels"]
        noise_path1, noise_path2 = daq["noise_paths"]
        noise_temperature = daq["noise_temperature"]
        spec_prefix = daq["spec_prefix"]
        spec_suffix = daq["spec_suffix"]
        threshold_factor = daq["threshold_factor"]
        trap_off_bin = daq["trap_off_bin"]
        rigol_voltage_mv = daq["rigol_voltage_mv"]

        self.acqLengthDoubleSpinBox.setValue(acq_length)
        self.nAcquisitionsSpinBox.setValue(n_acquisitions)
        self.freqBandwidthLineEdit.setText(str(freq_bw))
        self.freqBinsComboBox.setCurrentText(str(freq_bins))
        self.nChannelsComboBox.setCurrentText(str(n_channels))
        self.roachAvgSpinBox.setValue(roach_avg)
        self.roachInvertedCheckBox.setChecked(roach_inverted_flag)
        self.buildLabelsCheckBox.setChecked(build_labels)
        self.noisePath1Widget.pathLineEdit.setText(noise_path1)
        self.noisePath2Widget.pathLineEdit.setText(noise_path2)
        self.noiseTemperatureDoubleSpinBox.setValue(noise_temperature)
        self.specPrefixLineEdit.setText(spec_prefix)
        self.specSuffixLineEdit.setText(spec_suffix)
        self.thresholdFactorDoubleSpinBox.setValue(threshold_factor)
        self.trapOffBinDoubleSpinBox.setValue(trap_off_bin)
        self.rigolVoltageDoubleSpinBox.setValue(rigol_voltage_mv)

# ~~~~~~~~~~~~~~~~~~~~ PyQt connections ~~~~~~~~~~~~~~~~~~~~
    def sim_daq_toggled(self, checked):
        self.daq_tab.setEnabled(checked)

    def hardware_entropy_toggled(self, checked):
        self.randSeedLineEdit.setEnabled(not checked)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select file",
            "",
            "All Files (*.*)"
        )
        return path
    
    def load_yaml_to_ui(self, *, path = None):
        if path is None:
            path = self.browse_file()
        if Path(path).is_file():
            loaded_dict = load_yaml(path)
            self.display_dict(loaded_dict)

    def save_ui_to_yaml(self, *, path = "config.yaml"):
        config_dict = self.build_dict()
        path = Path(path).resolve()
        # TODO: file browser window to select where to save
        print(f"Saving current config to {path}")
        with open(path, "w") as f:
            yaml.dump(config_dict, f)
        return path

    def run_spec_sims(self):
        self.runButton.setEnabled(False)
        config_path = self.save_ui_to_yaml(path = self.default_yaml)
        # TODO: make this universal
        run_script = Path("/home/luciano/src/spec_sims/run_spec_sims.py")
        sp.run([
            "uv",
            "run",
            str(run_script),
            str(config_path),
            "--project",
            str(run_script.parent / ".venv/bin/python"),
            ])
        self.runButton.setEnabled(True)
        

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        qdarktheme.setup_theme()
        main = MainWindow()
        main.show()
        sys.exit(app.exec())
    except KeyboardInterrupt as e:
        sys.exit(0)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
