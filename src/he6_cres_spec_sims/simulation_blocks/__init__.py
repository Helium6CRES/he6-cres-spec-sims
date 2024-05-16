""" simulation_blocks

This module contains all of the simulation "blocks" used by the 
Simulation class (see simulation.py). Each block simulates the action of
a concrete part of the pipeline from beta creation to a .spec file being
 written to disk by the ROACH DAQ. The module also contains the Config
class that reads the JSON config file and holds all of the configurable
parameters as well as the field profile. An instance of  the Config
class linked to a specific JSON config file is passed to each simulation
block.


The general approach is that pandas dataframes, each row describing a
single CRES data object (event, segment,  band, or track), are passed between
the blocks, each block adding complexity to the simulation. This general
structure is broken by the last class (Daq),
which is responsible for creating the .spec (binary) file output. This
.spec file can then be fed into Katydid just as real data would be.

Classes contained in module: 

    * DotDict
    * Config
    * Physics
    * EventBuilder
    * SegmentBuilder
    * BandBuilder
    * TrackBuilder
    * DMTrackBuilder
    * Daq

"""

from .config import *
from .eventBuilder import *
from .segmentBuilder import *
from .bandBuilder import *
from .trackBuilder import *
from .dmTrackBuilder import *
from .DAQ import *