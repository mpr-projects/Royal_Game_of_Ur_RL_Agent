"""
This submodule contains files related to the simulation of the problem under
consideration. Files related to the training process can be found in submodule
'training', files that sit at the interface between the simulation and
training can be found in submodule 'simulator'.

All files contain a short description in the first few lines. The first line
indicates if a file is 'general' or 'problem-specific'. For new projects,
typically, only problem-specific files have to be adjusted. General files can
also be identified by a leading underscore '_' in the filename.

All files of this submodule ('simulator') are problem-specific. This submodule
should be a self-contained implementation of the simulation, but should also
provide ways for the methods in submodule 'interface' to establish a connection
between the training process and the simulation.
"""
