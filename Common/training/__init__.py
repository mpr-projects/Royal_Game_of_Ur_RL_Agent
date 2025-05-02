"""
This submodule contains files related to the actual training process, without
a direct connection to the simulation. Files related to the simulation can be
found in submodule 'simulation', files that sit at the interface between the
simulation and training can be found in submodule 'simulator'.

All files contain a short description in the first few lines. The first line
indicates if a file is 'general' or 'problem-specific'. For new projects,
typically, only problem-specific files have to be adjusted. General files can
also be identified by a leading underscore '_' in the filename.

This submodule must contain the following files and methods:
    - file utils.py with functions
        + def get_dummy_input(settings)
        + def check_settings(settings)

    - file loss_functions.py with function
        + get_loss_fn(settings)

    - file target_functions.py with function
        + get_target_fn(settings)
"""
