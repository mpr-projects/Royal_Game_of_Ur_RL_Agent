"""
This submodule contains files related to the connection between the simulation
and the training of the process under consideration. Files related to the
training process can be found in submodule 'training', files related to the
simulation of the problem under consideration can be found in submodule 
'simulator'. 

All files contain a short description in the first few lines. The first line
indicates if a file is 'general' or 'problem-specific'. For new projects,
typically, only problem-specific files have to be adjusted. General files can
also be identified by a leading underscore '_' in the filename.

Specifically, the problem-specific files in this folder have to include two
files. i) The file 'sample_creation.py' which provides the following two
functions:

  - def create_sample(params, state, apply, settings, seed=None)
  - def sample_from_rb(rb, settings)

The first function creates a new sample and returns it. The sample will be
stored in the replay buffer. The second file samples a (batch of) sample(s)
from the replay buffer.
ii) The file 'machine_agent.py' which provides the function:

  - def create(params, state, policy, apply, print_probs=False)

which return a machine agent (called from the evaluation script).
Todo: sort this out, should I standardize evaluation to some extent? Should _simulator_interface.py be renamed and lose its _?
"""
