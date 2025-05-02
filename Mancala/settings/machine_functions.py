"""  --- Problem-Specific File ---
To manually test a trained model we need to know which outputs to use and how
it has to be loaded. Each entry of the dict 'machine_fns' is a tuple containing
i) the function to load a trained model, ii) the arguments to that function
(typically includes the path to where the model is saved), and iii) the keyword
arguments to the function.
"""
from simulator.machine_game import machine_fn_random
from interface._simulator_interface import get_machine_fn_rl


machine_fns = dict(
    random_action=(lambda: machine_fn_random, tuple(), dict()),
    q_mc=(
        get_machine_fn_rl,
        ('outputs/outputs_q_mc/',), dict(backend='cpu', is_human_game=True)),
    q_mc_off_policy=(
        get_machine_fn_rl,
        ('outputs/outputs_q_mc_off_policy/',), dict(backend='cpu', is_human_game=True)),
    q_td0=(
        get_machine_fn_rl,
        ('outputs/outputs_q_td0/',), dict(backend='cpu', is_human_game=True)),
    q_tdl=(
        get_machine_fn_rl,
        ('outputs/outputs_q_tdl/',), dict(backend='cpu', is_human_game=True)),
    q_tdl_off_policy=(
        get_machine_fn_rl,
        ('outputs/outputs_q_tdl_off_policy/',), dict(backend='cpu', is_human_game=True)),
)


