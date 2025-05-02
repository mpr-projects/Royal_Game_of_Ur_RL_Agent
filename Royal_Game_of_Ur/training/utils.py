import numpy as np


def get_dummy_input(settings):
    return np.random.randn(1, 3, 8, 5)


def check_settings(settings):
    # Todo: may want to add method-specific checks
    method = settings['training']['method']
    sso = settings['training']['skip_single_option']

    if method in ['q_mc', 'q_td0', 'q_tdl', 'q_td0_off_policy',
                  'q_tdl_off_policy']:
        assert sso is False, 'Q-learning needs to learn from single options.'

    if method in ['q_tdn', 'ac_tdn']:
        assert 'td_n' in settings['training'], \
            'tdn requires settings to contain key \'td_n\' specifying n.'

    irbs = settings['replay_buffer']['initial_replay_buffer_size']
    rbso = settings['replay_buffer']['replay_buffer_sampling_offset']
    bs = settings['training']['batch_size']

    # temp
    assert True or irbs-rbso >= bs, (
        'Sum of batch_size and replay_buffer_sampling_offset must be at least'
        ' as large as the initial_replay_buffer_size.')

    if method.find('td0') != -1 or method.find('tdl') != -1:
        assert 'gamma' in settings['training'], \
            'For td-learning you need to set a gamma value.'

    if method.find('tdl') != -1:
        assert 'alpha' in settings['training'], \
            'For td-lambda you need to set an alpha value.'


