import os
import sys
import training._utils


if __name__ == '__main__':
    args = training._utils.parse_arguments()
    settings = training._utils.load_settings_file(args)

    if settings['training'].get('use_replay_buffer', False) is True:
        f = 'train_with_rb.py'

    else:
        f = 'train_without_rb.py'

    args = ' '.join(sys.argv[1:])
    path = os.path.join(os.path.dirname(__file__), f)

    print(f'python {path} {args}')
    os.system(f'python {path} {args}')
