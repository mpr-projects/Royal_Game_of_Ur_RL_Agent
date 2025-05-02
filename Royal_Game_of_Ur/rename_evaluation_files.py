import os
import sys

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'You need to provide the path to an evaluation folder.'
    print('bs:', os.path.basename(sys.argv[1]))
    assert os.path.basename(sys.argv[1]) == 'evaluation', 'You have to select the "evaluation" folder.'

    files = os.listdir(sys.argv[1])

    for file in files:
        file_ = file[:-4]
        components = file_.split('_')

        if not (len(components) == 4 and components[-1] == '0201173'):
            continue

        file_new = '_'.join(components[:-1] + ['20230614_162638-0201173.txt'])

        old_path = os.path.join(sys.argv[1], file)
        new_path = os.path.join(sys.argv[1], file_new)
        print(f'os.rename({old_path}, {new_path})')
        os.rename(old_path, new_path)
