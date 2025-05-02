import os
import sys
import numpy as np


if __name__ == '__main__':
    assert len(sys.argv) == 2, (
        'You must pass the path to a folder containing a'
        ' file called "sample_use.txt".')

    fpath = os.path.join(sys.argv[1], 'sample_use.txt')

    with open(fpath, 'r') as f:
        lines = f.readlines()

    n_sequences = len(lines)
    n_unused = 0  # completely unused sequences
    n_list = list()

    for line in lines:
        line = line.strip()

        if line == '0':
            n_unused += 1
            continue

        line = [int(l) for l in line.split()]
        n_list += line

    n_list = np.array(n_list)
    mean_all = np.mean(n_list)
    std_all = np.std(n_list)

    n_list = n_list[n_list != 0]
    mean_used = np.mean(n_list)
    std_used = np.std(n_list)

    print('Total number of sequences:', n_sequences)
    print(f'Number of completely unused sequences: {n_unused} ({n_unused/n_sequences*100:.2f}%)')
    print(f'Mean number of uses of used moves: {mean_used:.2f} (std={std_used:.2f})')
