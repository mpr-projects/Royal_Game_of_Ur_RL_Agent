# this file will be symlinked in the respective game-folders, running a
# symlink'ed file uses the path of the target file, not of the link file
import os
import sys
sys.path.remove(sys.path[0])
sys.path.append(os.path.dirname(__file__))


from training._utils import plot_evaluation_results
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_evaluation_results(sys.argv[1])
        sys.exit()

    fig, ax = plt.subplots()

    for fpath in sys.argv[1:]:
        plot_evaluation_results(fpath, ax=ax)

    plt.xlabel('Epochs')
    plt.ylabel('% Wins of Player 1')

    yb, yt = plt.ylim()
    yb, yt = min(yb-0.01, 0.49), min(yt+0.01, 1.01)
    plt.ylim(yb, yt)

    plt.axhline(1.0, linestyle='--', color='black', lw=0.8)
    plt.axhline(0.5, linestyle='--', color='black', lw=0.8)

    plt.legend()
    plt.show()
