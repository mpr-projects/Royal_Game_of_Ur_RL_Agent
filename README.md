# Royal Game of Ur RL Agent
This repository contains the code that I wrote to train reinforcement-learning agents for *The Royal Game of Ur* and a version of the game *Mancala*. See the *Portfolio* page on my website for more details.

Currently the repository only contains parameters for the on-policy Q-Learning Monte Carlo (*q_mc*) agent of the Royal Game of Ur. To use it
1) clone this repository,
2) install the python packages given in requirements.txt (I created this on Python version 3.11.3) and
3) go to the subfolder *Royal_Game_of_Ur* and run `python play_game.py`. 

When you start the program you should see the list of available RL agents, which in this case is just the *q_mc* model. Then you can enter the names of the two players. If one of the two names is *q_mc* then the RL agent is used. If none of the two names is *q_mc* then a game between two human players will be started.

During the game you see the randomly chosen roll of the die. When it's your turn you'll have to click on the square where you want your stone to land. When it's the RL agent's turn you have to click anywhere to make it move. Depending on your computer it may take a second or two for the agent to make its move. If you're unfamiliar with the rules of the game then please check out its Wikipedia page.

You can also run `python play_match.py n_games` for a match over *n_games* games.

Note, the requirements file only contains the required packages to play the game, not to train it. My code uses JAX and the requirements file downloads the CPU version of JAX. If you want to train models then you should probably install the GPU version of JAX, jax[cuda12], and you may also need additional packages. I've tested the code on two Arch Linux computers and on a MacBook Air M1.
