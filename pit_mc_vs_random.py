import Arena
from MC import MonteCarlo
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = OthelloGame(6)

# all players
rp = RandomPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','MC_checkpoint_13.pth.tar')

args1 = dotdict({'numMCsims': 50, 'cpuct':1.0, 'maxRollouts':5})
mc1 = MonteCarlo(g, n1, args1)
n1p = lambda x: np.argmax(mc1.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, rp, g, display=OthelloGame.display)

print(arena.playGames(20, verbose=True))
