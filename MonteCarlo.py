import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)

class MonteCarlo():
  def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
      
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s

#call this rollout
  def simulate(self, canonicalBoard):
        """
        This function performs one monte carlo simulation of rollouts
        """

        s = self.game.stringRepresentation(canonicalBoard)
        init_start_state = s
        temp_v = 0 
        isfirstAction = None

        for i in range(self.args.maxDepth): #maxDepth

          if s not in self.Es:
              self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
          if self.Es[s] != 0:
              # terminal state
              temp_v= -self.Es[s]
              break

          self.Ps[s], v = self.nnet.predict(canonicalBoard)
          valids = self.game.getValidMoves(canonicalBoard, 1)
          self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
          sum_Ps_s = np.sum(self.Ps[s])
          if sum_Ps_s > 0:
              self.Ps[s] /= sum_Ps_s  # renormalize
          else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

          a = np.random.choice(self.game.getActionSize(), p=self.Ps[s])
          next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
          next_s = self.game.getCanonicalForm(next_s, next_player)

          if isfirstAction is None:
            init_action = a

          s = self.game.stringRepresentation(next_s)
          temp_v = v

        # for odd number of steps in a rollout
        if self.args.maxDepth % 2 != 0:
          temp_v = -temp_v

        return temp_v