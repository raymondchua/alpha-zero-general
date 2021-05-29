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
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

  def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs number of rollouts starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        outcomes = []
        
        for i in range(self.args.numMCsims):
          outcomes.append(self.rollouts(canonicalBoard))

        best_value =  -float('inf')
        best_action = -1

        for value, action in outcomes:

          if value >=best_value:
            best_value = value
            best_action = action

        action_probs = np.zeros((self.game.getActionSize()))
        action_probs[best_action] = 1

        return action_probs





  def rollouts(self, canonicalBoard):
        """
        This function performs one monte carlo simulation of rollouts
        """

        s = self.game.stringRepresentation(canonicalBoard)
        init_start_state = s
        init_action = np.random.choice(self.game.getActionSize())
        isfirstAction = True
        temp_v = 0 


        for i in range(self.args.maxRollouts):

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

          if isfirstAction:
            isfirstAction = False
            init_action = a

          s = self.game.stringRepresentation(next_s)
          temp_v = v
           
        if (s, a) in self.Qsa:
          self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + temp_v) / (self.Nsa[(s, a)] + 1)
          self.Nsa[(s, a)] += 1

        else:
          self.Qsa[(s, a)] = temp_v
          self.Nsa[(s, a)] = 1


        return (-temp_v, init_action)


        

        

        
