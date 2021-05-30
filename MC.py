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
        for i in range(self.args.numMCTSSims):
            self.rollouts(canonicalBoard, self.args.maxRollouts)

        s = self.game.stringRepresentation(canonicalBoard)
 
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        valids = self.Vs[s]
        cur_best_Q = -float('inf')
        best_actions = []
        for a in range(self.game.getActionSize()):
          if valids[a] and (s,a) in self.Nsa:
            if self.Qsa[(s, a)] > cur_best_Q:
              best_actions = [a]
              cur_best_Q = self.Qsa[(s, a)]

            elif self.Qsa[(s, a)] == cur_best_Q:
              best_actions.append(a)

        bestA = np.random.choice(best_actions)
        probs = [0] * self.game.getActionSize()
        probs[bestA] = 1

        return probs




  def rollouts(self, canonicalBoard, rolloutsCounts):
        """
        This function performs one monte carlo simulation of rollouts
        """

        s = self.game.stringRepresentation(canonicalBoard)

        


        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # if rolloutsCounts == 0:
        #     _, v = self.nnet.predict(canonicalBoard)
        #     return -v


        if s not in self.Ps:
            # leaf node
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

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        while True:
          best_act = np.random.choice(self.game.getActionSize())
          if valids[best_act]:
            a = best_act
            break

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.rollouts(next_s, rolloutsCounts-1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


        

        

        
