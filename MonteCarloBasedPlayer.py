import numpy as np
from MonteCarlo import MonteCarlo

class MonteCarloBasedPlayer():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mc = MonteCarlo(game, nnet, args)
        self.K = self.args.mc_topk
        self.qsa = []

    def play(self, canonicalBoard):
      s = self.game.stringRepresentation(canonicalBoard)
      Ps, v = self.nnet.predict(canonicalBoard)
      valids = self.game.getValidMoves(canonicalBoard, 1)
      Ps = Ps * valids  # masking invalid moves
      sum_Ps_s = np.sum(Ps)

      if sum_Ps_s > 0:
          Ps /= sum_Ps_s  # renormalize
      else:
          # if all valid moves were masked make all valid moves equally probable

          # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
          # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
          log.error("All valid moves were masked, doing a workaround.")
          Ps = Ps + valids
          Ps /= np.sum(Ps)

      top_k_actions = np.argpartition(Ps,-self.K)[-self.K:] #to get actions that belongs to top k prob

      for action in top_k_actions: 
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, action)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        values = []

        #do some rollouts
        for rollout in range(self.args.numMCsims):
          value = self.mc.simulate(canonicalBoard)
          values.append(value)

        #average out values
        avg_value = np.mean(values)
        self.qsa.append((avg_value, action))

      best_action = self.qsa.sort(key=lambda a: a[0]).reverse()[1]

      return best_action

    def getActionProb(self, canonicalBoard, temp=1):
      action_probs = np.zeros((self.game.getActionSize()))
      best_action = self.play(canonicalBoard)
      action_probs[best_action] = 1

      return action_probs










