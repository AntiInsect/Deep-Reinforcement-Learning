from game import Game
from policy_value_net import PolicyValueNet, PolicyValueResidualNet
from players import MCTSPlayer
from trainer import Trainer
import numpy as np


# init_model = './some_model_if_exist'
init_model = './gomoku_forbidden_rule88'
policy_value_net = PolicyValueResidualNet(model_file=init_model)
n_rollout = 1000
player = MCTSPlayer(policy_value_net.policy_value_fn,5,n_rollout, is_selfplay=1)
game = Game(player,player)

trainer = Trainer(policy_value_net)

loss_file = './loss.log'
entropy_file = './entropy.log'
log_loss = open(loss_file, mode="a", encoding="utf-8")
log_entropy = open(entropy_file, mode="a", encoding="utf-8")

while True:
    print('Start game playing ... ')
    winner, game_data = game.do_play()
    print('Start feeding data to the trainer ... ')
    loss, entropy = trainer.feed(game_data)
    np.savetxt(log_loss, [loss], fmt='%1.4e')
    np.savetxt(log_entropy, [entropy], fmt='%1.4e')
    player.reset_player()


# loss_data = np.loadtxt('./loss.log')
# entropy_data = np.loadtxt('./entropy.log')


