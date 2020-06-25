from game import Game
from policy_value_net import PolicyValueNet, PolicyValueResidualNet
from players import MCTSPlayer
from trainer import Trainer


# init_model = './some_model_if_exist'
init_model = './gomoku_forbidden_rule88'
policy_value_net = PolicyValueResidualNet(model_file=init_model)
n_rollout = 1000
player = MCTSPlayer(policy_value_net.policy_value_fn,5,n_rollout, is_selfplay=1)
game = Game(player,player)

trainer = Trainer(policy_value_net)

while True:
    print('Start game playing ... ')
    winner, game_data = game.do_play()
    print('Start feeding data to the trainer ... ')
    trainer.feed(game_data)
    player.reset_player()
