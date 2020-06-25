from game import Game
from policy_value_net import PolicyValueNet, PolicyValueResidualNet
from players import MCTSPlayer, Human
from trainer import Trainer

# if there is already a model, we use it as the start
init_model = './gomoku_forbidden_rule88'
policy_value_net = PolicyValueResidualNet(model_file=init_model)

trainer = Trainer(policy_value_net)
n_rollout = 1000
player_ai = MCTSPlayer(policy_value_net.policy_value_fn, 5, n_rollout, is_selfplay=0, debug=True)
player_me = Human()
game = Game(player_ai, player_me)

while True:
    winner, game_data = game.do_play()
    player_ai.reset_player()
    # learn while playing with human
    trainer.feed(game_data)
