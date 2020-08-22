import torch
from game import Connect2Game
from model import Connect2Model
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":

    args = {
        'batch_size': 64,
        'numRollouts': 100, 
        'numIters': 100,                              # Number of Learning iterations
        'numEps': 100,                                # Number of episodes during per iteration
        'numEpochs': 100,                              # Number of epochs of training per iteration
        'checkpoint_path': 'model.pth'               # location to save latest set of weights
    }

    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Connect2Model(board_size, action_size, device)

    trainer = Trainer(game, model, args)
    policy_loss, value_loss  = trainer.learn()
    policy_loss = np.array(policy_loss)
    value_loss = np.array(value_loss)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(policy_loss.shape[1]), np.mean(policy_loss, axis=0)) 
    plt.title("Policy Loss")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(value_loss.shape[1]), np.mean(value_loss, axis=0)) 
    plt.title("Value Loss")
    plt.show()

