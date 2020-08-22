import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim
from mcts import MCTS
from tqdm import tqdm



class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

    def exceute_episode(self):

        train_examples = []
        current_player = 1
        episode_step = 0
        state = self.game.get_init_board()

        while True:
            episode_step += 1

            canonical_board = self.game.get_canonical_board(state, current_player)

            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.rollout(self.model, canonical_board, to_play=1)

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=0)
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self):
        poliy_loss = []
        value_loss = []

        for _ in tqdm(range(self.args['numIters']), ascii=True, desc='Total Learning Iterations '):
            
            train_examples = []
            for eps in tqdm(range(self.args['numEps']), ascii=True, desc='Collect Training Data '):
                train_examples.extend(self.exceute_episode())
            shuffle(train_examples)

            pi_losses, v_losses = self.train(train_examples)
            poliy_loss.append(pi_losses)
            value_loss.append(v_losses)

            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder=".", filename=filename)
        
        return poliy_loss, value_loss


    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in tqdm(range(self.args['numEpochs']), ascii=True, desc='Training Model '):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous()
                target_pis = target_pis.contiguous()
                target_vs = target_vs.contiguous()


                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            if epoch == self.args['numEpochs'] - 1:
                print()
                print("Policy Loss", np.mean(pi_losses))
                print("Value Loss", np.mean(v_losses))
            # print("Examples:")
            # print(out_pi[0].detach())
            # print(target_pis[0])
        return pi_losses, v_losses

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)
