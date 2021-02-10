import copy
import time
from typing import Dict, List

import ray
import torch
from torch import distributed
from torch.utils import data

from crowd_search import models
from crowd_search import replay_buffer
from crowd_search import shared_storage


class Dataset(data.Dataset):

    def __init__(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


class Trainer:
    def __init__(
        self,
        models_cfg: Dict,
        device: torch.device,
        process_group: distributed.ProcessGroupGloo,
        num_explorers: int,
    ):
        self.process_group = process_group
        self.num_explorers = num_explorers
        self.epochs = 10
        self.device = device
        # Create models
        gnn_cfg = models_cfg.get("gnn")
        self.gnn = models.GNN(
            gnn_cfg.get("robot-state-dimension"),
            gnn_cfg.get("human-state-dimension"),
            attn_layer_channels=gnn_cfg.get("mlp-layer-channels"),
            num_attention_layers=gnn_cfg.get("gnn-layers"),
        )
        self.gnn.to(device=device)
        self.gnn.train()

        value_net_cfg = models_cfg.get("value-net")
        self.value_estimator = models.ValueNet(
            value_net_cfg.get("channel-depth"), value_net_cfg.get("net-dims")
        )
        self.value_estimator.to(self.device)
        self.value_estimator.train()

        state_net_cfg = models_cfg.get("state-net")
        self.state_estimator = models.StateNet(
            state_net_cfg.get("time-step"), state_net_cfg.get("channel-depth")
        )
        self.state_estimator.to(device)
        self.state_estimator.train()

        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters())
            + list(self.value_estimator.parameters())
            + list(self.state_estimator.parameters()),
            lr=1.0e-3,
        )

        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

        self.dataset = Dataset()
        self.broadcast_models()

    def broadcast_models(self):
        """Send the models out to the agent nodes."""
        models = [self.gnn, self.value_estimator, self.state_estimator]
        distributed.broadcast_object_list(
            models, src=distributed.get_rank(), group=self.process_group
        )

    def _collate_explorations(self):
        """Call out to the explorer nodes associated with this training agent and
        request the training data."""
        output = [None for _ in range(self.num_explorers + 1)]
        distributed.gather_object(
            [], output, dst=distributed.get_rank(), group=self.process_group
        )
        for node_update in output:
            self.dataset.transitions.extend(node_update)
        
        for t in self.dataset.transitions:
            print(type(t))


    def continous_train(self) -> None:

        for epoch in range(self.epochs):

            # Update memory from explorer workers
            self._collate_explorations()

            loader = data.DataLoader(self.dataset, batch_size=20, pin_memory=True)

            for batch in loader:
                print(batch.shape)
                (
                    robot_states,
                    human_states,
                    rewards,
                    next_robot_states,
                    next_human_states,
                ) = batch
                robot_states = torch.stack(robot_states).squeeze(1)
                human_states = torch.stack(human_states).squeeze(1)
                rewards = torch.stack(rewards)
                next_robot_states = torch.stack(next_robot_states).squeeze(1)
                next_human_states = torch.stack(next_human_states).squeeze(1)

                robot_states = robot_states.to(self.device)
                human_states = human_states.to(self.device)
                rewards = rewards.to(self.device)
                next_robot_states = next_robot_states.to(self.device)
                next_human_states = next_human_states.to(self.device)

                robot_state, human_state = self.gnn(robot_states, human_states)
                outputs = self.value_estimator(robot_state)

                gamma_bar = pow(0.9, 0.25 * 1.0)

                next_robot_states, next_human_state = self.gnn(
                    next_robot_states.transpose(1, 2), next_human_states.transpose(1, 2)
                )
                next_value = self.value_estimator(next_robot_states)
                target_values = rewards.unsqueeze(-1) + gamma_bar * next_value
                value_loss = self.l1(outputs, target_values)

                _, next_human_states_est = self.state_estimator(
                    robot_states, human_state, None,
                )
                state_loss = self.l2(
                    next_human_states_est.transpose(1, 2), next_human_states.squeeze(1)
                )
                total_loss = value_loss + state_loss

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"Total loss")



    def process_batch(self, batch):
        (
            robot_states,
            human_states,
            rewards,
            next_robot_states,
            next_human_states,
        ) = batch

        robot_states = torch.stack(robot_states).squeeze(1)
        human_states = torch.stack(human_states).squeeze(1)
        rewards = torch.stack(rewards)
        next_robot_states = torch.stack(next_robot_states).squeeze(1)
        next_human_states = torch.stack(next_human_states).squeeze(1)

        # if use_cuda:
        #    robot_states = robot_states.cuda()
        #    human_states = human_states.cuda()
        #    rewards = rewards.cuda()
        #    next_robot_states = next_robot_states.cuda()
        #    next_human_states = next_human_states.cuda()

        robot_state, human_state = self.gnn(robot_states, human_states)
        outputs = self.value_estimator(robot_state)

        gamma_bar = pow(0.9, 0.25 * 1.0)

        next_robot_states, next_human_state = self.gnn(
            next_robot_states.transpose(1, 2), next_human_states.transpose(1, 2)
        )
        next_value = self.value_estimator(next_robot_states)
        target_values = rewards.unsqueeze(-1) + gamma_bar * next_value
        value_loss = self.l1(outputs, target_values)

        _, next_human_states_est = self.state_estimator(
            robot_states, human_state, None,
        )
        state_loss = self.l2(
            next_human_states_est.transpose(1, 2), next_human_states.squeeze(1)
        )
        total_loss = value_loss + state_loss

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        print(f"loss={total_loss.item():.5f}, rewards={torch.mean(rewards).item():.5f}")

    def _update_weights(self, shared_storage_worker: shared_storage) -> None:
        shared_storage_worker.set_info.remote(
            {
                "gnn_weights": copy.deepcopy(self.gnn.state_dict()),
                "state_estimator_weights": copy.deepcopy(
                    copy.deepcopy(self.state_estimator.state_dict()),
                ),
                "value_estimator_weights": copy.deepcopy(
                    copy.deepcopy(self.value_estimator.state_dict()),
                ),
            }
        )
