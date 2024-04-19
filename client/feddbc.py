import torch
from .client import Client
from utils import *
from optimizer import *
from random import random

class fedavg(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedavg, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        self.trigger_low = torch.tensor(0)
        self.trigger_high = torch.tensor(1)

    def train(self):
        # local training
        self.model.train()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                predictions = self.model(inputs)
                loss = self.loss(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

        last_state_params_list = get_mdl_params(self.model)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
        # update parameter logic line 14-18
        """
        Start with modifying the algorithm in algorithm line 10-18
        """
        # this is delta^{t}_{i}, line 10
        delta = last_state_params_list - self.received_vecs['Params_list']
        # line 11: give bandwith, officially this is grabed randomly range 0-1
        self.bandwith = torch.rand(1)
        # line 12:
        rho = self.bandwith * self.trigger_low + (1 - self.bandwith) * self.trigger_high
        # Calculate varphi = \varphi_i^r = \|-\Delta^t_i+\widehat{\Delta}_{i}^{t}-\mathbf{e}^t_i\|^2 -\rho_i^t\|\Delta^t_i\|^2, (eqution3)
        varphi = torch.norm((-delta + delta_head - error), p=2) - rho * torch.norm(delta, p=2)


        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
        self.comm_vecs['local_update_list'] = delta
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs