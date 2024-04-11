import torch
from .client import Client
from utils import *
from optimizer import *

class fedavg(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedavg, self).__init__(device, model_func, received_vecs, dataset, lr, args)

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
        # update parameter logic line 14-18
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs