import torch
from .client import Client
from utils import *
from optimizer import *
from random import random


class feddbc(Client):
    def __init__(self, device, id, model, model_func, received_vecs, dataset, lr, bandwith, args):
        super(feddbc, self).__init__(device, id, model, model_func, received_vecs, dataset, lr, bandwith, args)
        self.trigger_low = torch.tensor(0)
        self.trigger_upper = torch.tensor(1)
        # self.delta_head = torch.tensor(0)
        self.delta_head = torch.zeros(get_mdl_params(self.model).shape)
        self.error = torch.zeros(get_mdl_params(self.model).shape)
        self.u = torch.zeros(get_mdl_params(self.model).shape)
        self.isupdate = True
        # self.bandwith = bandwith

    def train(self):
        if self.isupdate:
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
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
            # update parameter logic line 14-18
            """
            Start with modifying the algorithm in algorithm line 10-18
            """
            # this is delta^{t}_{i}, line 10
            delta = last_state_params_list - self.received_vecs['Params_list']
            # line 11: give bandwith, officially this is grabed randomly range 0-1, now grab from upper level
            # self.bandwith = torch.rand(1)
            # line 12:
            rho = self.trigger_low + (self.trigger_upper - self.trigger_low) * torch.pow(torch.e, -self.bandwith * torch.norm(delta))
            # line 13: Calculate varphi = \varphi_i^r = \|-\Delta^t_i+\widehat{\Delta}_{i}^{t}-\mathbf{e}^t_i\|^2 -\rho_i^t\|\Delta^t_i\|^2, (eqution3)
            varphi = torch.norm((-delta + self.delta_head - self.error), p=2) - rho * torch.norm(delta, p=2)
            print('$$$$$$$rho is {}, varphi1 is {}, varphi2 is {}'.format(rho, torch.norm((-delta + self.delta_head - self.error), p=2) , rho * torch.norm(delta, p=2)))
            if varphi >= 0:
                """20240428 YW:use top-k compressor funcction"""
                """20240429 YW:replace the removed component with zero"""
                print('$$$$$$$$$$$$$$varphi value is {}, compress and use topk'.format(varphi))
                self.delta_head = self.topKcompress(self.error - delta, self.bandwith)
                self.comm_vecs['local_update_list'] = self.delta_head
                self.comm_vecs['local_model_param_list'] = last_state_params_list
                # line15: transmit u to global server
            else:
                """question for @Yixing: I am still wondering how u^{t}_{i} update.... ---  from Yilin 20240421"""
                print('$$$$$$$$$$$$$not compress and use u')
                self.delta_head = self.u
            # update local error, line 18
            self.error = self.error + delta - self.delta_head
        ########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#########
        else:
            self.error = self.error
        # line 21, 22
        self.trigger_low = self.trigger_low / (1 + self.trigger_low * torch.norm(self.error, p=2))
        self.trigger_upper = (self.trigger_upper * torch.norm(self.error, p=2) + self.trigger_upper) / (
                    1 + torch.norm(self.error, p=2))

        # self.comm_vecs['local_update_list'] = delta
        # self.comm_vecs['local_model_param_list'] = last_state_params_list  # not sure if this is needed.
        return self.comm_vecs, varphi

    @staticmethod
    def topKcompress(x, ratio):
        res = torch.zeros(x.shape)  # create a zero tensor to save the results
        tensor, indice = torch.topk(x, k=int(ratio*x.shape[0]), sorted=False)
        for i in range(tensor.shape[0]):
            res[indice[i]] = tensor[i]
        return res
