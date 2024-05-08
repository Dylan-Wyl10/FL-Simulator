"""
Date: 20240421
Author: Dylan Wang
Notes: This script is the server part for feddbc
"""
import torch
from client import *
from .server import Server
import time
import json

from utils import *


class FedDBC(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(FedDBC, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        self.Client = feddbc

    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta \
                                                * (self.server_model_params_list - self.clients_params_list[client]))

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        Averaged_update = torch.zeros(self.server_model_params_list.shape)

        self.train_results = {
            'active_client_list': [],
            'clients_varphi_list': []
        }
        self.client_ls = {}  # initial client dictionary to restore all Client Class.

        for t in range(self.args.comm_rounds):
            # initial a bandwith
            bandwith = torch.rand(1)
            start = time.time()
            # select active clients list
            selected_clients = self._activate_clients_(t)

            # for data collection
            self.train_results['active_client_list'].append(selected_clients)
            varphi_ls = []
            print('============= Communication Round', t + 1, '=============', flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            selected_clients = [1]

            for client in selected_clients:
                print(
                    '#################Communication Round {}, client {}, start to train##################'.format(t + 1,
                                                                                                                  client))
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                if client in self.client_ls.keys() and self.client_ls[client]:
                    _edge_device = self.client_ls[client]
                else:
                    _edge_device = self.Client(device=self.device, id=str(client), model=self.server_model,
                                               model_func=self.model_func, received_vecs=self.comm_vecs,
                                               dataset=dataset, lr=self.lr, bandwith=bandwith, args=self.args)

                # update u
                _edge_device.u = Averaged_update
                self.received_vecs, varphi = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']

                # aa = self.clients_updated_params_list[selected_clients]
                print('the error is', _edge_device.error)

                self.postprocess(client, self.received_vecs)

                varphi_ls.append(varphi)

                # release the salloc
                self.client_ls[client] = _edge_device
                del _edge_device

            self.train_results['clients_varphi_list'].append(varphi_ls)

            # calculate averaged model
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model = torch.mean(self.clients_params_list[selected_clients], dim=0)

            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)

            self._test_(t, selected_clients)
            self._lr_scheduler_()

            # time
            end = time.time()
            self.time[t] = end - start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush=True)

        with open('result_parameter.json', 'w') as json_file:
            json.dump(self.train_results, json_file)

        self._save_results_()
        self._summary_()
