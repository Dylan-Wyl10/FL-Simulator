"""
Date: 20240421
Author: Dylan Wang
Notes: This script is the server part for feddbc
"""
import torch
from client import *
from .server import Server

class FedDBC(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(FedAvg, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        self.Client =  feddbc