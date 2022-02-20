
import logging
import os
import sys
import copy
sys.path.append('.')
sys.path.append('..')
from brownie import accounts, ComputingMarket ,network

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from scripts.deploy import train_setting
from scripts.deploy import deploy_contract

from client_module.client import MockClient
import client_module.utils as u
from client_module.log import logger as l,init_logging

class ClusterSimulator(object):

    def __init__(self):
        init_logging(logging.DEBUG)

        dataset_dir = '/Users/bytedance/blockchian/federate-learning-computing-market/federate_learning_with_blockchain/client_module/dataset/MNIST'
        client_train_dataset, test_x_tensor, test_y_tensor = u.split_mnist_dataset(
            True, dataset_dir, len(accounts))
        self.accounts = accounts

        self.test_dl = DataLoader(TensorDataset(
            test_x_tensor, test_y_tensor), batch_size=100, shuffle=False)
        l.info(f"load test_dl finish...")
        # reset gas limit
        # network.gas_limit(1000000000000)
        self.contract = deploy_contract()
        l.info(f"build contract,contract's info:{self.contract}")

        self.client_setting = copy.deepcopy(train_setting)

        self.clients = []
        for id in range(len(self.accounts)):
            custome_setting = copy.deepcopy(self.client_setting)
            custome_setting["model_name"] = "mnist_2nn"
            train_x_tesnor,train_y_tensor = client_train_dataset[id]
            custome_setting["dataset"] = TensorDataset(train_x_tesnor,train_y_tensor)
            custome_setting["id"] = id
            custome_setting["account"] = self.accounts[id]
            custome_setting["contract"] = self.contract
            new_client = MockClient(custome_setting)
            del custome_setting
            self.clients.append(new_client)
            l.info(f"build client {id}")
        l.info(f"build {len(self.accounts)} clients finish")
        # upload the init model params to let all client start at same params
        # self.clients[0].aggregate_and_upload()
        l.info(f"upload the init model params")
        self.round = 0

    def update_one_version_sequential(self,n):
        for client in self.clients:
            client.flesh_global_model()
            l.info(f"client {client.id} flesh, model abstract is {client.trainer.get_model_abstract()}")
            client.local_train()
            client.upload_model_update()
        self.round += 1
        pass

    def update_one_version_parallel(self,n):
        self.round += 1
        pass

    def evalute_global_model(self):
        accuracy = self.clients[0].evalute(self.test_dl)
        l.info(f"round {self.round},accuracy {accuracy}")
        return accuracy