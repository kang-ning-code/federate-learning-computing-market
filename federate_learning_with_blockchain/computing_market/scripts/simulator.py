
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

simulator_setting = copy.deepcopy(train_setting)
simulator_setting['dataset_dir'] = r'D:\cs\Code\VisualStudioCode\federate-learning-computing-market\federate_learning_with_blockchain\client_module\data\MNIST'
# simualtor_setting['dataset_dir'] = '/Users/bytedance/blockchian/federate-learning-computing-market/federate_learning_with_blockchain/client_module/dataset/MNIST'
simulator_setting['log_dir'] = r'D:\cs\Code\VisualStudioCode\federate-learning-computing-market\federate_learning_with_blockchain\client_module\reports\logs'
# simulator_setting['log_dir'] = '/Users/bytedance/blockchian/federate-learning-computing-market/federate_learning_with_blockchain/client_module/reports/logs'
    
class ClusterSimulator(object):

    def __init__(self):
        log_dir = simulator_setting['log_dir']
        init_logging(log_dir,log_level = logging.DEBUG)
        dataset_dir = simulator_setting['dataset_dir']
        client_train_dataset, test_x_tensor, test_y_tensor = u.split_mnist_dataset(
            True, dataset_dir, len(accounts))
        self.accounts = accounts

        self.test_dl = DataLoader(TensorDataset(
            test_x_tensor, test_y_tensor), batch_size=100, shuffle=False)
        l.info(f"load test_dl finish...")
        self.contract = deploy_contract()
        l.info(f"build contract,contract's info:{self.contract}")

        self.client_setting = copy.deepcopy(train_setting)

        self.clients = []
        for id in range(len(self.accounts)):
            custome_setting = copy.deepcopy(self.client_setting)
            custome_setting["model_name"] = "mnist_2nn"
            custome_setting["ipfs_api"] = "/ip4/127.0.0.1/tcp/5001"
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

    def fresh_clients_model(self):
        l.info('[fresh_clients_model]')
        for client in self.clients:
            client.fresh_global_model()

    def clients_local_train(self):
        l.debug('[clients_local_train]')
        for client in self.clients:
            l.debug(f"client {client.id} before train,model hash is {client.cur_model_hash()}")
            client.local_train()
            client.upload_model_update()
            l.debug(f"client {client.id} after train,model hash is {client.cur_model_hash()}")

    def clients_local_evalute(self):
        l.info('[clients_local_evalute]')
        for client in self.clients:
            l.info(f"client {client.id} local model\'s accuracy is {client.evalute(self.test_dl)}")

    def evalute_global_model(self):
        l.info('[evalute_global_model]')
        accuracy = self.clients[0].evalute(self.test_dl)
        l.info(f"round {self.round},accuracy {accuracy}")
        return accuracy

    def aggregate_and_upload_global_model(self):
        l.info('[aggregate_and_upload_global_model]')
        self.clients[0].aggregate_and_upload()

    def simulate_sequential(self,n=1):
        # loop n times
        # the first round should upload init model,specific handle 
        if self.round == 0:
            self.aggregate_and_upload_global_model()
        for r in range(n):
            l.info(f'--------{self.round}--------')
            self.flesh_clients_model()
            self.clients_local_evalute()
            self.clients_local_train()
            self.clients_local_evalute()
            self.aggregate_and_upload_global_model()
            self.evalute_global_model()
            self.round += 1
    
    def simulate_parallel(self,n=1):
        l.info(f'--------{self.round}--------')
        self.round += 1