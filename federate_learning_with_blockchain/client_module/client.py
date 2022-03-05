from turtle import update
from client_module.invoker import MockInvoker
from client_module.trainer import Trainer
from client_module.ipfs_client import IPFSwrapper
from client_module.log import logger as l
import copy
from typing import Dict, Tuple, Sequence

class MockClient(object):
    def __init__(self, setting):
        # l.debug(f"type of train_setting is {type(setting)}")
        assert isinstance(setting, dict)
        self.invoker = MockInvoker(setting)
        self.trainer = Trainer(setting)
        self.ipfs = IPFSwrapper(setting)
        self.id = setting["id"]

    def flesh_global_model(self):
        # get the lasted global model
        bytes_model_hash, version = self.invoker.get_global_model_hash()
        bytes_model = self.ipfs.get_bytes(bytes_model_hash)
        self.trainer.load_bytes_model(bytes_model)
        l.info(f"client {self.id} flesh global model,version {version} ,hash {bytes_model_hash}")

    def local_train(self):
        l.info(f"client {self.id} start local training...")
        self.trainer.local_training()
        l.info(f"client {self.id} finish local training...")

    def cur_model_hash(self):
        bytes_model = self.trainer.get_bytes_model()
        model_hash = self.ipfs.add_bytes(bytes_model)
        return model_hash

    def upload_model_update(self):
        data_size = self.trainer.get_data_size()
        bytes_model = self.trainer.get_bytes_model()
        bytes_model_hash = self.ipfs.add_bytes(bytes_model)
        self.invoker.upload_model_update(data_size, bytes_model_hash)
        l.info(f"client {self.id} upload model update,hash {bytes_model_hash}")

    def aggregate_and_upload(self):
        update_info_list = self.invoker.get_model_updates()
        format_info_list = []
        # get all update model from ipfs  with given model_hash
        for update in update_info_list:
            model_hash = update['bytes_model_hash']
            bytes_model = self.ipfs.get_bytes(model_hash)
            update['bytes_model'] = bytes_model
            format_info_list.append(update)
        bytes_aggregate_model = self.trainer.aggregate(format_info_list)
        self.trainer.load_bytes_model(bytes_aggregate_model)
        l.info(f"client {self.id} aggregate model finish")
        aggreagate_model_hash = self.ipfs.add_bytes(bytes_aggregate_model)
        self.invoker.upload_aggregation_hash(aggreagate_model_hash)
        l.info(f"client {self.id} aggregate model and upload")

    def evaluate(self, test_dl) -> Tuple[float,float]:
        accuracy,loss = self.trainer.evaluate(test_dl)
        return accuracy,loss
