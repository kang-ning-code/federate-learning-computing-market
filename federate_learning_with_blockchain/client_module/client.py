from turtle import update
from client_module.invoker import MockInvoker
from client_module.trainer import Trainer,ModelInfo
from client_module.ipfs_client import IPFSwrapper
from client_module.log import logger as l
import copy
from typing import Dict, Tuple, Sequence,List

class MockClient(object):
    def __init__(self, setting):
        # l.debug(f"type of train_setting is {type(setting)}")
        assert isinstance(setting, dict)
        self.invoker = MockInvoker(setting)
        self.trainer = Trainer(setting)
        self.ipfs = IPFSwrapper(setting)
        self.id = setting["id"]

    def get_format_model_updates(self)->List[ModelInfo]:
        update_info_list = self.invoker.get_model_updates()
        model_infos = []
        # get all update model from ipfs  with given model_hash
        for update in update_info_list:
            uploader = update['uploader']
            train_size = update['train_size']
            version = update['version']
            bytes_model_hash = update['bytes_model_hash']
            poll = update['poll']
            bytes_model = self.ipfs.get_bytes(bytes_model_hash)
            model_info = ModelInfo(uploader, train_size, version, bytes_model,poll = poll,bytes_model_hash=bytes_model_hash)
            model_infos.append(model_info)
        return model_infos

    def get_bytes_global_model(self)->bytes:
        model_infos = self.get_format_model_updates()
        bytes_model = self.trainer.aggregate(model_infos)
        return bytes_model

    def flesh_global_model(self):
        bytes_model = self.get_bytes_global_model()
        self.trainer.load_bytes_model(bytes_model)

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

    # def aggregate_and_upload(self):
    #     update_info_list = self.invoker.get_model_updates()
    #     model_infos = []
    #     # get all update model from ipfs  with given model_hash
    #     for update in update_info_list:
    #         uploader = update['uploader']
    #         train_size = update['train_size']
    #         version = update['version']
    #         bytes_model_hash = update['bytes_model_hash']
    #         bytes_model = self.ipfs.get_bytes(bytes_model_hash)
    #         model_info = ModelInfo(uploader, train_size, version, bytes_model,bytes_model_hash=bytes_model_hash)
    #         model_infos.append(model_info)
    #     bytes_aggregate_model = self.trainer.aggregate(model_infos)
    #     self.trainer.load_bytes_model(bytes_aggregate_model)
    #     l.info(f"client {self.id} aggregate model finish")
    #     aggreagate_model_hash = self.ipfs.add_bytes(bytes_aggregate_model)
    #     self.invoker.upload_aggregation_hash(aggreagate_model_hash)
    #     l.info(f"client {self.id} aggregate model and upload")

    def evaluate(self, test_dl) -> Tuple[float,float]:
        accuracy,loss,_ = self.trainer.evaluate(test_dl)
        return accuracy,loss

    def vote(self):
        model_infos = self.get_format_model_updates()
        candidates = self.trainer.get_vote_for_list(model_infos)
        l.debug(f'client {self.id} vote for {candidates}')
        self.invoker.vote(candidates)
    
    def init_model(self):
        bytes_model = self.trainer.get_bytes_model()
        init_model_hash = self.ipfs.add_bytes(bytes_model)
        self.invoker.init_model(init_model_hash)
        l.info(f"client {self.id} init model {init_model_hash}")
        



