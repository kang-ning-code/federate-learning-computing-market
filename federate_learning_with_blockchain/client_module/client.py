from turtle import update
from client_module.invoker import MockInvoker
from client_module.trainer import Trainer
from client_module.log import logger as l
import copy
class MockClient(object):
    def __init__(self,setting):
        # l.debug(f"type of train_setting is {type(setting)}")
        assert isinstance(setting,dict)
        self.invoker = MockInvoker(setting)
        self.trainer = Trainer(setting)
        self.id = setting["id"]

    def flesh_global_model(self):
        bytes_model,version = self.invoker.get_global_model()
        l.info(f"client {self.id} get version {version} global bytes model")
        self.trainer.load_bytes_model(bytes_model)

    def local_train(self):
        l.info(f"client {self.id} start local training")
        self.trainer.local_training()
        l.info(f"client {self.id} finish local training")

    def upload_model_update(self):
        l.info(f"client {self.id} upload model update")
        
        data_size = self.trainer.get_data_size()
        self.invoker.upload_model_update(data_size,bytes_model)
        

    def aggregate_and_upload(self):
        l.info(f"client {self.id} aggregate model and upload")
        update_models = self.invoker.get_model_updates()
        bytes_aggregate_model = self.trainer.aggregate(update_models)
        self.invoker.upload_aggregation(bytes_aggregate_model)
        

    def evalute(self,test_dl):
        accuracy =  self.trainer.evaluate(test_dl)
        return accuracy