from turtle import update
from invoker import MockInvoker
from trainer import Trainer
import copy
class MockClient(object):
    def __init__(self,setting):
        assert isinstance(setting,map)
        self.invoker = MockInvoker(setting)
        self.trainer = Trainer(setting)
        self.id = id

    def flesh_global_model(self):
        bytes_model,version = self.invoker.get_global_model()
        self.trainer.load_bytes_model(bytes_model)

    def local_train(self):
        self.trainer.local_training()
    
    def upload_model_update(self):
        bytes_model = self.trainer.get_bytes_model()
        data_size = self.trainer.get_data_size()
        self.invoker.upload_model_update(data_size,bytes_model)

    def aggregate_and_update(self):
        update_models = self.invoker.get_model_updates()
        bytes_aggregate_model = self.trainer.aggregate(update_models)
        self.invoker.upload_aggregation(bytes_aggregate_model)
    
    def evalute(self,test_dl):
        accuracy =  self.trainer.evaluate(test_dl)
        return accuracy