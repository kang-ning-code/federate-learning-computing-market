from brownie import accounts
import client_module.utils as utils
from client_module.log import logger as l

setting_item = [
    "batchSize",
    "learningRate",
    "epochs",
    "nParticipators",
]

class MockInvoker(object):

    def __init__(self,setting) :
        self.contract = setting['contract']
        self.account = setting['account']
        
    def get_setting(self):
        setting_list = self.contract.setting({"from":self.account})
        assert len(setting_list) == 4
        setting_map = {
            'batchSize':setting_list[0],
            'learningRate':float(setting_list[1]),
            'epochs':setting_list[2],
            'nParticipators':setting_list[3],
        }
        return setting_map

    def get_global_model_hash(self):
        bytes_model_param_hash,lastestVersion = self.contract.getGlobalModel({"from":self.account})
        return bytes_model_param_hash,lastestVersion

    def get_model_updates(self):
        return_value = self.contract.getModelUpdates({"from":self.account})
        model_update_list = return_value
        l.debug(f'get_model_updates\'s len :{len(model_update_list)}')
        format_model_updates = []
        for model_update in model_update_list:
            model_info = {
                "uploader":model_update[0],
                "train_size":model_update[1],
                "version":model_update[2],
                "bytes_model_hash":model_update[3],
            }
            format_model_updates.append(model_info)
        return format_model_updates

    def upload_aggregation_hash(self,aggregated_model_hash):
        assert isinstance(aggregated_model_hash,str)
        self.contract.uploadAggregation(aggregated_model_hash,{"from":self.account,"gas_limit":1000000000})

    def upload_model_update(self,data_size,update_model_hash):
        assert isinstance(update_model_hash,str)
        self.contract.uploadModelUpdate(data_size,update_model_hash,{"from":self.account})
    
    
