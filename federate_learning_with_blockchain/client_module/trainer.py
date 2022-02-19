import torch
import model 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pickle
class Trainer(object):
    def __init__(self,train_setting):
        assert isinstance(train_setting,map)
        # training setting
        self.epochs = train_setting['epochs']
        self.batch_size = train_setting['batch_size']
        self.model_name = train_setting['model_name']
        self.learning_rate = train_setting['learning_rate']
        self.train_ds = train_setting['dataset']
        assert isinstance(self.train_ds,TensorDataset)
        # init training model
        self.model = model.get_model(self.model_name)
        dev = torch.device('cpu')
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(dev)
        self.opti = model.get_opti(self.model.parameters,self.learning_rate)
        self.loss_fn = model.get_loss_fn()
        # init dataloader
        self.train_dl = DataLoader(self.train_ds,batch_size= self.batch_size,shuffle=True)
         
    def load_bytes_model(self,bytes_model):
        assert isinstance(bytes_model,bytes)
        dict = pickle.loads(bytes_model)
        self._load_model(dict)

    def get_bytes_model(self):
        dict = self.model.state_dict()
        # transfer obj to bytes
        bytes_model = pickle.dumps(dict)
        return bytes_model

    def get_data_size(self):
        return len(self.train_dl)

    def _load_model(self,model_params_dict):
        self.model.load_state_dict(model_params_dict,strict=True)

    def local_training(self):
        for epoch in range(self.epochs):
            for train_x, train_y in self.train_dl:
                train_x, train_y = train_x.to(self.dev), train_y.to(self.dev)
                self.opti.zero_grad()
                pred_y = self.model(train_x)
                loss = self.loss_fn(pred_y, train_y)
                loss.backward()
                self.opti.step()
    
    def evaluate(self,test_dl):
        assert isinstance(test_dl,DataLoader)
        for test_x, test_y in test_dl:
            # get batch test_x & batch test_y
            test_x, test_y = test_x.to(self.dev), test_y.to(self.dev)
            pred = self.model(test_x)
            pred = torch.argmax(pred, dim=1)
            sum_accu += (pred == test_y).float().mean()
            num += 1
        accuary = sum_accu/num
        return accuary
    
    def aggregate(self,update_models):
        # [ (data_size_1,bytes_model_1) , .... , (data_size_n,bytes_model_n) ]
        assert isinstance(update_models,list)
        average_params = None
        all_data_size = 0
        for update_info in update_models:
            # model_info = {
            #     "uploader":
            #     "train_size":
            #     "version":
            #     "bytes_model":
            # }
            data_size = update_info['train_size']
            bytes_model = update_info['bytes_model']
            model_dict = pickle.loads(bytes_model)
            if average_params is None:
                average_params = {}
                for k,v in  model_dict.items():
                    average_params[k] = v.clone() * data_size
            else:
                for k in average_params:
                    average_params[k] = average_params[k] + model_dict[k] * data_size
            all_data_size  = all_data_size + data_size
        for k in average_params:
            average_params[k] = average_params / (all_data_size * 1.0)
        # bytes
        bytes_aggregate_model = pickle.dumps(average_params)
        return bytes_aggregate_model


