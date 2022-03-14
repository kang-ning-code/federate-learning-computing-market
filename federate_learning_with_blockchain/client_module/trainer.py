import hashlib
import pickle
from typing import Dict, Sequence, Tuple
import numpy as np
import torch
import copy
import math
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import client_module.model as model
from client_module.log import logger as l
from client_module.utils import find_max_clique
from typing import Dict,Tuple,List

class ModelInfo(object):
    def __init__(self,uploader:str,train_size:int,version:int,bytes_model:bytes,poll=1,bytes_model_hash=None):
        self.uploader = uploader
        self.train_size = train_size
        self.version = version
        self.bytes_model = model
        self.bytes_model_hash = bytes_model_hash
        self.poll = poll
        self.model_dict = pickle.loads(bytes_model)
           
class Trainer(object):
    def __init__(self,train_setting:dict):
        # l.debug(f"type of train_setting is {type(train_setting)}")
        assert isinstance(train_setting,dict)
        # training setting
        self.epochs = train_setting['epochs']
        self.batch_size = train_setting['batch_size']
        self.n_vote = train_setting['n_vote']
        self.n_participators = train_setting['n_participator']
        self.model_name = train_setting['model_name']
        self.learning_rate = float(train_setting['learning_rate'])
        self.train_ds = train_setting['dataset']
        self.test_ds = train_setting['dataset']
        self.aggregate_method = train_setting['aggreagate_method']
        assert isinstance(self.train_ds,TensorDataset)
        # init training model
        self.model = model.get_model(self.model_name)
        self.dev = torch.device('cpu')
        if torch.cuda.is_available():
            l.info(f'use cuda as dev')
            self.dev = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.dev)
        self.opti = model.get_opti(self.model,self.learning_rate)
        self.loss_fn = model.get_loss_fn()
        # init dataloader
        self.train_dl = DataLoader(self.train_ds,batch_size= self.batch_size,shuffle=True)
        self.test_dl = DataLoader(self.test_ds,batch_size = self.batch_size,shuffle=True)
    def load_bytes_model(self,bytes_model:bytes):
        assert isinstance(bytes_model,bytes)
        model_param_dict = pickle.loads(bytes_model)
        # l.debug(f"load model param,{type(model_param_dict)}")
        self._load_model(model_param_dict)

    def get_bytes_model(self)->bytes:
        model_param_dict = self.model.state_dict()
        # transfer obj to bytes
        bytes_model = pickle.dumps(model_param_dict)
        return bytes_model

    def get_data_size(self)->int:
        return len(self.train_dl)

    def get_model_abstract(self)->str:
        m = hashlib.md5()
        m.update(self.get_bytes_model())
        return m.hexdigest()

    def _load_model(self,model_params_dict:dict):
        self.model.load_state_dict(model_params_dict,strict=True)

    def local_training(self):
        for epoch in range(self.epochs):
            for train_x, train_y in self.train_dl:
                train_y = train_y.type(torch.LongTensor)
                train_x, train_y = train_x.to(self.dev), train_y.to(self.dev)
                self.opti.zero_grad()
                pred_y = self.model(train_x)
                loss = self.loss_fn(pred_y, train_y)
                loss.backward()
                self.opti.step()
        
    def evaluate(self,test_dl:DataLoader,return_output = False)->Tuple[float,float,torch.Tensor]:   
        correct = 0
        data_size = 0
        running_loss = 0.0
        total_output = None
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_dl:
                test_x, test_y = data
                test_x = test_x.to(self.dev)
                test_y = test_y.to(self.dev).long()
                # calculate outputs by running test_x through the network
                outputs = self.model(test_x)

                loss = self.loss_fn(outputs,test_y)
                running_loss += loss
                if return_output:
                    if total_output is None:
                        total_output = outputs
                    else:
                        total_output = torch.cat((total_output,outputs),0)
                _, predicted = torch.max(outputs.data, 1)
                data_size += test_y.size(0)
                correct += (predicted == test_y).sum().item()
            running_loss = running_loss.item() / data_size * test_dl.batch_size
            acc = correct / data_size
            return acc,running_loss,total_output
    
    def get_vote_for_list(self,model_infos:List[ModelInfo])->List[str]:
        # get the outs and accs
        accs = []
        for m in model_infos:
            self.model.load_state_dict(m.model_dict,strict=True)
            acc,_,_=self.evaluate(self.test_dl)
            accs.append(acc)

        args = np.argsort(accs)[::-1][:self.n_vote]
        l.debug(f'accs in current model is {accs},choice idx is {args}')
        candicates = []
        for idx in args:
            candicates.append(model_infos[idx].uploader)
        return candicates

    def aggregate(self,model_infos:List[ModelInfo])->bytes:
        if len(model_infos) == 0:
            return pickle.dumps(self.model.state_dict())
            
        aggregate_param = None
        if len(model_infos) == 1:
            aggregate_param =  model_infos[0].model_dict
        elif self.aggregate_method == 'sniper':
            l.info("use sniper to aggregate")
            aggregate_param = self.aggregate_sniper(model_infos)
        elif self.aggregate_method =='fed_vote_avg':
            l.info("use fed_vote_avg to aggregate")
            aggregate_param = self.aggregate_fed_vote_avg(model_infos)
        else:
            l.info("use fed_avg to aggregate")
            aggregate_param = self.aggregate_fed_avg(model_infos)
        self.model.load_state_dict(aggregate_param,strict=True)
        bytes_aggregate_model = pickle.dumps(aggregate_param)
        return bytes_aggregate_model

    def aggregate_fed_avg(self,model_infos:List[ModelInfo])->dict:
        average_params = None
        all_data_size = 0
        for model_info in model_infos:
            all_data_size  = all_data_size + model_info.train_size
        for model_info in model_infos:
            fraction = model_info.train_size / all_data_size
            l.debug(f'fraction of uploader {model_info.uploader} is {fraction},data_size is {model_info.train_size}')
            if average_params is None:
                average_params = {}
                for k,v in  model_info.model_dict.items():
                    average_params[k] = v.clone() * fraction
            else:
                for k in average_params:
                    average_params[k] = average_params[k] + model_info.model_dict[k] * fraction
        return average_params

    def aggregate_fed_vote_avg(self,model_infos:List[ModelInfo])->dict:
        average_params = None
        denominator = 0
        def sigmoid(z):
            return 1 / (1 + math.exp(-z))
        for model_info in model_infos:
            denominator += model_info.train_size * sigmoid(model_info.poll -self.n_vote)
        for model_info in model_infos:
            fraction = model_info.train_size * sigmoid(model_info.poll -self.n_vote) / denominator
            l.debug(f'fraction of uploader {model_info.uploader} is {fraction},data_size is {model_info.train_size},poll is {model_info.poll}')
            if average_params is None:
                average_params = {}
                for k,v in  model_info.model_dict.items():
                    average_params[k] = v.clone() * fraction
            else:
                for k in average_params:
                    average_params[k] = average_params[k] + model_info.model_dict[k] * fraction
        return average_params

    # deprecated
    def aggregate_sniper(self,model_infos:List[ModelInfo])->dict:
        
        # net = copy.deepcopy(self.model)
        n_participants = len(model_infos)
        # get the outs and accs
        accs = []
        for m in model_infos:
            self.model.load_state_dict(m.model_dict,strict=True)
            acc,loss,_=self.evaluate(self.test_dl,return_output=False)
            accs.append(acc)
        reshape_models = []
        for info in model_infos:
            sd = info.model_dict
            reshape_model = None
            for k in sd:
                if reshape_model is None:
                    reshape_model  = sd[k].reshape(-1) 
                else:
                    reshape_model  = torch.cat((reshape_model,sd[k].reshape(-1)))
            reshape_models.append(reshape_model)
        l.debug(f'reshape model size is {reshape_models[0].shape}')
        # get the dists
        dists = []
        for i in range(n_participants):
            i_dists = []
            for j in range(n_participants):
                dist =F.pairwise_distance(reshape_models[i],reshape_models[j],p=2).item()
                i_dists.append(dist)
            # print(i_dists)
            dists.append(i_dists)
        dists = np.array(dists)
        l.debug(f'dists is as follow:\n{dists}')
        # get the max clique whose size >= (n+1)/2
        stop = False
        gama = 0.0
        beta = 0.1
        clique = []
        clique_n = 0
        l.debug(f'dists.mean = {dists.mean()},std = {dists.std()}')
        while not stop:
            gama = gama + beta
            edge_threshold = 0.4 * dists.mean() + gama * dists.std()
            graph = dists <= edge_threshold
            clique,clique_n = find_max_clique(graph)
            if clique_n >= round(n_participants*0.6):
                stop = True
        l.debug(f'get the max clique,clique_n is {clique_n},clique is {clique}.gama is {gama}')
        aggregate_param = None
        alpha = 1.0
        beta = 0.1
        denominator = clique_n * alpha + (n_participants - clique_n) * beta
        props = []
        for idx,model_info in enumerate(model_infos):
            param = model_info.model_dict
            prop = alpha if clique[idx] else beta
            prop = prop / denominator
            props.append(prop)
            if aggregate_param is None:
                aggregate_param = {}
                for key, val in param.items():
                    aggregate_param[key] = val.clone() * prop
            else:
                for key in param:
                    aggregate_param[key] += param[key] * prop
        
        show_infos = []
        for idx in range(n_participants):
            l.debug(f'acc:{accs[idx]},selected:{clique[idx]},prop:{props[idx]}')
        return aggregate_param
      