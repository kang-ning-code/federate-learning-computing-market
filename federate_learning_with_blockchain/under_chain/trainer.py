import torch
import model 
class Trainer(object):
    def __init__(self,train_setting):
        assert isinstance(train_setting,map)
        # training setting
        self.epochs = train_setting['epochs']
        self.batch_size = train_setting['batch_size']
        self.model_name = train_setting['model_name']
        
        # init for deeplearning
        dev = torch.device('cpu')
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)
            
    else:
        logger.info('use cpu as device')
        self.global_model_params= self.gen_init_model()
        self.net = model.get_model(self.model_name)
        self.opt = 
    def gen_init_model(self,model_name = ""):
        pass
    
    def update_local_model(self,global_model_params):
        self.global_model_params = global_model_params

    def _train_one_step(self):
        pass

    def 
