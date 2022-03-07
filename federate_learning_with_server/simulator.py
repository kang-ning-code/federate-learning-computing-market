import os
import argparse
import sys
import time
import copy
from tqdm import tqdm,trange
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models import Mnist_2NN, Mnist_CNN
from clients import Cluster, Client
from model.WideResNet import WideResNet
from log import logger
import log
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--n_clients', type=int, default=100, help='numer of the clients')
# c_fraction 每次挑选的客户端占总客户端的比例
parser.add_argument('-cf', '--c_fraction', type=float, default=0.1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
#
parser.add_argument('-B', '--batch_size', type=int, default=10, help='local train batch size')
#
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
#
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="the dataset you input")
#
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
#
parser.add_argument('-ncomm', '--n_comm', type=int, default=1000, help='number of communications')
#
parser.add_argument('-ck', '--check_point_path', type=str, default='checkpoints', help='the saving folder path of model checkpoints')
#
parser.add_argument('-lp','--log_path',type=str,default='logs',help='the saving folder path of log.py file')
#
parser.add_argument('-rp','--result_path',type=str,default='results',help='the saving folder path of result file')
#
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
#
parser.add_argument('-at','--attack',type=int,default=4,help='exist attacker (every [at] clients exist 1 attacker)')
#
parser.add_argument('-ag','--aggregate',type=int,default=0,help= 'if aggragate is 1 ,use the specific aggragate method')

def find_max_clique(graph):
    global n,best_clique,best_clique_n,cur_clique,cur_clique_n
    n = graph.shape[0]
    best_clique = [False for _ in range(n)]
    best_clique_n = 0
    cur_clique = [False for _ in range(n)]
    cur_clique_n = 0
    def can_place(t):
        global cur_clique,graph
        for i in range(t):
            if cur_clique[i] and (not graph[i][t]):
                return False
        return True

    def backward(cur):
        global cur_clique,cur_clique_n,best_clique,best_clique_n
        if(cur >= n):
            # record the best_result
            for i in range(n):
                best_clique[i] = cur_clique[i]
            
            best_clique_n = cur_clique_n
            return 
        # place into cur clique
        if(can_place(cur)):
            cur_clique[cur] = True
            cur_clique_n = cur_clique_n +  1
            backward(cur+1)
            cur_clique_n = cur_clique_n - 1
            cur_clique[cur] = False
        # do not place into cur clique
        if(cur_clique_n + n-1-cur>best_clique_n):
            cur_clique[cur] = False
            backward(cur+1)
    backward(0)
    return best_clique,best_clique_n

def evaluate(net,test_x_loader):
    correct = 0
    total = 0
    total_output = None
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_x_loader:
            images, labels = data
            images = images.to(dev)
            labels = labels.to(dev)
            # calculate outputs by running images through the network
            outputs = net(images)
            if total_output is None:
                total_output = outputs
            else:
                total_output = torch.cat((total_output,outputs),0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
    return total_output,acc


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__


    if not os.path.exists(args['check_point_path']):
        os.mkdir(args['check_point_path'])
    if not os.path.exists(args['log_path']):
        os.mkdir(args['log_path'])
    if not os.path.exists(args['result_path']):
        os.mkdir(args['result_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    log.init_logging(log_level=logging.DEBUG, log_file=os.path.join(args['log_path'],time_str + '.log'))
    result_path = os.path.join(args['result_path'],time_str + '_result.txt')
    result_file = open(result_path, mode="w")

    dev = torch.device('cpu')
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        logger.info('use cuda as device')
    else:
        logger.info('use cpu as device')

    net = None
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet
    elif args['model_name'] == 'wide_res_net':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)
    # TODO add to parser
    loss_fn = F.cross_entropy
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])


    cluster = Cluster('mnist', args['IID'], args['n_clients'], dev,args['attack'])
    test_x_loader = cluster.test_x_loader

    logger.info('---finish prepare work---')
    n_participants = int(max(args['n_clients'] * args['c_fraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        logger.debug(f"tensor shape:{var.shape},size:{var.size()}")
        global_parameters[key] = var.clone()

    loop_times = args['n_comm']
    for i in range(loop_times):
        logger.info(f"---communicate round {i+1} start---")

        order = np.random.permutation(args['n_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:n_participants]]
        logger.debug(f'selected clients {clients_in_comm}')

        if args['aggregate'] == 0:
            sum_parameters = None
            for client in tqdm(clients_in_comm,unit='client',ncols=100):
                local_param = cluster.clients_set[client].local_update(args['epoch'], args['batch_size'], net,
                                                                        loss_fn, opti, global_parameters)
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_param.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_param[var]
            # use the average as the update global_parameters
            for var in global_parameters:
                global_parameters[var] = (sum_parameters[var] / n_participants)
            net.load_state_dict(global_parameters, strict=True)
            _,global_acc = evaluate(net, test_x_loader)
            logger.info(f'accuracy: {global_acc}')
            result_file.write(f"{i+1},{global_acc}\n")
            
            if (i + 1) % args['save_freq'] == 0:
                torch.save(net, os.path.join(args['check_point_path'],
                                            '{}_n_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                    i, args['epoch'],
                                                                                                    args['batch_size'],
                                                                                                    args['learning_rate'],
                                                                                                    args['n_clients'],
                                                                                                    args['c_fraction'])))
                                                                            
            continue



        # local train every participant
        ###############################################################
        local_params = []
        for client in tqdm(clients_in_comm,unit='client',ncols=100):
            local_param = cluster.clients_set[client].local_update(args['epoch'], args['batch_size'], net,
                                                                        loss_fn, opti, global_parameters)
            local_params.append(copy.deepcopy(local_param))

        ###############################################################
        
        # get all out(type tensor)
        ###############################################################
        outs = []
        accs = []
        for param in local_params :
            net.load_state_dict(param,strict=True)
            out,acc = evaluate(net,test_x_loader)
            outs.append(out)
            accs.append(acc)
        # logger.debug(f"accs is {accs}")
        ###############################################################

        # get the distance
        ###############################################################
        dists = []
        for idx in range(len(outs)):
            i_dists = []
            for j in range(len(outs)):
                dist =F.pairwise_distance(outs[idx].reshape(-1),outs[j].reshape(-1),p=2).item()
                i_dists.append(dist)
            # print(i_dists)
            dists.append(i_dists)
        dists = np.array(dists)
        # logger.debug(f'get the dist dist.shape{dists.shape}')
        ###############################################################

        # get the max clique whose size >= (n+1)/2
        ###############################################################
        stop = False
        gama = 0.1
        clique = []
        clique_n = 0
        logger.debug(f'dists.mean = {dists.mean()},std = {dists.std()}')
        while not stop:
            gama = gama * 2
            edge_threshold = 0.5 * dists.mean() + gama * dists.std()
            graph = dists <= edge_threshold
            # print(dists)
            # print(graph[:,0])
            clique,clique_n = find_max_clique(graph)
            if best_clique_n >= int(len(outs)+1)/2:
                stop = True
        logger.debug(f'get the max clique,clique_n is {clique_n},clique is {clique}.gama is {gama}')
        ###############################################################

        # aggregate the model
        ###############################################################
        aggregate_model = None
        alpha = 1.0
        beta = 0.1
        denominator = clique_n * alpha + (n_participants - clique_n) * beta
        props = []
        for idx,param in enumerate(local_params):
            prop = alpha if clique[idx] else beta
            prop = prop / denominator
            props.append(prop)
            if aggregate_model is None:
                aggregate_model = {}
                for key, val in param.items():
                    aggregate_model[key] = val.clone() * prop
            else:
                for key in param:
                    aggregate_model[key] += param[key] * prop
        for key in global_parameters:
            global_parameters[key] = copy.deepcopy(aggregate_model[key])
        net.load_state_dict(global_parameters, strict=True)
        # logger.debug(f'aggregate the model ,props is {props}')
        display = []
        for idx in range(n_participants):
            info = ((order[idx]+1)%args['attack']==0,clique[idx],props[idx],accs[idx])
            display.append(info)
        logger.debug(f"-----info<is_attaker,in_the_clique,prop,acc>---------:\n{display}")
        ###############################################################
        _,global_acc = evaluate(net,test_x_loader)
        logger.info(f'accuracy: {global_acc}')

        result_file.write(f"{i+1},{global_acc}\n")
        
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['check_point_path'],
                                         '{}_n_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['n_clients'],
                                                                                                args['c_fraction'])))

    result_file.close()
