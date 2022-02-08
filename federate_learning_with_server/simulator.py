import os
import argparse
import sys
import time

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


    cluster = Cluster('mnist', args['IID'], args['n_clients'], dev)
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

        sum_parameters = None

        # local train every participant
        for client in tqdm(clients_in_comm,unit='client',ncols=100):
            local_parameters = cluster.clients_set[client].local_update(args['epoch'], args['batch_size'], net,
                                                                        loss_fn, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # use the average as the update global_parameters
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / n_participants)
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        for test_x, test_y in test_x_loader:
            # get batch test_x & batch test_y
            test_x, test_y = test_x.to(dev), test_y.to(dev)
            pred = net(test_x)
            pred = torch.argmax(pred, dim=1)
            sum_accu += (pred == test_y).float().mean()
            num += 1
        logger.info(f'accuracy: {sum_accu/num}')
        result_file.write(f"round {i + 1} ,accuracy = {float(sum_accu / num)}\n")

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['check_point_path'],
                                         '{}_n_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batch_size'],
                                                                                                args['learning_rate'],
                                                                                                args['n_clients'],
                                                                                                args['c_fraction'])))

    result_file.close()
