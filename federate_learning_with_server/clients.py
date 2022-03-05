import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset_helper import DatasetHelper
from log import logger

class Client(object):
    def __init__(self, train_dataset, dev):
        self.train_ds = train_dataset
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def local_update(self, epochs, batch_size, net, loss_fn, opti, global_params):
        """
        :param epochs:
        :param batch_size:
        :param net:
        :param loss_fn:
        :param opti:
        :param global_params:
        :return: 当前client本地训练后的local_params
        """

        # load global params
        net.load_state_dict(global_params, strict=True)
        # load local data
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for data, label in self.train_dl:
                label = label.type(torch.LongTensor)
                data, label = data.to(self.dev), label.to(self.dev)
                opti.zero_grad()
                pred = net(data)
                loss = loss_fn(pred, label)
                loss.backward()
                opti.step()
        return net.state_dict()


class Cluster(object):
    def __init__(self, dataset_name, is_IID, n_clients, dev,attacked):
        self.dataset_name = dataset_name
        self.is_IID = is_IID
        self.n_clients = n_clients
        self.dev = dev
        self.clients_set = {}
        self.test_x_loader = None
        self.attacked = attacked
        self._balance_alloc_dataset()

    def _balance_alloc_dataset(self):
        """
        :return:
        """
        dataset_helper = DatasetHelper(self.dataset_name, self.is_IID)
        test_data = torch.tensor(dataset_helper.test_x)
        test_label = torch.argmax(torch.tensor(dataset_helper.test_y), dim=1)
        self.test_x_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_x = dataset_helper.train_x
        train_y = dataset_helper.train_y

        # every client get 2 shard
        # 60000 // 100 // 2 = 600 // 2 = 300
        shard_size = dataset_helper.train_x_size // self.n_clients // 2

        # permutation
        # np.random.permutation(60000 // 300= 200)
        shards_id = np.random.permutation(dataset_helper.train_x_size // shard_size)
        logger.debug(f"shards_id\'s shape:{shards_id.shape},shard_size is {shard_size}")
        for i in range(self.n_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            # if i = 10 ,shards_id1 = 10 * 2 = 20 ,shards_id2 = 10 * 2 + 1 =20
            # train_x[20*300
            data_shards1 = train_x[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_x[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_y[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_y[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            # attacker 0..4..8
            if i%4 == 0 and self.attacked:
                logger.debug("exist attack %d",i)
                poision_label = {
                    1:0,
                    2:4,
                    3:9,
                    4:2,
                    5:5,
                    6:8,
                    7:7,
                    8:6,
                    9:3,
                    0:1,
                }
                logger.debug(f'local_label before poison{local_label[:20]}')
                local_label = np.array(list(map(lambda x:poision_label[x],local_label)))
                logger.debug(f"local_label after poison{local_label[:20]}")
            client = Client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = client

if __name__=="__main__":
    MyClients = Cluster('mnist', True, 100, 0)
    print(Client)
    print(MyClients.clients_set['client10'].train_ds[0:10])
    train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    i = 0
    for x_train in train_ids[0]:
        print("client10 数据:"+str(i))
        print(x_train)
        i = i+1
    print(MyClients.clients_set['client11'].train_ds[400:500])


