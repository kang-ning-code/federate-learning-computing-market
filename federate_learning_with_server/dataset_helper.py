import numpy as np
import gzip
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from log import  logger
from typing import Tuple

def get_MNIST_data():
    transformer =transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST('./data',train=True,download=True,transform=transformer)
    testset = torchvision.datasets.MNIST('./data',train=False,download=True,transform=transformer)
    # train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    # test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)
    train_data = trainset.data.numpy()
    test_data = testset.data.numpy()
    train_label = trainset.targets.numpy()
    test_label = testset.targets.numpy()

    train_data = train_data.reshape(train_data.shape[0],-1)
    test_data = test_data.reshape(test_data.shape[0],-1)

    train_data = train_data.astype(np.float32)
    train_data = np.multiply(train_data, 1.0 / 255.0)

    test_data = test_data.astype(np.float32)
    test_data = np.multiply(test_data,1.0/255.0)
    
    return train_data,train_label,test_data,test_label

class DatasetLoader(object):
    def __init__(self,dataset_name='MNIST',is_iid = True,n_client = 10,batch_size = 5):
        self.dataset_name = dataset_name
        self.is_iid= is_iid
        self.n_client = n_client
        self.batch_size = batch_size
        self.train_loader = None
        self.tset_loader = None
        if dataset_name == 'MNIST':
            train_loader,test_loader = get_MNIST_loader(self.batch_size)

class DatasetHelper(object):
    def __init__(self, dataSetName, is_IID):
        self.name = dataSetName
        self.train_x = None
        self.train_y = None
        self.train_x_size = None
        self.test_x = None
        self.test_y = None
        self.test_x_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.construct_mnist_dataset(is_IID)
        elif self.name == 'cifar10':
            self.construct_cifar10_dataset(is_IID)
        else:
            pass
    # mnistDataSetConstruct
    def construct_mnist_dataset(self, is_IID):
        '''
            IID：
                独立同分布
            Non-IID：
                非独立同分布，将dataset按照label值排序后划分数据集
        '''

        data_dir = r'.\data\MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

        train_images = extract_images(train_images_path)
        logger.info(f'load train_x,shape:{train_images.shape}')  # (60000,28,28,1)
        train_labels = extract_labels(train_labels_path)
        logger.info(f'load train_y,shape:{train_labels.shape}')
        test_images = extract_images(test_images_path)
        logger.info(f'load test_x,shape:{test_images.shape}')
        test_labels = extract_labels(test_labels_path)
        logger.info(f'load test_y,shape:{test_labels.shape}')

        self.train_x_size = train_images.shape[0]
        self.test_x_size = test_images.shape[0]

        # reshape & normalization
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
        logger.debug(f'after reshape,train_x\'s shape:{train_images.shape},test_x\'s shape{test_images.shape}')
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if is_IID:

            order = np.arange(self.train_x_size)
            '''
                num = np.arange(20)
                print(num)
                # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
                np.random.shuffle(num)
                print(num)
                # [ 1  5 19  9 14  2 12  3  6 18  4  8 16  0 10 17 13  7 15 11]
            '''
            np.random.shuffle(order)
            self.train_x = train_images[order]
            self.train_y = train_labels[order]
        else:
            '''
                # numpy.argmax(array, axis)
                two_dim_array = np.array([[1, 3, 5], [0, 4, 3]])
                max_index_axis0 = np.argmax(two_dim_array, axis = 0)
                max_index_axis1 = np.argmax(two_dim_array, axis = 1)
                print(max_index_axis0)
                print(max_index_axis1)
                # [0 1 0] 
                # [2 1]
            '''
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_x = train_images[order]
            self.train_y = train_labels[order]

        self.test_x = test_images
        self.test_y = test_labels

    def construct_cifar10_dataset(self, isIID):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为
        print(type(train_labels))  # <class 'numpy.ndarray'>
        print(train_labels.shape)  # (50000,)

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)
        # print()

        self.train_x_size = train_data.shape[0]
        self.test_x_size = test_data.shape[0]

        # 将训练集转化为（50000，32*32*3）矩阵
        train_images = train_data.reshape(train_data.shape[0],
                                          train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        print(train_images.shape)
        # 将测试集转化为（10000，32*32*3）矩阵
        test_images = test_data.reshape(test_data.shape[0],
                                        test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        # ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # ----------------------------------------------------------------#

        '''
            一工有60000个样本
            100个客户端
            IID：
                我们首先将数据集打乱，然后为每个Client分配600个样本。
            Non-IID：
                我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
                然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
        '''
        if isIID:
            # 这里将50000 个训练集随机打乱
            order = np.arange(self.train_x_size)
            np.random.shuffle(order)
            self.train_x = train_images[order]
            self.train_y = train_labels[order]
        else:
            # 按照标签的
            # labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序
            order = np.argsort(train_labels)
            print("标签下标排序")
            print(train_labels[order[20000:25000]])
            self.train_x = train_images[order]
            self.train_y = train_labels[order]

        self.test_x = test_images
        self.test_y = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    logger.info(f'extracting {filename}')
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    logger.info(f'extracting {filename}')
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__ == "__main__":
    'test data set'
    mnistDataSet = DatasetHelper('mnist', 0)  # test NON-IID
    if type(mnistDataSet.train_x) is np.ndarray and type(mnistDataSet.test_x) is np.ndarray and \
            type(mnistDataSet.train_y) is np.ndarray and type(mnistDataSet.test_y) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_x.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_x.shape))
    print(mnistDataSet.train_y[0:100], mnistDataSet.train_y[11000:11100])
