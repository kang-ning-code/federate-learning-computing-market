from cmath import log
from client_module.log import logger as l
import gzip
import os
import numpy as np
import torch
def hex2bytes(hex_str):
    '''
    transfer hex_str(type HexStr) to bytes
    '''
    return bytes.fromhex(str(hex_str)[2:])

def build_mnist_dataset(is_iid,dataset_dir):
    '''
    split mnnist_dataset into n_client piece
    '''
    train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz')

    train_images = extract_images(train_images_path) # (60000,28,28,1) train_images' shape
    train_labels = extract_labels(train_labels_path) # (60000,10) train_labels' shape
    test_images = extract_images(test_images_path)
    test_labels = extract_labels(test_labels_path)

    n_train_images = train_images.shape[0]
    n_test_images = test_images.shape[0]
    # reshape & normalization
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    train_images = train_images.astype(np.float32)
    train_images = np.multiply(train_images, 1.0 / 255.0)
    test_images = test_images.astype(np.float32)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    test_x = test_images
    test_y = test_labels
    train_x,train_y = None,None
    if is_iid:
        order = np.arange(n_train_images)
        np.random.shuffle(order)
        train_x = train_images[order]
        train_y = train_labels[order]
    else:
        labels = np.argmax(train_labels, axis=1)
        order = np.argsort(labels)
        train_x = train_images[order]
        train_y = train_labels[order]
    return train_x,train_y,test_x,test_y
    
def split_mnist_dataset(is_iid,dataset_dir,n_clients):
    train_x,train_y,test_x,test_y = build_mnist_dataset(is_iid,dataset_dir)
    # every client get 2 shard
    # 60000 // 100 // 2 = 600 // 2 = 300
    train_x_size = train_x.shape[0]
    shard_size = train_x_size// n_clients // 2
    # the id list of shard, every shard's size is [shard_size]
    shard_ids= np.random.permutation(train_x_size // shard_size)
    l.info(f"train_x_size:{train_x_size},shard_size:{shard_size},shard_ids.len:{len(shard_ids)}")
    split_dataset_tensor = {}
    for i in range(n_clients):
        first_id = shard_ids[i * 2]
        second_id = shard_ids[i * 2 + 1]
        # if i = 10 ,first_id = 10 * 2 = 20 ,second_id = 10 * 2 + 1 =20
        first_shard = train_x[first_id * shard_size: first_id * shard_size + shard_size]
        second_shard = train_x[second_id * shard_size: second_id * shard_size + shard_size]
        label_shards1 = train_y[first_id * shard_size: first_id * shard_size + shard_size]
        label_shards2 = train_y[second_id * shard_size: second_id * shard_size + shard_size]
        client_train_x, client_train_y = np.vstack((first_shard, second_shard)), np.vstack((label_shards1, label_shards2))
        client_train_y = np.argmax(client_train_y, axis=1)
        x_tensor ,y_tensor= torch.tensor(client_train_x),torch.tensor(client_train_y)
        if i == 0:
            l.info(f"splited_train_x_tensor.shape:{x_tensor.shape},splited_train_y_tensor.shape:{y_tensor.shape}")
        split_dataset_tensor[i]=(x_tensor,y_tensor)
    test_x_tensor = torch.tensor(test_x)
    test_y_tensor = torch.argmax(torch.tensor(test_y), dim=1)
    return split_dataset_tensor,test_x_tensor,test_y_tensor

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
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

def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
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

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


