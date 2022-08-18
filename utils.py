import os
import numpy as np
import pandas as pd
from conf import conf


def label_skew(data, label, K, n_parties, beta, min_require_size=10):
    """
    :param data: 数据dataframe
    :param label: 标签列名
    :param K: 标签数
    :param n_parties:参与方数
    :param beta: 狄利克雷参数
    :param min_require_size: 点最小数据量，如果低于这个数字会重新划分，保证每个节点数据量不会过少
    :return: 根据狄利克雷分布划分数据到各个参与方
    """
    y_train = data[label]

    min_size = 0
    partition_all = []
    front = np.array([0])
    N = y_train.shape[0]  # N样本总数
    # return train_datasets, test_dataset, n_input, number_samples
    split_data = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            back = np.array([idx_k.shape[0]])
            partition = np.concatenate((front, proportions, back), axis=0)
            partition = np.diff(partition)  # 根据切分点求差值来计算各标签划分数据量
            partition_all.append(partition)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

    # 根据各节点数据index划分数据
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]

    return split_data, partition_all


def get_data():
    train_dataset = os.path.join(conf["dataroot"], f'{conf["which_dataset"]}/{conf["which_dataset"]}_train.csv')
    test_dataset = os.path.join(conf["dataroot"], f'{conf["which_dataset"]}/{conf["which_dataset"]}_test.csv')
    train_data = pd.read_csv(train_dataset)
    test_data = pd.read_csv(test_dataset)
    train_data, partition_all = label_skew(train_data,
                                           conf["label_column"][conf["which_dataset"]],
                                           conf["num_classes"][conf["which_dataset"]],
                                           conf["num_parties"], conf["beta"])
    print("各节点数据划分完成")
    return train_data,  test_data
