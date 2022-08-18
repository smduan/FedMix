import os
import sys

import copy
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from fedavg.server import Server
from fedavg.client import Client
from conf import conf
from fedavg.models import CNN_Model, weights_init_normal, MLP
from utils import get_data


def min_max_norm(train_datasets, test_dataset, cat_columns, label):

    train_data = None
    for key in train_datasets.keys():
        train_datasets[key]['tag'] = key
        train_data = pd.concat([train_data, train_datasets[key]])
    test_dataset['tag'] = key+1
    data = pd.concat([train_data, test_dataset])

    min_max = MinMaxScaler()
    con = []

    # 查找连续列
    for c in data.columns:
        if c not in cat_columns and c not in [label, 'tag']:
            con.append(c)

    data[con] = min_max.fit_transform(data[con])

    # 离散列one-hot
    data = pd.get_dummies(data, columns=cat_columns)

    for key in train_datasets.keys():
        c_data = data[data['tag'] == key]
        c_data = c_data.drop(columns=['tag'])
        train_datasets[key] = c_data

    test_dataset = data[data['tag'] == key+1]
    test_dataset = test_dataset.drop(columns=['tag'])

    return train_datasets, test_dataset


def main():

    cat_columns = conf["discrete_columns"][f"{conf['which_dataset']}"]
    
    if conf['diff_clients'] is False:
        train_files = [os.path.join(conf["dataroot"], f"{conf['which_dataset']}/beta{conf['beta']}/{conf['which_dataset']}_node_{j}.csv") for j in range(conf["num_parties"])]
    else:
        print('Notice: diff_clients is True, thus beta is fixed to 0.1.')
        rnd_id = np.random.choice(range(60), conf["num_parties"], replace=False)
        train_files = [os.path.join(conf["dataroot"], f"{conf['which_dataset']}/client60_beta0.1/{conf['which_dataset']}_node_{j}.csv") for j in rnd_id]
        print(rnd_id)
    train_datasets = {}

    for i in range(len(train_files)):
        train_datasets[i] = pd.read_csv(train_files[i])
        print(train_datasets[i][conf['label_column'][conf['which_dataset']]].value_counts())

    test_dataset = pd.read_csv(os.path.join(conf["dataroot"], f'{conf["which_dataset"]}/{conf["which_dataset"]}_test.csv'))
    print(test_dataset.shape)

    train_datasets, test_dataset = min_max_norm(train_datasets, test_dataset, cat_columns, conf["label_column"][conf['which_dataset']])

    # 初始化每个节点聚合权值
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("聚合权值初始化")
    print(client_weight)

    # 保存节点
    clients = {}
    # 保存节点模型
    clients_models = {}

    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"][conf["which_dataset"]])
    elif conf['model_name'] == 'cnn':
        # 训练目标模型
        model = CNN_Model()
    else:
        raise ValueError
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda()

    server = Server(conf, model, test_dataset)

    print("Server初始化完成!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, copy.deepcopy(server.global_model), train_datasets[key])

    print("参与方初始化完成！")

    # 保存模型
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    if not os.path.isdir(os.path.join(conf["model_dir"], "clients")):
        os.mkdir(os.path.join(conf["model_dir"], "clients"))
    if not os.path.isdir(os.path.join(conf["model_dir"], "server")):
        os.mkdir(os.path.join(conf["model_dir"], "server"))
    max_acc = 0
    max_auc = 0
    maxe = -1

    # 理论上，Xg Yg 应该存储在服务器上，但是鉴于本代码仅是模拟联邦学习的机制，为了代码方便，我把它们放在这里
    Xg, Yg = [], []
    up_cost, down_cost = 0., 0.
    for key in clients.keys():
        Xmean, ymean = clients[key].calculate_mean_data(mean_batch=conf["mean_batch"])
        Xg.append(Xmean)
        Yg.append(ymean)
        up_cost += sys.getsizeof(Xmean.storage())
        up_cost += sys.getsizeof(ymean.storage())
        down_cost += sys.getsizeof(Xmean.storage()) * (len(clients.keys()) - 1)
        down_cost += sys.getsizeof(ymean.storage()) * (len(clients.keys()) - 1)
    Xg = torch.cat(Xg, dim=0)
    Yg = torch.cat(Yg, dim=0)
    for key in clients.keys():
        clients[key].get_mean_data(Xg, Yg)
    print('up cost:', up_cost)
    print('down cost:', down_cost)

    # 联邦训练
    for e in range(conf["global_epochs"]):

        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model, conf["lambda"])
            clients_models[key] = copy.deepcopy(model_k)

        # 联邦聚合
        server.model_aggregate(clients_models, client_weight)

        # 测试全局模型
        acc, loss, auc_roc, f1 = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f, auc_roc: %f, f1: %f\n" % (e, acc, loss, auc_roc, f1))

        # 保存最好的模型
        if conf['num_classes'][conf['which_dataset']] == 2:
            if auc_roc >= max_auc:
                torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "server", "best_model.pth"))
                for key in clients.keys():
                    torch.save(clients_models[key], os.path.join(conf["model_dir"], "clients", "best_model-{}.pth".format(key)))
                print("model save done !")
                max_auc = auc_roc
                maxe = e
        else:
            if acc >= max_acc:
                torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "server", "best_model.pth"))
                for key in clients.keys():
                    torch.save(clients_models[key], os.path.join(conf["model_dir"], "clients", "best_model-{}.pth".format(key)))
                print("model save done !")
                max_acc = acc
                maxe = e

    if conf['num_classes'][conf['which_dataset']] == 2:
        print('max auc = {0}, epoch = {1}'.format(max_auc, maxe))
    else:
        print('max acc = {0}, epoch = {1}'.format(max_acc, maxe))


if __name__ == '__main__':
    main()
