# coding: utf-8
import argparse
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import random
import pickle
import numpy as np
import torch.utils.data
from utils import aggregator, encoder, simpleAttention, decomposer
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class wrjModel(nn.Module):
    def __init__(self, u_embedding, i_embedding, embed_dim):
        super(wrjModel, self).__init__()

        self.u_embed = u_embedding
        self.i_embed = i_embedding
        self.embed_dim = embed_dim

        self.u_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.u_layer2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.i_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.i_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.ui_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ui_layer2 = nn.Linear(self.embed_dim, 16)
        self.ui_layer3 = nn.Linear(16, 1)

        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.i_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.ui_bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.ui_bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        nodes_u_embed = self.u_embed(nodes_u)
        nodes_i_embed = self.i_embed(nodes_i)

        x_u = F.relu(self.u_bn(self.u_layer1(nodes_u_embed)))
        x_u = F.dropout(x_u, training = self.training)
        x_u = self.u_layer2(x_u)

        x_i = F.relu(self.i_bn(self.i_layer1(nodes_i_embed)))
        x_i = F.dropout(x_i, training = self.training)
        x_i = self.i_layer2(x_i)

        x_ui = torch.cat((x_u, x_i), 1)
        x = F.relu(self.ui_bn1(self.ui_layer1(x_ui)))
        x = F.dropout(x, training = self.training)
        x = F.relu(self.ui_bn2(self.ui_layer2(x)))
        x = F.dropout(x, training = self.training)

        scores = self.ui_layer3(x)

        return scores.squeeze()

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)

        return self.criterion(scores, ratings)

def train(model, train, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train, 0):
        batch_u, batch_i, batch_ratings = data
        optimizer.zero_grad()
        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device))
        loss.backward(retain_graph = True)
        optimizer.step()
        avg_loss += loss.item()

        if i % 100 == 0:
            print('Training: [%d epoch, %5d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                epoch, i, avg_loss / 100, rmse_mn, mae_mn))
            avg_loss = 0.0

    return 0

def test(model, test, device):
    model.eval()
    pred = []
    ground_truth = []

    with torch.no_grad():
        for test_u, test_i, test_ratings in test:
            test_u, test_v, test_ratings = test_u.to(device), test_v.to(device), test_ratings.to(device)
            scores = model.forward(test_u, test_v)
            pred.append(list(scores.data.cpu().numpy()))
            ground_truth.append(list(test_ratings.data.cpu().numpy()))

    pred = np.array(sum(pred, []))
    ground_truth = np.array(sum(ground_truth, []))

    rmse = sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    return rmse, mae

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'wrj model')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--patience', type = int, default = 100,
                        metavar = 'N', help = 'for early stopping strategy')
    parser.add_argument('--lr', type = float, default = 0.001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 64,
                        metavar = 'N', help = 'embedding size')
    parser.add_argument('--n_components', type = int, default = 3,
                        metavar = 'N', help = 'number of components')
    parser.add_argument('--batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--val_batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for validating')
    parser.add_argument('--test_batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'amazon',
                        metavar = 'STR', help = 'dataset')
    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('----- Hyperparams -----')
    print('patience: ' + str(args.patience))
    print('lr: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('number of latent components: ' + str(args.n_components))

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    dataset_dir = './datasets/'
    dataset = args.dataset
    data_path = dataset_dir + dataset + ".pickle"
    data_file = open(data_path, 'rb')

    ufeature, ifeature, u_adj, i_adj, \
    u_train, i_train, r_train, u_val, i_val, r_val, u_test, i_test, r_test = pickle.load(data_file)
    """
    ## dataset
    ufeature: users' feature
    ifeature: items' feature

    u_adj: user's purchased history (item set in training set)
    i_adj: user set (in training set) who have interacted with the item

    u_train, i_train, r_train: training set (user, item, rating)
    u_val, i_val, r_val: validating set (user, item, rating)
    u_test, i_test, r_test: testing set (user, item, rating)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(i_train),
                                                torch.FloatTensor(r_train))
    valset = torch.utils.data.TensorDataset(torch.LongTensor(u_val), torch.LongTensor(i_val),
                                                torch.FloatTensor(r_val))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(i_test),
                                                torch.FloatTensor(r_test))

    train = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size,
                                        shuffle = True)
    val = torch.utils.data.DataLoader(valset, batch_size = args.val_batch_size,
                                        shuffle = True)
    test = torch.utils.data.DataLoader(testset, batch_size = args.test_batch_size,
                                        shuffle = True)

    ufeature_size = ufeature.__len__()
    ifeature_size = ifeature.__len__()

    u_f2v_1 = decomposer(ufeature, ufeature_size, embed_dim, device)
    u_f2v_2 = decomposer(ufeature, ufeature_size, embed_dim, device)
    u_f2v_3 = decomposer(ufeature, ufeature_size, embed_dim, device)
    i_f2v_1 = decomposer(ifeature, ifeature_size, embed_dim, device)
    i_f2v_2 = decomposer(ifeature, ifeature_size, embed_dim, device)
    i_f2v_3 = decomposer(ifeature, ifeature_size, embed_dim, device)

    # user
    u_agg_embed_1 = aggregator(u_f2v_1, i_f2v_1, embed_dim, cuda = device,
                                is_user = True)
    u_embed_1 = encoder(u_f2v_1, embed_dim, u_adj, u_agg_embed_1, cuda = device)
    u_agg_embed_2 = aggregator(u_f2v_2, i_f2v_2, embed_dim, cuda = device,
                                is_user = True)
    u_embed_2 = encoder(u_f2v_2, embed_dim, u_adj, u_agg_embed_2, cuda = device)
    u_agg_embed_3 = aggregator(u_f2v_3, i_f2v_3, embed_dim, cuda = device,
                                is_user = True)
    u_embed_3 = encoder(u_f2v_3, embed_dim, u_adj, u_agg_embed_3, cuda = device)

    u_final_embed = simpleAttention(u_embed_1, u_embed_2, u_embed_3, cuda = device)

    # item
    i_agg_embed_1 = aggregator(u_f2v_1, i_f2v_1, embed_dim, cuda = device,
                                is_user = False)
    i_embed_1 = encoder(i_f2v_1, embed_dim, i_adj, i_agg_embed_1, cuda = device)
    i_agg_embed_2 = aggregator(u_f2v_2, i_f2v_2, embed_dim, cuda = device,
                                is_user = False)
    i_embed_2 = encoder(i_f2v_2, embed_dim, i_adj, i_agg_embed_2, cuda = device)
    i_agg_embed_3 = aggregator(u_f2v_3, i_f2v_3, embed_dim, cuda = device,
                                is_user = False)
    i_embed_3 = encoder(i_f2v_3, embed_dim, i_adj, i_agg_embed_3, cuda = device)

    i_final_embed = simpleAttention(i_embed_1, i_embed_2, i_embed_3, device)

    # model
    wrjmodel = wrjModel(u_final_embed, i_final_embed, embed_dim).to(device)
    optimizer = torch.optim.RMSprop(wrjmodel.parameters(), lr = args.lr, alpha = 0.9)

    rmse_mn = np.inf
    mae_mn = np.inf

    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        # ================   training    ================
        train(wrjmodel, train, optimizer, epoch, rmse_mn, mae_mn, device)
        # ================     val       ================
        rmse, mae = test(wrjmodel, val, device)

        if rmse_mn > rmse:
            rmse_mn = rmse
            mae_mn = mae
            endure_count = 0
        else:
            endure_count += 1

        print("Val: RMSE: %.5f, MAE:%.5f " % (rmse, mae))

        if endure_count > args.patience:
            print('Early stop! The best RMSE/MAE: %.5f / %.5f' % (rmse_mn, mae_mn))
            print("Early stop model: RMSE: %.5f, MAE:%.5f " % (rmse, mae))
            break

    rmse, mae = test(wrjmodel, test, device)
    print("Test: RMSE: %.5f, MAE:%.5f " % (rmse, mae))

if __name__ == "__main__":
    main()
