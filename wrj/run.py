# coding: utf-8
import argparse
import os
import torch
import pickle
import numpy as np
import torch.utils.data
from utils import aggregator, encoder, simpleAttention, componentProcess


def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'wrj model')
    parser.add_argument('--batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--patience', type = int, default = 5,
                        metavar = 'N')
    parser.add_argument('--lr', type = float, default = 0.001,
                        metavar = 'LR', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 64,
                        metavar = 'N', help = 'embedding size')
    parser.add_argument('--n_components', type = int, default = 3,
                        metavar = 'N', help = 'number of components')
    parser.add_argument('--val_batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for validating')
    parser.add_argument('--test_batch_size', type = int, default = 1024,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'amazon',
                        metavar = 'STR', help = 'dataset')
    args = parser.parse_args()

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

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(u_train), torch.LongTensor(v_train),
                                                torch.FloatTensor(r_train))
    valset = torch.utils.data.TensorDataset(torch.LongTensor(u_val), torch.LongTensor(v_val),
                                                torch.FloatTensor(r_val),
                                                torch.FloatTensor(ufeature_val), torch.FloatTensor(ifeature_val))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(u_test), torch.LongTensor(v_test),
                                                torch.FloatTensor(r_test)
                                                torch.FloatTensor(ufeature_test), torch.FloatTensor(ifeature_test))

    train = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size,
                                        shuffle = True)
    val = torch.utils.data.DataLoader(valset, batch_size = args.val_batch_size,
                                        shuffle = True)
    test = torch.utils.data.DataLoader(testset, batch_size = args.test_batch_size,
                                        shuffle = True)

    ufeature_size = ufeature_train.__len__()
    ifeature_size = ifeature_train.__len__()

    u_f2v_1 = componentProcess(ufeature, ufeature_size, embed_dim, device)
    u_f2v_2 = componentProcess(ufeature, ufeature_size, embed_dim, device)
    u_f2v_3 = componentProcess(ufeature, ufeature_size, embed_dim, device)
    i_f2v_1 = componentProcess(ifeature, ifeature_size, embed_dim, device)
    i_f2v_2 = componentProcess(ifeature, ifeature_size, embed_dim, device)
    i_f2v_3 = componentProcess(ifeature, ifeature_size, embed_dim, device)

    # user
    u_tmp_embed_1 = aggregator(u_f2v_1, i_f2v_1, embed_dim,
                                cuda = device, is_user = True)
    u_embed_1 = encoder(u_f2v_1, embed_dim, u_adj, u_adj_rating, u_tmp_embed_1,
                                cuda = device, is_user = True)
    u_tmp_embed_2 = aggregator(u_f2v_2, i_f2v_2, embed_dim,
                                cuda = device, is_user = True)
    u_embed_2 = encoder(u_f2v_2, embed_dim, u_adj, u_adj_rating, u_tmp_embed_2,
                                cuda = device, is_user = True)
    u_tmp_embed_3 = aggregator(u_f2v_3, i_f2v_3, embed_dim,
                                cuda = device, is_user = True)
    u_embed_3 = encoder(u_f2v_3, embed_dim, u_adj, u_adj_rating, u_tmp_embed_3,
                                cuda = device, is_user = True)

    u_final_embed = simpleAttention(u_embed_1, u_embed_2, u_embed_3, device)

    # item
    i_tmp_embed_1 = aggregator(u_f2v_1, i_f2v_1, embed_dim,
                                cuda = device, is_user = False)
    i_embed_1 = encoder(i_f2v_1, embed_dim, i_adj, i_adj_rating, i_tmp_embed_1,
                                cuda = device, is_user = False)
    i_tmp_embed_2 = aggregator(u_f2v_2, i_f2v_2, embed_dim,
                                cuda = device, is_user = True)
    i_embed_2 = encoder(i_f2v_2, embed_dim, i_adj, i_adj_rating, i_tmp_embed_2,
                                cuda = device, is_user = False)
    i_tmp_embed_3 = aggregator(u_f2v_3, i_f2v_3, embed_dim,
                                cuda = device, is_user = True)
    i_embed_1 = encoder(i_f2v_3, embed_dim, i_adj, i_adj_rating, i_tmp_embed_3,
                                cuda = device, is_user = False)

    i_final_embed = simpleAttention(i_embed_1, i_embed_2, i_embed_3, device)

dataset = args['<dataset>']
checkpt_file = 'pre_trained/{}/{}_allMP_multi.ckpt'.format(dataset, dataset)
print('model: {}'.format(checkpt_file))

# training params
batch_size = int(args['--batch_size'])
epochs = int(args['--epochs'])
patience = int(args['--patience']])
lr = float(args['--learn_rate'])  # learning rate
hid_units = int(args['--num_of_hidden_units'])  # numbers of hidden units
n_components = int(args['--num_of_components'])
n_mlp_units = int(args['--num_of_mlp_units'])
residual = False
nonlinearity = tf.nn.elu
model = wrjGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('----- Archi. hyperparams -----')
print('nb. hidden units: ' + str(hid_units))
print('nb. latent preference components: ' + str(n_components))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))


import numpy as np
import scipy.io as sio
from utils import process
from models import wrjGAT_multi


def sample_mask(idx, shape):
    """Create mask."""
    mask = np.zeros(shape)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(path = args['<dataset_dir>']):
    data = sio.loadmat(path + dataset + ".mat")
    rating, ufeature, ifeature = data['rating'], data['ufeature'], data['ifeature']
    network = data['adj']

    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx,rating[0][1])
    val_mask = sample_mask(val_idx,rating[0][1])
    test_mask = sample_mask(test_idx,rating[0][1])

    train = np.zeros(rating.shape)
    val = np.zeros(rating.shape)
    test = np.zeros(rating.shape)
    train[train_mask, :] = rating[train_mask, :]
    val[val_mask, :] = rating[val_mask, :]
    test[test_mask, :] = rating[test_mask, :]

    print('train:{}, val:{}, test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(train.shape,
                                                                                    val.shape,
                                                                                    test.shape,
                                                                                    train_idx.shape,
                                                                                    val_idx.shape,
                                                                                    test_idx.shape))

    feature_list = [ufeature, ifeature]
    return network, feature_list, train, val, test, train_mask, val_mask, test_mask

adj, fea_list, train, val, test, train_mask, val_mask, test_mask = load_data()

nb_u = fea_list[0].shape[0]
u_fea_size = fea_list[0].shape[1]
nb_i = fea_list[1].shape[0]
i_fea_size = fea_list[1].shape[1]

fea_list = [fea[np.newaxis] for fea in fea_list]
adj = adj[np.newaxis]
train = train[np.newaxis]
val = val[np.newaxis]
test = test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

bias = process.adj_to_bias(adj)

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ufeature = [tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, nb_u, u_fea_size),
                                    name='ufeature']
        ifeature = [tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, nb_i, i_fea_size),
                                    name='ifeature']
        _bias = [tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, nb_u, nb_i),
                                    name='bias']
        rating_in = tf.placeholder(dtype=tf.int32,
                                    shape=(batch_size, nb_u, nb_i),
                                    name='rating_in')
        msk_in = tf.placeholder(dtype=tf.int32,
                                    shape=(batch_size, nb_u, nb_i),
                                    name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32,
                                    shape=(),
                                    name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32,
                                    shape=(),
                                    name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool,
                                    shape=(),
                                    name='is_train')

    # forward
    u_final_embedding, i_final_embedding = model.inference(ufeature, ifeature, attn_drop, ffd_drop,
                                                       bias=_bias, hid_units=hid_units, n_components=n_components,
                                                       activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0

            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:
                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step))

            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, _final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        print('start knn, kmean.....')
        xx = np.expand_dims(_final_embedding, axis=0)[test_mask]

        from numpy import linalg as LA

        # xx = xx / LA.norm(xx, axis=1)
        yy = y_test[test_mask]

        print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        from exps import my_KNN, my_Kmeans, my_TSNE, my_Linear

        my_KNN(xx, yy)
        my_Kmeans(xx, yy)

        sess.close()
