# coding: utf-8
import os
import tensorflow as tf
import docopt


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

args = docopt("""
    Usage:
        run.py [options] <dataset_dir> <dataset>

    Options:
        --batch_size NUM                [default: 1]
        --epochs NUM                    [default: 200]
        --patience NUM                  [default: 5]
        --learn_rate NUM                [default: 0.00005]
        --num_of_hidden_units NUM       [default: 128]
        --num_of_components NUM         [default: 3]
    """)

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
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('----- Archi. hyperparams -----')
print('nb. units per layer: ' + str(hid_units))
print('nb. preference components: ' + str(n_components))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))


import numpy as np
import scipy.io as sio
from utils import process


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
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)
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
