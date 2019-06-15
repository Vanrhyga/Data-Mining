import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d


def attn_component(ufeature, ifeature, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0,
                return_coef=False, is_user=True):
    """[summary]
    [description]
    Arguments:
        ufeature {[type]} -- shape=(batch_size, nb_u, fea_size))
        ifeature {[type]} -- shape=(batch_size, nb_i, fea_size))
    """
    if is_user:
        with tf.name_scope('u_attn'):
            if in_drop != 0.0:
                ifeature = tf.nn.dropout(ifeature, 1.0 - in_drop)

            item_fts = tf.layers.conv1d(ifeature, out_sz, 1, use_bias=False)
            f_1 = tf.layers.conv1d(item_fts, 1, 1)

            user_fts = tf.layers.conv1d(ufeature, out_sz, 1, use_bias=False)
            f_2 = tf.layers.conv1d(user_fts, 1, 1)

            logits = f_2 + tf.transpose(f_1, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

            if in_drop != 0.0:
                item_fts = tf.nn.dropout(item_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, item_fts)
            ret = tf.contrib.layers.bias_add(vals)

    else:
        with tf.name_scope('i_attn'):
            if in_drop != 0.0:
                ufeature = tf.nn.dropout(ufeature, 1.0 - in_drop)

            user_fts = tf.layers.conv1d(ufeature, out_sz, 1, use_bias=False)
            f_1 = tf.layers.conv1d(user_fts, 1, 1)

            item_fts = tf.layers.conv1d(ifeature, out_sz, 1, use_bias=False)
            f_2 = tf.layers.conv1d(item_fts, 1, 1)

            logits = f_2 + tf.transpose(f_1, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

            if in_drop != 0.0:
                item_fts = tf.nn.dropout(user_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, user_fts)
            ret = tf.contrib.layers.bias_add(vals)

    if return_coef:
        return activation(ret), coefs
    else:
        return activation(ret)  # activation

def SimpleAttLayer(inputs, attention_size, time_major=False, return_beta=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    beta = tf.nn.softmax(vu, name='beta')         # (B,T) shape

    # Output is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(beta, -1), 1)

    if not return_beta:
        return output
    else:
        return output, beta
