import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class wrjGAT_multi(BaseGAttN):
    def inference(ufeature, ifeature, attn_drop, ffd_drop, bias, hid_units,
                n_components, activation=tf.nn.elu, residual=False, att_size=128):
        u_embed_list = []
        i_embed_list = []

        u_attns = []
        for _ in range(n_components):
            u_attns.append(layers.attn_component(ufeature, ifeature, out_sz=hid_units,
                                                bias_mat=bias, activation=activation,
                                                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                is_user=True))
            u_h_1 = tf.concat(u_attns, axis=-1)
            u_embed_list.append(tf.expand_dims(tf.squeeze(u_h_1), axis=1))

        u_multi_embed = tf.concat(u_embed_list, axis=1)
        u_final_embed, u_att_val = layers.SimpleAttLayer(u_multi_embed, att_size,
                                                     time_major=False,
                                                     return_beta=True)

        i_attns = []
        for _ in range(n_components):
            i_attns.append(layers.attn_component(ufeature, ifeature, out_sz=hid_units,
                                                bias_mat=tf.transpose(bias, [1, 0, 2]), activation=activation,
                                                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                is_user=False))
            i_h_1 = tf.concat(u_attns, axis=-1)
            i_embed_list.append(tf.expand_dims(tf.squeeze(i_h_1), axis=1))

        i_multi_embed = tf.concat(i_embed_list, axis=1)
        i_final_embed, i_att_val = layers.SimpleAttLayer(i_multi_embed, att_size,
                                                     time_major=False,
                                                     return_beta=True)

        return u_final_embed, u_att_val, i_final_embed, i_att_val
