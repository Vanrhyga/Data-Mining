# coding: utf-8
from docopt import docopt
import numpy as np
import tensorflow as tf
from datetime import datetime
import math
import heapq
import pandas as pd
from sklearn.utils import shuffle



args = docopt("""
    Usage:
        NeuACF++.py [options] <dataset_dir> <dataset> 

    Options:
        --mat mat_list			[default: ""]
        --epochs NUM			[default: 40]
        --last_layer_size NUM	[default: 32]
        --num_of_layers NUM		[default: 2]
        --num_of_neg NUM		[default: 10]
        --learn_rate NUM		[default: 0.00005]
        --batch_size NUM		[default: 1024]
        --margin NUM            [default: 1.0]
        --mat_select STR		[default: median]
        --merge STR				[default: attention]
    """)


dataset_dir = args['<dataset_dir>']    
dataset = args['<dataset>']

mat = args['--mat']
mat_list = mat.rsplit(sep=',')
mat_select = args['--mat_select']
merge = args['--merge']

epochs = int( args['--epochs'] )
last_layer_size = int( args['--last_layer_size'] )
num_of_layers = int( args['--num_of_layers'] )
num_negs = int( args['--num_of_neg'] )
learn_rate = float( args['--learn_rate'] )
batch_size = int( args['--batch_size'] )
margin = float( args['--margin'] )


for key in args:
    print( "#", key, args[key] )


train_ratings_sparse_dir = dataset_dir + dataset +".train.rating"
test_negative_dir = dataset_dir + dataset +".test.negative"

ob_train_dir =  dataset_dir  + "U.feature"

U_feature_dir = dataset_dir +mat_list[0]+".pathsim.feature." + mat_select
I_feature_dir = dataset_dir +mat_list[1]+".pathsim.feature." + mat_select

if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
	U_feature_dir2 = dataset_dir +mat_list[2]+".pathsim.feature." + mat_select
	I_feature_dir2 = dataset_dir +mat_list[3]+".pathsim.feature."+ mat_select

if( len(mat_list) == 6 or len(mat_list) == 8 ):
	U_feature_dir3 = dataset_dir +mat_list[4]+".pathsim.feature."+ mat_select
	I_feature_dir3 = dataset_dir +mat_list[5]+".pathsim.feature."+ mat_select

if( len(mat_list) == 8 ):
	U_feature_dir4 = dataset_dir +mat_list[6]+".pathsim.feature."+ mat_select
	I_feature_dir4 = dataset_dir +mat_list[7]+".pathsim.feature."+ mat_select


train_ratings_sparse = pd.read_csv( train_ratings_sparse_dir ,sep="\t", header=None ).as_matrix()

testset = pd.read_csv( test_negative_dir, header=None, sep="\t",converters={0: eval} )
testset = testset.set_index( 0 )
test_instances = testset.T.to_dict(orient='list')

ob_train = pd.read_csv( ob_train_dir, sep=",", header=None ).fillna( 0 ).as_matrix()

U_feature = pd.read_csv( U_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# U_feature1 shape:", U_feature.shape  )
I_feature = pd.read_csv( I_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# I_feature1 shape:", I_feature.shape  )
I_feature_num = I_feature.shape[1]
U_feature_num = U_feature.shape[1]

if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
	U_feature2 = pd.read_csv( U_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# U_feature2 shape:", U_feature2.shape  )
	I_feature2 = pd.read_csv( I_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# I_feature2 shape:", I_feature2.shape  )
	I_feature_num2 = I_feature2.shape[1]
	U_feature_num2 = U_feature2.shape[1]

if(len(mat_list) == 6 or len(mat_list) == 8):
	U_feature3 = pd.read_csv( U_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# U_feature3 shape:", U_feature3.shape  )
	I_feature3 = pd.read_csv( I_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# I_feature3 shape:", I_feature3.shape  )
	I_feature_num3 = I_feature3.shape[1]
	U_feature_num3 = U_feature3.shape[1]

if(len(mat_list) == 8):
	U_feature4 = pd.read_csv( U_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# U_feature4 shape:", U_feature4.shape  )
	I_feature4 = pd.read_csv( I_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
	print( "# I_feature4 shape:", I_feature4.shape  )
	I_feature_num4 = I_feature4.shape[1]
	U_feature_num4 = U_feature4.shape[1]


I_num = I_feature.shape[0]
U_num = U_feature.shape[0]


def get_train_instances( train_ratings_sparse, num_negs, ob_matrix = ob_train):
    np.random.seed( 2019 )
    pos_u, pos_i, neg_u, neg_i = [],[],[],[]

    num_users = len( np.unique( train_ratings_sparse[:,0] ) )
    num_items = len( np.unique( train_ratings_sparse[:,1] ) )
    
    train_ratings_sparse = train_ratings_sparse[:,0:2]
    
    for (u, i) in train_ratings_sparse:
        u = int(u)
        i = int(i)

        # negative instances
        for t in range(num_negs):
            j = np.random.randint(num_items)
            while ob_train[u][j] != 0:
                j = np.random.randint(num_items)
            pos_u.append( [u] )
            pos_i.append( [i] )
            neg_u.append( [u] )
            neg_i.append( [j] )
    
    return pos_u, pos_i, neg_u, neg_i


print( "# Starting Negative Sample..." )
pos_u, pos_i, neg_u, neg_i = get_train_instances( train_ratings_sparse, num_negs)
print( "# All Training Instances:", len( pos_u ) )
print( "# Negative Sample Done." )


def HIN_MODEL(name,U_embedding, I_embedding, U_feature_num, I_feature_num, hidden_size):
    u_w1 = tf.get_variable(str(name)+"_u_w1", shape=(U_feature_num, hidden_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b1 = tf.get_variable(str(name)+"_u_b1", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w2 = tf.get_variable(str(name)+"_u_w2", shape=(hidden_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b2 = tf.get_variable(str(name)+"_u_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w3 = tf.get_variable(str(name)+"_u_w3", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b3 = tf.get_variable(str(name)+"_u_b3", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w4 = tf.get_variable(str(name)+"_u_w4", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b4 = tf.get_variable(str(name)+"_u_b4", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w1 = tf.get_variable(str(name)+"_v_w1", shape=(I_feature_num, hidden_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b1 = tf.get_variable(str(name)+"_v_b1", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w2 = tf.get_variable(str(name)+"_v_w2", shape=(hidden_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b2 = tf.get_variable(str(name)+"_v_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w3 = tf.get_variable(str(name)+"_v_w3", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b3 = tf.get_variable(str(name)+"_v_b3", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w4 = tf.get_variable(str(name)+"_v_w4", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b4 = tf.get_variable(str(name)+"_v_b4", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    
    net_u_1 = tf.nn.relu( tf.matmul(U_embedding, u_w1) + u_b1 )
    net_v_1 = tf.nn.relu( tf.matmul(I_embedding, v_w1) + v_b1 )
    
    if( num_of_layers == 1 ):
        return net_u_1, net_v_1 

    net_u_2 = tf.matmul(net_u_1, u_w2) + u_b2 
    net_v_2 = tf.matmul(net_v_1, v_w2) + v_b2 

    if( num_of_layers == 2 ):
        return net_u_2, net_v_2 

    net_u_3 = tf.matmul(net_u_2, u_w3) + u_b3
    net_v_3 = tf.matmul(net_v_2, v_w3) + v_b3

    if( num_of_layers == 3 ):
        return net_u_3, net_v_3

    net_u_4 = tf.matmul(net_u_3, u_w4) + u_b4
    net_v_4 = tf.matmul(net_v_3, v_w4) + v_b4

    return net_u_4, net_v_4


def distance( U, I ):
    distance = tf.reduce_sum((U - I) ** 2, 1, keep_dims=True)
    return distance


def get_embedding_matrix( name, U_num, U_feature_num,U_feature, I_num ,I_feature_num, I_feature):
    U_embedding_matrix = tf.get_variable(name+"embeddings_u", 
                                     shape=[U_num, U_feature_num], 
                                     initializer=tf.constant_initializer(np.array(U_feature)),
                                     trainable=False)
                             
    I_embedding_matrix = tf.get_variable(name+"embeddings_i", 
                                     shape=[I_num, I_feature_num], 
                                     initializer=tf.constant_initializer(np.array(I_feature)),
                                     trainable=False)
    
    return U_embedding_matrix, I_embedding_matrix


tf.reset_default_graph()

U_feature_input = tf.placeholder(tf.int32, [None,1])
I_feature_input = tf.placeholder(tf.int32, [None,1])
pos_u_input = tf.placeholder(tf.int32, [None,1])
pos_i_input = tf.placeholder(tf.int32, [None,1])
neg_u_input = tf.placeholder(tf.int32, [None,1])
neg_i_input = tf.placeholder(tf.int32, [None,1])

U_embedding_matrix,I_embedding_matrix =  get_embedding_matrix( "1", U_num, U_feature_num,U_feature, I_num ,I_feature_num, I_feature  )

if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
    U_embedding_matrix2,I_embedding_matrix2 =  get_embedding_matrix( "2", U_num, U_feature_num2,U_feature2, I_num ,I_feature_num2, I_feature2  )

if(len(mat_list) == 6 or len(mat_list) == 8):
    U_embedding_matrix3,I_embedding_matrix3 =  get_embedding_matrix( "3", U_num, U_feature_num3,U_feature3, I_num ,I_feature_num3, I_feature3  )

if(len(mat_list) == 8):
    U_embedding_matrix4,I_embedding_matrix4 =  get_embedding_matrix( "4", U_num, U_feature_num4,U_feature4, I_num ,I_feature_num4, I_feature4  )

U_embedding = tf.nn.embedding_lookup(U_embedding_matrix, U_feature_input) 
I_embedding = tf.nn.embedding_lookup(I_embedding_matrix, I_feature_input)  
U_embedding = tf.reshape(U_embedding, [-1,U_feature_num])
I_embedding = tf.reshape(I_embedding, [-1,I_feature_num])

if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
    U_embedding2 = tf.nn.embedding_lookup(U_embedding_matrix2, U_feature_input) 
    I_embedding2 = tf.nn.embedding_lookup(I_embedding_matrix2, I_feature_input)  
    U_embedding2 = tf.reshape(U_embedding2, [-1,U_feature_num2])
    I_embedding2 = tf.reshape(I_embedding2, [-1,I_feature_num2])

if(len(mat_list) == 6 or len(mat_list) == 8):
    U_embedding3 = tf.nn.embedding_lookup(U_embedding_matrix3, U_feature_input) 
    I_embedding3 = tf.nn.embedding_lookup(I_embedding_matrix3, I_feature_input)  
    U_embedding3 = tf.reshape(U_embedding3, [-1,U_feature_num3])
    I_embedding3 = tf.reshape(I_embedding3, [-1,I_feature_num3])

if(len(mat_list) == 8):
    U_embedding4 = tf.nn.embedding_lookup(U_embedding_matrix4, U_feature_input) 
    I_embedding4 = tf.nn.embedding_lookup(I_embedding_matrix4, I_feature_input)  
    U_embedding4 = tf.reshape(U_embedding4, [-1,U_feature_num4])
    I_embedding4 = tf.reshape(I_embedding4, [-1,I_feature_num4])


U1, I1 = HIN_MODEL("A1", U_embedding, I_embedding, U_feature_num, I_feature_num, 600)

if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
    U2, I2 = HIN_MODEL("A2", U_embedding2, I_embedding2, U_feature_num2, I_feature_num2,600)

if(len(mat_list) == 6 or len(mat_list) == 8):
    U3, I3 = HIN_MODEL("A3", U_embedding3, I_embedding3, U_feature_num3, I_feature_num3,600)

if(len(mat_list) == 8):
    U4, I4 = HIN_MODEL("A4", U_embedding4, I_embedding4, U_feature_num4, I_feature_num4,600)


def attention(name, input_vec):
    with tf.variable_scope( 'V', reuse = tf.AUTO_REUSE):
        att_w1 = tf.get_variable( name+"att_w1", shape=(last_layer_size, 64 ), initializer=tf.contrib.layers.xavier_initializer(), )
        att_b1 = tf.get_variable( name+"att_b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())

        att_w2 = tf.get_variable( name+"att_w2", shape=(64,1), initializer=tf.contrib.layers.xavier_initializer())
        att_b2 = tf.get_variable( name+"att_b2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
            
        net_1 = tf.nn.sigmoid( tf.matmul(input_vec, att_w1) + att_b1 )
        net_2 = tf.nn.sigmoid( tf.matmul(net_1, att_w2) + att_b2 )
            
        return net_2  


if( len(mat_list) == 4 or len(mat_list) == 6 or len(mat_list) == 8 ):
    w1 = tf.exp(  attention( "U", U1 ) )
    w2 = tf.exp(  attention( "U", U2 ) )

if( len(mat_list) == 6 or len(mat_list) == 8):
    w3 = tf.exp(  attention( "U", U3 ) )

if( len(mat_list) == 8):
    w4 = tf.exp(  attention( "U", U4 ) )


#if( len(mat_list) == 2 ):
U = U1
I = I1

if( len(mat_list) == 4 ):
    if merge == "attention":
        U = w1/(w1+w2)*U1 + w2/(w1+w2)*U2
        I = w1/(w1+w2)*I1 + w2/(w1+w2)*I2
    if merge == "avg":
        U = 1/2*U1 + 1/2*U2
        I = 1/2*I1 + 1/2*I2

if( len(mat_list) == 6 ):
    if merge == "attention":
        U = w1/(w1+w2+w3)*U1 + w2/(w1+w2+w3)*U2 + w3/(w1+w2+w3)*U3
        I = w1/(w1+w2+w3)*I1 + w2/(w1+w2+w3)*I2 + w3/(w1+w2+w3)*I3
    if merge == "avg":
        U = 1/3*U1 + 1/3*U2 + 1/3*U3
        I = 1/3*I1 + 1/3*I2 + 1/3*I3

if( len(mat_list) == 8 ):
    if merge == "attention":
        U = w1/(w1+w2+w3+w4)*U1 + w2/(w1+w2+w3+w4)*U2 + w3/(w1+w2+w3+w4)*U3 + w4/(w1+w2+w3+w4)*U4
        I = w1/(w1+w2+w3+w4)*I1 + w2/(w1+w2+w3+w4)*I2 + w3/(w1+w2+w3+w4)*I3 + w4/(w1+w2+w3+w4)*I4
    if merge == "avg":
        U = 1/4*U1 + 1/4*U2 + 1/4*U3 + 1/4*U4
        I = 1/4*I1 + 1/4*I2 + 1/4*I3 + 1/4*I4


pred_val = distance( U, I )


pos_u_embedding = tf.nn.embedding_lookup( U, pos_u_input)
pos_i_embedding = tf.nn.embedding_lookup( I, pos_i_input)
neg_u_embedding = tf.nn.embedding_lookup( U, neg_u_input)
neg_i_embedding = tf.nn.embedding_lookup( I, neg_i_input)

pos = tf.reduce_sum((pos_u_embedding - pos_i_embedding) ** 2, 1, keep_dims = True)
neg = tf.reduce_sum((neg_u_embedding - neg_i_embedding) ** 2, 1, keep_dims = True)

loss_all = tf.reduce_mean(tf.maximum(pos - neg + margin, 0))

train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss_all)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


final_ndcg_list_5 = []
final_hr_list_5 = []
final_ndcg_list_10 = []
final_hr_list_10 = []
final_ndcg_list_15 = []
final_hr_list_15 = []
final_ndcg_list_20 = []
final_hr_list_20 = []


for epoch in range( epochs ): 
    print( epoch )
    one_epoch_loss = 0.0
    one_epoch_batchnum = 0.0
    print( "# Start Training...." )

    pos_u, pos_i, neg_u, neg_i = shuffle(pos_u, pos_i, neg_u, neg_i)

    for index in range( len(pos_u) // batch_size + 1 ):
        batch_pos_u =  pos_u[index * batch_size :(index + 1) * batch_size]
        batch_pos_i =  pos_i[index * batch_size :(index + 1) * batch_size]
        batch_neg_u =  neg_u[index * batch_size :(index + 1) * batch_size]
        batch_neg_i =  neg_i[index * batch_size :(index + 1) * batch_size]

        batch_u = batch_pos_u + batch_neg_u
        batch_i = batch_pos_i + batch_neg_i

        if(len(batch_pos_u) > 0):
            if(len(mat_list) == 8):
                _, loss_val,w1_ob,w2_ob,w3_ob,w4_ob = sess.run(
	                [train_step, loss_all,w1,w2,w3,w4],
	                feed_dict={U_feature_input: batch_u, I_feature_input: batch_i, 
	                pos_u_input: batch_pos_u, pos_i_input: batch_pos_i, neg_u_input: batch_neg_u, neg_i_input: batch_neg_i})

            if(len(mat_list) == 6):
	            _, loss_val,w1_ob,w2_ob,w3_ob = sess.run(
	                [train_step, loss_all,w1,w2,w3],
	                feed_dict={U_feature_input: batch_u, I_feature_input: batch_i, 
	                pos_u_input: batch_pos_u, pos_i_input: batch_pos_i, neg_u_input: batch_neg_u, neg_i_input: batch_neg_i})
	        
            if(len(mat_list) == 4):
	            _, loss_val,w1_ob,w2_ob = sess.run(
	                [train_step, loss_all,w1,w2],
	                feed_dict={U_feature_input: batch_u, I_feature_input: batch_i, 
	                pos_u_input: batch_pos_u, pos_i_input: batch_pos_i, neg_u_input: batch_neg_u, neg_i_input: batch_neg_i})

            if(len(mat_list) == 2):
	             _, loss_val = sess.run(
	                [train_step, loss_all],
	                feed_dict={U_feature_input: batch_u, I_feature_input: batch_i, 
	                pos_u_input: batch_pos_u, pos_i_input: batch_pos_i, neg_u_input: batch_neg_u, neg_i_input: batch_neg_i})

            one_epoch_loss += loss_val
            one_epoch_batchnum += 1.0

        if index % 100 == 0:
            format_str = '# %s: Progress %.2f %%, Loss = %.4f'
            print (format_str % ( datetime.now(), index /( len(pos_u) // batch_size ) * 100 , one_epoch_loss / (index+1) ) )

        if index == len(pos_u) // batch_size:
                format_str = '# ****%s: %d epoch, iteration averge loss = %.4f '
                print (format_str % (datetime.now(), epoch, one_epoch_loss / one_epoch_batchnum))

                hr_list_5 = []
                ndcg_list_5 = []

                hr_list_10 = []
                ndcg_list_10 = []
                
                hr_list_15 = []
                ndcg_list_15 = []
                
                hr_list_20 = []
                ndcg_list_20 = []
                
                for u, i in test_instances:
                    v_random = [i]
                    v_random.extend( test_instances[(u,i)] )
                    batch_u_test = [[u]]*len( v_random )
                    batch_v_test = list( np.reshape( v_random, (-1,1) ) )
                    pred_value = sess.run([pred_val], feed_dict={U_feature_input: batch_u_test, I_feature_input: batch_v_test})
                    pre_real_val = np.array(pred_value).reshape((-1))

                    items = v_random
                    gtItem = i

                    # Get prediction scores
                    map_item_score = {}
                    for i in range(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    # Evaluate top rank list
                    ranklist = heapq.nsmallest(20, map_item_score, key=map_item_score.get)
            
                    hr_list_5.append(getHitRatio(ranklist[:5], gtItem))
                    ndcg_list_5.append(getNDCG(ranklist[:5], gtItem))
            
                    hr_list_10.append(getHitRatio(ranklist[:10], gtItem))
                    ndcg_list_10.append(getNDCG(ranklist[:10], gtItem))
                    
                    hr_list_15.append(getHitRatio(ranklist[:15], gtItem))
                    ndcg_list_15.append(getNDCG(ranklist[:15], gtItem))
                    
                    hr_list_20.append(getHitRatio(ranklist[:20], gtItem))
                    ndcg_list_20.append(getNDCG(ranklist[:20], gtItem))

                final_hr_list_5.append( np.array(hr_list_5).mean() )
                final_ndcg_list_5.append( np.array(ndcg_list_5).mean() )
                
                final_hr_list_10.append( np.array(hr_list_10).mean() )
                final_ndcg_list_10.append( np.array(ndcg_list_10).mean() )
                
                final_hr_list_15.append( np.array(hr_list_15).mean() )
                final_ndcg_list_15.append( np.array(ndcg_list_15).mean() )
                
                final_hr_list_20.append( np.array(hr_list_20).mean() )
                final_ndcg_list_20.append( np.array(ndcg_list_20).mean() )
                
                
                print( "***Result HR@5,:NDCG@5: ", final_hr_list_5[-1], final_ndcg_list_5[-1] )
                print( "***Result HR@10,:NDCG@10: ", final_hr_list_10[-1], final_ndcg_list_10[-1] )
                print( "***Result HR@15,:NDCG@15: ", final_hr_list_15[-1], final_ndcg_list_15[-1] )
                print( "***Result HR@20,:NDCG@20: ", final_hr_list_20[-1], final_ndcg_list_20[-1] )