import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import os
import time


#ARD kernel
MATRIX_JITTER = 1e-3
NN_MAX = 20.0
FLOAT_TYPE = tf.float32
DELTA_JITTER= 1e-2

def kernel_cross_tf(tf_Xm, tf_Xn, tf_log_amp, tf_log_lengthscale, return_raw_K = False):
    '''
    :param tf_Xm: [m, d]
    :param tf_Xn: [n,d]
    :param tf_log_amp:
    :param tf_log_lengthscale:
    :return: K [ m,n]
    '''

    '''
    tf_Xm = tf.matmul(tf_Xm, tf.linalg.tensor_diag(1.0 / tf.exp(0.5 * tf.reshape(tf_log_lengthscale, [-1]))))
    tf_Xn = tf.matmul(tf_Xn, tf.linalg.tensor_diag(1.0 / tf.exp(0.5 * tf.reshape(tf_log_lengthscale, [-1]))))
    col_norm1 = tf.reshape(tf.reduce_sum(tf_Xm * tf_Xm, 1), [-1, 1])
    col_norm2 = tf.reshape(tf.reduce_sum(tf_Xn * tf_Xn, 1), [-1, 1])
    K = col_norm1 - 2.0 * tf.matmul(tf_Xm, tf.transpose(tf_Xn)) + tf.transpose(col_norm2)
    if return_raw_K:
        return tf.exp( -0.5 * K + tf_log_amp), K
    else:
        return tf.exp( -0.5 * K + tf_log_amp)
    '''
    lengthscale = 1.0 / tf.exp(tf_log_lengthscale)
    X = tf.expand_dims(tf_Xm, 1)
    Y = tf.expand_dims(tf_Xn, 0)
    K = (X - Y) *(X - Y) * tf.reshape( lengthscale, [-1])
    K = tf.reduce_sum(K, axis=-1)
    K = tf.exp(-0.5 * K + tf_log_amp)
    return K

def KL_q_p_tf( Kmm, Sig, Ltril, mu, k):
    '''
    return KL( q(alpha) || p( alpha))
    :param Kmm:
    :param Kmm_inv:
    :param Sig: Ltril@Ltril.T
    :param Ltril:
    :param mu: [ length, 1]
    :return:
    '''

    #KL = 0.5 * tf.linalg.trace( Kmm_inv @ ( Sig + mu@tf.transpose( mu)))
    KL = 0.5 * tf.linalg.trace( tf.linalg.solve( Kmm, Sig + mu@tf.transpose( mu) ))
    KL = KL  - k * 0.5  + 0.5 * tf.linalg.logdet( Kmm) - tf.reduce_sum( tf.log(  tf.abs( tf.linalg.diag_part( Ltril))))
    return KL

def sample_pst_f_tf(mu_alpha, Ltril_alpha,Kmm, Knm, log_amp, jitter, return_alpha = False ):
    '''

    :param mu_alpha:
    :param Ltril_alpha:
    :param Kmm:
    :param Knm:
    :param log_amp:
    :param jitter:
    :param return_alpha:
    :return:
    '''
    #sample alpha
    z = tf.random.normal( mu_alpha.shape, dtype=FLOAT_TYPE)
    alpha = mu_alpha + Ltril_alpha @ z

    z_f = tf.random.normal([Knm.shape[0].value, 1], dtype=FLOAT_TYPE)
    stdev = tf.sqrt( tf.exp( log_amp) + jitter - tf.reduce_sum( Knm * tf.transpose( tf.linalg.solve( Kmm, tf.transpose(Knm))), axis=1, keepdims=True ))
    f = Knm @ tf.linalg.solve(  Kmm,alpha)  + stdev*z_f

    if return_alpha:
        return f,alpha
    else:
        return f

def sample_pst_f_tf_MLE(mu_alpha,Kmm, Knm ):
    '''

    :param mu_alpha:
    :param Ltril_alpha:
    :param Kmm:
    :param Knm:
    :param log_amp:
    :param jitter:
    :param return_alpha:
    :return:
    '''
    #sample alpha
    alpha = mu_alpha
    f = Knm @ tf.linalg.solve(  Kmm,alpha)
    return f


def concat_embeddings( U, ind):
    '''
    get the concatenated embeddings
    :param U: list of embeddings [ nmod, num_item, rank]
    :param ind: index to embedings [ batch, nmod]
    :return:
    '''
    nmod = len(U)
    X = tf.concat([tf.gather(U[k], ind[:, k]) for k in range(nmod)], 1)
    return X

def log_CP_base_rate( U, ind):
    nmod = len(U)
    components = [tf.gather(U[k], ind[:, k]) for k in range(nmod)]
    cp = tf.reduce_prod( components, axis=0)
    
    base_rate = tf.reduce_sum( tf.log(tf.square(cp)+1e-6), axis=1, keepdims=True)

    return base_rate



def assemble_NN_input( X, U, t,T0, T, len_X):
    ave_U = [ tf.reduce_mean( U_i,axis=0, keep_dims=True) for U_i in U]
    ave_U_concat =tf.concat(  ave_U, axis=1) #[1, nmod * rank]
    ave_U_concat_tiled = tf.tile( ave_U_concat,[ len_X,1])

    input_tensor = tf.concat( [ X,( t - T0)/T,ave_U_concat_tiled], axis=1)
    #input_tensor = tf.concat( [ X,( t - T0)/T], axis=1)
    return input_tensor

def assemble_NN_input_v2( X_outer, X_inner, valid_delay, event_delay, T):
    valid_delay = tf.expand_dims( valid_delay, axis=2)
    event_delay = tf.expand_dims( event_delay, axis=2) / T #normlized T
    X_inner = tf.expand_dims( X_inner, axis=0)
    masked_X = X_inner * valid_delay
    concat_time = tf.concat( [ masked_X, event_delay], axis=2)
    input_tensor = tf.reduce_mean( concat_time, axis=1)
    #input_tensor = tf.reshape( concat_time,[ X_outer.shape[0].value, -1])

    print("Assemble NN input V2:" , "input tensor shape = ", input_tensor.shape)

    return input_tensor

def assemble_time_decay_GP_input( Xi, Xn, len_X, len_N):
    Xi = tf.expand_dims( Xi, 1)
    Xn = tf.expand_dims( Xn, 0)

    Xi = tf.tile( Xi, [1, len_N,1])
    Xn = tf.tile( Xn, [len_X, 1,1])
    input_tensor = tf.concat( [ Xi,Xn] ,axis=2)
    return input_tensor


class DataGenerator:
    def __init__(self, X, y=None, shuffle = True):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        #self.repeat = repeat

        self.num_elems = len(X)
        self.curr_idx = 0

        if self.shuffle:
            self.random_idx = np.random.permutation(self.num_elems)
        else:
            self.random_idx = np.arange( self.num_elems)


    def draw_last(self, return_idx = False):
        '''
        draw last batch sample
        :return:
        '''
        if self.y is not None:
            if return_idx:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx], self.y[self.last_arg_idx]
        else:
            if return_idx:
                return self.X[self.last_arg_idx], self.last_arg_idx
            else:
                return self.X[self.last_arg_idx]


    def draw_next(self, batch_size, return_idx = False):
        if batch_size > self.num_elems:
            raise NameError("Illegal batch size")

        if batch_size + self.curr_idx > self.num_elems:
            # shuffle

            if self.shuffle:
                self.random_idx = np.random.permutation(self.num_elems)
            else:
                self.random_idx = np.arange(self.num_elems)
            self.curr_idx = 0

        arg_idx = self.random_idx[self.curr_idx: self.curr_idx + batch_size]
        self.last_arg_idx = arg_idx

        self.curr_idx += batch_size

        if self.y is not None:
            if return_idx:
                return self.X[arg_idx], self.y[arg_idx], arg_idx
            else:
                return self.X[arg_idx], self.y[arg_idx]
        else:
            if return_idx:
                return self.X[arg_idx], arg_idx
            else:
                return self.X[arg_idx]


def get_end_time( train_y, start_idx, window_size):
    '''
    :param train_y:
    :param start_idx: [batch_size,]
    :param window_size: scalar
    :return:
    '''
    upper_bound = len( train_y) - 1
    upper_idx = start_idx + window_size
    upper_idx = np.minimum(  upper_idx, upper_bound )
    end_y = train_y[ upper_idx]

    return end_y

def generate_inner_event_y( outter_idx, train_ind, train_y, window_size):
    lower_idx = outter_idx - window_size

    ind_acc = []
    y_acc = []

    for i in range( len( outter_idx)):
        li = lower_idx[i]
        ri = outter_idx[i]
        if li >=0:
            idxes = np.arange( li, ri)
        else:
            idxes = list( range( 0, ri))
            idxes = [-1] * ( -li) + idxes
            idxes = np.array( idxes)

        ind = np.expand_dims ( train_ind[ idxes],0) #[1, 300, nmod]
        y = train_y[ idxes].reshape( ( 1,-1)) #[ 1, 300]

        ind_acc.append( ind)
        y_acc.append( y)

    ind_acc =  np.vstack( ind_acc)
    y_acc = np.vstack( y_acc)

    return ind_acc, y_acc



def init_base_gp_pseudo_inputs(U, ind, pseudo_num):
    nmod = ind.shape[1]
    uniq_inds = np.unique( ind, axis=0)
    part = [U[k][uniq_inds[:,k],:] for k in range(nmod)]
    X = np.hstack(part)
    kmeans = KMeans(n_clusters=pseudo_num, random_state=0, n_jobs= 7,).fit(X)
    return kmeans.cluster_centers_

def init_decay_gp_pseudo_inputs(U, ind, pseudo_num):
    nmod = ind.shape[1]
    uniq_inds = np.unique( ind, axis=0)

    N = len( uniq_inds)

    random_arg = np.random.choice( range( N),pseudo_num)
    sub_inds_1 = uniq_inds[ random_arg]

    random_arg = np.random.choice(range(N), pseudo_num)
    sub_inds_2 = uniq_inds[ random_arg]

    X1 = np.hstack( [U[k][sub_inds_1[:,k],:] for k in range(nmod)])
    X2 = np.hstack( [U[k][sub_inds_2[:,k],:] for k in range(nmod)])

    #Cartisian Product
    X1 = np.tile( X1, [ pseudo_num,1])
    X2 = np.repeat( X2, pseudo_num, axis=0)

    X = np.hstack( [X1,X2])
    kmeans = KMeans(n_clusters=pseudo_num, random_state=0, n_jobs= 7,).fit(X)
    return kmeans.cluster_centers_


#Loading data set
def load_dataSet( data_name, path_dataSet_folder = "../Data"):
    valid_names = {'911', 'article', 'slc', 'chicago'}

    if data_name not in valid_names:
        raise NameError("No such data set: %s. valid data set: %s" % ( data_name, str( valid_names)))

    if data_name == '911':
        ind = np.load( os.path.join(path_dataSet_folder, '911_60k_ind.npy'))
        y = np.load( os.path.join( path_dataSet_folder,  '911_60k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'article':
        ind = np.load(  os.path.join(path_dataSet_folder, 'article_70k_inds.npy'))
        y = np.load(  os.path.join(path_dataSet_folder, 'article_70k_ys.npy')).reshape(-1, 1)

        hours = 3600
        y = y / hours

        NUM_TRAIN = 50000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return (ind, y), (train_ind, train_y), (test_ind, test_y)

    if data_name =='slc':
        ind = np.load( os.path.join(path_dataSet_folder, 'SLC_60k_inds.npy'))
        y = np.load( os.path.join(path_dataSet_folder, 'SLC_60k_y_in_day.npy')).reshape(-1, 1).astype(np.float32)

        NUM_TRAIN = 40000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return (ind, y), (train_ind, train_y), (test_ind, test_y)

    if data_name == 'chicago':
        ind = np.load( os.path.join(path_dataSet_folder, 'CHICAGO_262k_inds.npy'))
        y = np.load( os.path.join(path_dataSet_folder, 'CHICAGO_262k_y_in_day.npy')).reshape(-1, 1).astype(np.float32)

        NUM_TRAIN = 200000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return (ind, y), (train_ind, train_y), (test_ind, test_y)

def log_results( log_file_path, data_name, rank, lr, test_log_llk ):
    with open( log_file_path, 'a') as log_file:
        date = time.asctime()

        log_file.write(date + '\n')
        log_file.write( "data set = %s, rank = %d, lr = %f\n" % ( data_name, rank, lr))
        log_file.write('test_log_llk:\n')
        for log_llk in test_log_llk:
            log_file.write( '%g\n' % log_llk)

        log_file.write('\n')

def extract_event_tensor_Reileigh( ind, y):
    '''
    Precompute Event tensor information for Rayleigh process
    :param ind:
    :param y:
    :return: uniq_inds, n_i: len of seq, sq_sum:sum of square of time diff, log_sum: sum of log time diff
    '''
    if len( ind) != len( y):
        raise NameError('lengths not match')

    N = len(ind)

    event_tensor = {}
    for i in range( N):
        idx = tuple( ind[i])
        if idx in event_tensor:
            event_tensor[idx].append( y[i,0])
        else:
            event_tensor[idx] = [y[i,0]]

    T0, T1 = y[0,0], y[-1,0]

    uniq_inds = [ * event_tensor.keys()]
    uniq_inds = np.array( uniq_inds)
    time_seq = [ * event_tensor.values()]

    if len( uniq_inds) != len( time_seq):
        raise NameError('K V len not match')

    N_seq = len( uniq_inds)

    n_i = []
    sq_sum = []
    log_sum = []
    for i in range( N_seq):
        seq = np.array( time_seq[i])
        n_i.append( len( seq))

        #insert beginning timestamp
        if T0 not in seq:
            seq = np.insert( seq, 0, T0)

        shift_seq = np.roll( seq, -1 )

        diff = (shift_seq - seq)[:-1]
        diff = diff[ diff > 0]

        log_sum.append( np.sum( np.log( diff)))
        sq_sum.append( 0.5 * np.sum( diff * diff) + 0.5 * (T1 - seq[-1]) ** 2)

    n_i = np.array( n_i).reshape(( -1,1))
    sq_sum = np.array( sq_sum).reshape( ( -1,1))
    log_sum = np.array( log_sum).reshape( ( -1,1))

    return uniq_inds, n_i, sq_sum, log_sum




if __name__ == '__main__':
    (ind, y), (train_ind, train_y), (test_ind, test_y) = load_dataSet('article', '../Data')
    uni_ind, n_i, sq_sum, log_sum = extract_event_tensor_Reileigh( train_ind, train_y)

    print( "Done")












