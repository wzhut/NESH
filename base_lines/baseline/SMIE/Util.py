""" Utility Functions """
import torch as t
import numpy as np
import os

def get_dtype_double():
    if t.cuda.is_available():
        return t.cuda.DoubleTensor
    else:
        return t.DoubleTensor

def get_dtype_float():
    if t.cuda.is_available():
        return t.cuda.FloatTensor
    else:
        return t.FloatTensor

def get_dtype_int():
    if t.cuda.is_available():
        return t.cuda.IntTensor
    else:
        return t.IntTensor 

def get_dtype_long():
    if t.cuda.is_available():
        return t.cuda.LongTensor
    else:
        return t.LongTensor

def softplus(x, beta, threshold):
    bx = beta * x
    res = t.zeros_like(bx)

    idx = bx > threshold
    res[idx] = x[idx]
    idx = bx <= threshold
    res[idx] = t.log(1 + t.exp(bx[idx])) / beta
    return res

def log_softplus(x, log_beta, lower_threshold, upper_threshold):
    beta = t.exp(log_beta)
    bx = beta * x 
    res = t.zeros_like(bx)
    idx = bx < lower_threshold
    res[idx] = -log_beta + bx[idx] - 0.5 * t.exp(bx[idx])
    idx = bx >= lower_threshold
    res[idx] = t.log(softplus(x[idx], beta, upper_threshold))
    return res


class NN(t.nn.Module):
    def __init__(self, layers, act=t.nn.functional.relu):
        super(NN,self).__init__()
        self.layers = layers
        self.act = act
        self.fc = t.nn.ModuleList([])
        self.input_dim = self.layers[0]
        self.output_dim = self.layers[-1]
        
        for i in range(len(self.layers) - 1):
            self.fc.append(t.nn.Linear(self.layers[i], self.layers[i+1]))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for i in range(len(self.layers) - 2):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x

class NN2L(t.nn.Module):
    def __init__(self, input_dim, output_dim, hid):
        super(NN2L, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = t.nn.Linear(input_dim, hid)
        self.fc2 = t.nn.Linear(hid, hid)
        self.fc3 = t.nn.Linear(hid, output_dim)
        self.act = t.nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        y = self.fc3(x)
        return y



class Kernel_ARD:
    def __init__(self, jitter):
        self.jitter = jitter
    
    def cross(self, X1, X2, amp, ls):
        norm1 = (X1**2).sum(dim=1).view((-1, 1))
        norm2 = (X2**2).sum(dim=1).view((1, -1))

        K = norm1 - 2.0 * X1 @ X2.transpose(0, 1) + norm2
        K = amp * t.exp(-1.0 * K / ls)
        return K

    def matrix(self, X, amp, ls):
        K = self.cross(X, X, amp, ls)
        K = K + self.jitter * t.eye(X.shape[0], device=K.device, dtype=K.dtype)
        return K

class Kernel_RBF:
    def __init__(self, jitter):
        self.jitter = jitter

    def cross(self, X1, X2, ls):
        norm1 = (X1**2).sum(dim=1).view((-1, 1))
        norm2 = (X2**2).sum(dim=1).view((1, -1))
        K = norm1 - 2.0 * X1 @ X2.transpose(0, 1) + norm2
        K = t.exp(-1.0 * K / ls)
        return K

    def cross_t(self, X1, X2, diff_t, ls):
        norm1 = (X1**2).sum(dim=1).view((-1, 1))
        norm2 = (X2**2).sum(dim=1).view((1, -1))
        K = norm1 - 2.0 * X1 @ X2.transpose(0, 1) + norm2
        K = K / diff_t**2
        K = t.exp(-1.0 * K / ls)
        return K
    
    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        K = K + self.jitter * t.eye(X.shape[0], device=K.device, dtype=K.dtype)
        return K
    
    def pair(self, X1, X2, ls):
        K = ((X1 - X2) ** 2).sum(1)
        K = t.exp(-1.0 * K / ls).view((-1, 1))
        return K



class Normalization:
    def __init__(self, x):
        self.mean = x.mean(dim=0).view((1, -1))
        self.std = x.std(dim=0).view((1, -1))
        self.std[self.std == 0] = 1
    
    def forward_transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return x * self.std + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

def load_dataSet( data_name, path_dataSet_folder = "./Data"):
    '''
    Returning the full, train and test data set. Without special notice full is the concatenation of train and test.
    :param data_name: name of data set
    :param path_dataSet_folder: path to the data folder
    :return: ( full_ind, full_y),( train_ind, train_y),( test_ind, test_y)
    '''
    # valid_names = {'911', 'article', 'slc', 'chicago', 'ufo', 'retail', 'lastfm', 'twitter', 'lastfm_s', 'retail_s', 
    # 'twitter_s', 'lastfm_l', 'taobao_s', 'shop_s', 'taobao_l', 'shopv1_s', 'shopv1_l'}

    # if data_name not in valid_names:
    #     raise NameError("No such data set: %s. valid data set: %s" % ( data_name, str( valid_names)))

    if data_name == 'lastfm':
        ind = np.load( os.path.join(path_dataSet_folder, 'lastfm_55k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'lastfm_55k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'music_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'music_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'music_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == '911_s':
        ind = np.load( os.path.join(path_dataSet_folder, '911_17k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  '911_17k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 10000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'shop_p_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_p_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_p_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'shop_p_l':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_p_70k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_p_70k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'shop_v_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_v_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_v_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'taobao_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'taobao_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'taobao_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'taobao_l':
        ind = np.load( os.path.join(path_dataSet_folder, 'taobao_70k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'taobao_70k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'shop_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'shopv1_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'shopv1_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shopv1_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'shopv1_l':
        ind = np.load( os.path.join(path_dataSet_folder, 'shopv1_70k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shopv1_70k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    
    if data_name == 'lastfm_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'lastfm_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'lastfm_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'lastfm_s2':
        ind = np.load( os.path.join(path_dataSet_folder, 'lastfm_10k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'lastfm_10k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 7000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'lastfm_s3':
        ind = np.load( os.path.join(path_dataSet_folder, 'lastfm_13k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'lastfm_13k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 10000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'lastfm_l':
        ind = np.load( os.path.join(path_dataSet_folder, 'lastfm_56k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'lastfm_56k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'twitter':
        ind = np.load( os.path.join(path_dataSet_folder, 'twitter_63k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'twitter_63k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'twitter_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'twitter_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'twitter_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'retail':
        ind = np.load( os.path.join(path_dataSet_folder, 'retail_70k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'retail_70k_y.npy')).astype(float).reshape(-1, 1)

        NUM_TRAIN = 40000
        NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
        test_y = y[NUM_TRAIN:NUM_TRAIN+NUM_TEST]

        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'retail_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'retail_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'retail_7k_y.npy')).astype(float).reshape(-1, 1)

        NUM_TRAIN = 5000
        NUM_TEST = 2000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
        test_y = y[NUM_TRAIN:NUM_TRAIN+NUM_TEST]

        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == '911':
        ind = np.load( os.path.join(path_dataSet_folder, '911_60k_ind.npy'))
        y = np.load( os.path.join( path_dataSet_folder,  '911_60k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 40000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'ufo':
        ind = np.load( os.path.join(path_dataSet_folder, 'ufo_70k_inds.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'ufo_70k_ys.npy')).reshape(-1, 1)

        y = y / 3600

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

    if data_name == 'article_s':
        ind = np.load(  os.path.join(path_dataSet_folder, 'article_12k_ind.npy')).astype(int)
        y = np.load(  os.path.join(path_dataSet_folder, 'article_12k_y.npy')).reshape(-1, 1)

        hours = 3600
        y = y / hours

        NUM_TRAIN = 10000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return (ind, y), (train_ind, train_y), (test_ind, test_y)

    if data_name == 'slc_s':
        ind = np.load(  os.path.join(path_dataSet_folder, 'slc_15k_ind.npy')).astype(int)
        y = np.load(  os.path.join(path_dataSet_folder, 'slc_15k_y.npy')).reshape(-1, 1)


        NUM_TRAIN = 10000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]

        return (ind, y), (train_ind, train_y), (test_ind, test_y)

    if data_name == 'chicago_s':
        ind = np.load(  os.path.join(path_dataSet_folder, 'chicago_15k_ind.npy')).astype(int)
        y = np.load(  os.path.join(path_dataSet_folder, 'chicago_15k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 10000

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

    if data_name == 'shop_p_m':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_p_12k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_p_12k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 10000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'shop_p_s2':
        ind = np.load( os.path.join(path_dataSet_folder, 'shop_p_10k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'shop_p_10k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 7000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'crash_2015_s':
        ind = np.load( os.path.join(path_dataSet_folder, 'crash_2015_7k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'crash_2015_7k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 5000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'crash_2015_s2':
        ind = np.load( os.path.join(path_dataSet_folder, 'crash_2015_15k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'crash_2015_15k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 10000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'crash_2015':
        ind = np.load( os.path.join(path_dataSet_folder, 'crash_2015_32k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'crash_2015_32k_y.npy')).reshape(-1, 1)

        NUM_TRAIN = 20000
        # NUM_TEST = 30000

        train_ind = ind[:NUM_TRAIN]
        train_y = y[:NUM_TRAIN]

        test_ind = ind[NUM_TRAIN:]
        test_y = y[NUM_TRAIN:]
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'simu-1k':
        ind = np.load( os.path.join(path_dataSet_folder, 'simu-1k_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'simu-1k_y.npy')).reshape(-1, 1)

        train_ind = np.copy(ind)
        train_y = np.copy(y)

        test_ind = np.copy(ind)
        test_y = np.copy(y)
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    if data_name == 'simu-1200':
        ind = np.load( os.path.join(path_dataSet_folder, 'simu-1200_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'simu-1200_y.npy')).reshape(-1, 1)

        train_ind = np.copy(ind)
        train_y = np.copy(y)

        test_ind = np.copy(ind)
        test_y = np.copy(y)
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)
    
    if data_name == 'simu-1200-v2':
        ind = np.load( os.path.join(path_dataSet_folder, 'simu-1200-v2_ind.npy')).astype(int)
        y = np.load( os.path.join( path_dataSet_folder,  'simu-1200-v2_y.npy')).reshape(-1, 1)

        train_ind = np.copy(ind)
        train_y = np.copy(y)

        test_ind = np.copy(ind)
        test_y = np.copy(y)
        return ( ind, y),( train_ind, train_y), ( test_ind, test_y)

    


    
