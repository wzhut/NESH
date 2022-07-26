import numpy as np
import os

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