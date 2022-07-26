import numpy as np
from SNETD_k import SNETD
import argparse
import os
from Util import load_dataSet

def convert_data(data):
    train_ind = []
    train_y = []
    test_ind = []
    test_y = []

    for item in data['train']:
        event = item[1].reshape((-1, 1))
        nevent = event.shape[0]
        train_ind.append(np.tile(item[0], [nevent, 1]))
        train_y.append(event)
    train_ind = np.concatenate(train_ind, 0)
    train_y = np.concatenate(train_y, 0)
    
    for item in data['test']:
        event = item[1].reshape((-1, 1))
        nevent = event.shape[0]
        test_ind.append(np.tile(item[0], [nevent, 1]))
        test_y.append(event)
    test_ind = np.concatenate(test_ind, 0)
    test_y = np.concatenate(test_y, 0)

    train = zip(train_ind, train_y)
    train = sorted(train, key= lambda x: x[1])
    train_ind, train_y = zip(*train)
    train_ind = np.array(train_ind)
    train_y = np.array(train_y).reshape((-1, 1))

    test = zip(test_ind, test_y)
    test = sorted(test, key=lambda x: x[1])
    test_ind, test_y = zip(*test)
    test_ind = np.array(test_ind)
    test_y = np.array(test_y).reshape((-1, 1))
    
    return train_ind, train_y, test_ind, test_y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset name: ufo, taobao_l, crash_2015, lastfm_s3', type=str, required=True)
    parser.add_argument('-r', '--rank', help='Embedding rank', type=int, required=True)
    parser.add_argument('-l', '--lr', help='Learning rate', type=float, default=1e-2)
    parser.add_argument('-w', '--window', help='Trig window size', type=int, default=50)
    parser.add_argument('-e', '--epoch', help='Number of epoch', type=int, default=60)
    # parser.add_argument('-i', '--inducenb', help='Number of inducing point', type=int, required=True)
    parser.add_argument('-c', '--cuda', help='Usage of GPU(-1 cpu, >0 cuda device number)', type=int, default=-1)
    parser.add_argument('-g', '--granularity', help='Time Granularity', type=int, default=0)
    # parser.add_argument('-s', '--spscale', help='Initial softplus scale', type=int, default=100)
    parser.add_argument('-t', '--tauratio', help='Initial decay rate ratio', type=int, default=0)
    parser.add_argument('-f', '--fold', type=int, required=True)


    
    args = parser.parse_args()
    
    fold = args.fold
    R = args.rank
    dataset= args.dataset
    res_file = '{}_r_{}_f_{}_k.txt'.format(dataset, R, fold)
    
    data = np.load('./data_folds/{}_f_{}.npy'.format(dataset, fold), allow_pickle=True).item()
    train_ind, train_y, test_ind, test_y = convert_data(data)
    y_max = np.max(train_y)

    entity_dim = np.max(np.concatenate((train_ind, test_ind), axis=0), axis=0) + 1

    if args.granularity > 0:
        y_max = y_max / args.granularity
    else:
        y_max = 1
    
    if args.tauratio > 0:
        tau_ratio = args.tauratio
    else:
        tau_ratio = 1

    train_y = train_y / y_max
    test_y = test_y / y_max

    entity_dim = np.max(np.concatenate((train_ind, test_ind), axis=0), axis=0) + 1
    
    embedding_dim = args.rank
    cuda = args.cuda
    lr = args.lr
    trig_window = args.window
    neppoch = args.epoch

    log_file = 'SNETD_{0}_r{1}_g{2}_w{3}_t{4}.txt'.format(args.dataset, embedding_dim, args.granularity, trig_window, tau_ratio)
    cfg = {
        'entity_dim': entity_dim,
        'embedding_dim': embedding_dim,
        'induce_nb': 100,
        'cuda': cuda,
        'entry_batch_sz': 100,
        'inner_event_batch_sz': 512,
        'outer_event_batch_sz': 100,
        'trig_window': trig_window,
        't_batch_sz': 10,
        'te_batch_sz': 100,
        'jitter': 1e-5,
        'lr': lr,
        'epoch_nb': neppoch,
        'test_every': 5,
        'sp_lower_threshold': -10,
        'sp_upper_threshold': 20,
        'tr_event': train_y,
        'tr_ind': train_ind,
        'te_event': test_y,
        'te_ind': test_ind,
        'tau_ratio': tau_ratio,
        'log_file': log_file
    }

    model = SNETD(cfg)
    ll = model.train()
    np.savetxt(res_file, [np.max(ll)])
    # print('Best LL:', np.max(ll, axis=0))
    # ofile = 'SNETD_%s_r%d_g%d_w%d.txt' % (args.dataset, embedding_dim, args.granularity, trig_window)
    # np.savetxt(ofile, ll)





