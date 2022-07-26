import numpy as np
# from Util import load_dataSet
from DP_pp import SparseIE
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-r', '--rank', type=int, required=True)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-c', '--cuda', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, required=True)
    # parser.add_argument('-t', '')

    args = parser.parse_args()

    if args.cuda >= 0:
        device = 'cuda:{}'.format(args.cuda)
    else:
        device = 'cpu'

    batch_size = args.batch
    dataset = args.data
    nepoch = args.epoch
    lr = args.lr
    rank = args.rank
    # log_file = '{}_r_{}_b_{}_dp_pp.txt'.format(dataset, rank, batch_size)

    # (ind, y),( train_ind, train_y), ( test_ind, test_y) = load_dataSet(dataset, './Data/')
    # nvec = np.max(np.concatenate((train_ind, test_ind), axis=0), axis=0) + 1
    ll_list = []
    for fold in range(5):
        log_file = '{}_r_{}_b_{}_f_{}_dp_pp.txt'.format(dataset, rank, batch_size, fold+1)
        data = np.load('./data_folds/{}_f_{}.npy'.format(dataset, fold+1), allow_pickle=True).item()
        cfg = {
            'data': data,
            'rank': rank,
            'batch_size': batch_size,
            'nepoch': nepoch,
            'lr': lr,
            'test_every': 5,
            'log_file': log_file,
            'n_time': 10,
            'npseudo': 100,
            'device': device
        }
        model = SparseIE(cfg)
        ll_list.append(model.train())
    
    with open('{}_r_{}_5fold_dp_pp.txt'.format(dataset, rank), 'w') as f:
        # f.write(ll_list)
        f.write('\n{} \t{}'.format(np.mean(ll_list), np.std(ll_list) / np.sqrt(5)))
