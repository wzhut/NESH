import numpy as np
from SMIE import SMIE
import argparse
import os
from Util import load_dataSet


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset name: ufo, taobao_l, crash_2015, lastfm_s3', type=str, required=True)
    parser.add_argument('-r', '--rank', help='Embedding rank', type=int, required=True)
    parser.add_argument('-l', '--lr', help='Learning rate', type=float, required=True)
    parser.add_argument('-e', '--epoch', help='Number of epoch', type=int, required=True)
    # parser.add_argument('-i', '--inducenb', help='Number of inducing point', type=int, required=True)
    parser.add_argument('-c', '--cuda', help='Usage of GPU(-1 cpu, >0 cuda device number)', type=int, default=-1)
    parser.add_argument('-g', '--granularity', help='Time Granularity', type=int, default=0)
    parser.add_argument('-t', '--tauratio', help='Initial decay rate ratio', type=int, default=0)
    parser.add_argument('-s', '--spscale', help='Initial softplus scale', type=int, default=100)

    args = parser.parse_args()

    (ind, y),( train_ind, train_y), ( test_ind, test_y) = load_dataSet(args.dataset, './Data/')
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


    embedding_dim = args.rank
    cuda = args.cuda
    lr = args.lr
    neppoch = args.epoch

    log_file = 'SMIE_{0}_r{1}_g{2}_t{3}_s{4}.txt'.format(args.dataset, embedding_dim, args.granularity, tau_ratio, args.spscale)
    cfg = {
        'entity_dim': entity_dim,
        'embedding_dim': embedding_dim,
        'induce_nb': 100,
        'cuda': cuda,
        'entry_batch_sz': 100,
        'inner_event_batch_sz': 512,
        'outer_event_batch_sz': 100,
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
        'sp_scale': args.spscale,
        'log_file': log_file
    }

    model = SMIE(cfg)
    ll = model.train()
    # print('Best LL:', np.max(ll))
    # ofile = 'SMIE_%s_r%d_g%d.txt' % (args.dataset, embedding_dim, args.granularity)
    # np.savetxt(ofile, ll)





