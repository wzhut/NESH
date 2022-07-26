import numpy as np
from SMIE import SMIE
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
    
    return train_ind, train_y, test_ind, test_y

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset name: 911, article, ufo, slc', type=str, required=True)
    parser.add_argument('-r', '--rank', help='Embedding rank', type=int, required=True)
    parser.add_argument('-f', '--fold', type=int, required=True)
    parser.add_argument('-l', '--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epoch', help='Number of epoch', type=int, default=400)
    # parser.add_argument('-i', '--inducenb', help='Number of inducing point', type=int, required=True)
    parser.add_argument('-g', '--gpu', help='Usage of GPU', action='store_true', default=False)

    args = parser.parse_args()

    fold = args.fold
    R = args.rank
    dataset= args.dataset
    res_file = '{}_r_{}_f_{}.txt'.format(dataset, R, fold)
    
    data = np.load('{}_f_{}.npy'.format(dataset, fold), allow_pickle=True).item()
    train_ind, train_y, test_ind, test_y = convert_data(data)
    nvec = data['nvec']

    # (ind, y),( train_ind, train_y), ( test_ind, test_y) = load_dataSet(args.dataset, './Data/')

    entity_dim = np.max(np.concatenate((train_ind, test_ind), axis=0), axis=0) + 1

    embedding_dim = args.rank
    cuda = args.gpu
    lr = args.lr
    neppoch = args.epoch


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
        'te_ind': test_ind
    }

    model = SMIE(cfg)
    ll = model.train()
    print('Best LL:', np.max(ll))
    ofile = '%s_r%d_f%d.txt' % (args.dataset, embedding_dim, args.fold)
    np.savetxt(ofile, ll)





