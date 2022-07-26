# from generate_data_folds import generate_crash
from numpy.core.fromnumeric import size
import torch
import numpy as np
from numpy.random import rand, uniform
from scipy.special import logit
from kernels import KernelRBF
from tqdm import tqdm
from torch.nn.functional import logsigmoid
# from kmeans_pytorch import kmeans
from scipy.cluster.vq import vq, kmeans, whiten
from numpy.polynomial.legendre import leggauss
from torch import nn

np.random.seed(0)
torch.random.manual_seed(0)

class SparseIE:
    def __init__(self, cfg):
        self.jitter = 1e-6
        self.device = torch.device(cfg['device'])
        self.nepoch = cfg['nepoch']
        self.lr = cfg['lr']
        self.batch_size = cfg['batch_size']
        self.test_every = cfg['test_every']
        self.log_file = cfg['log_file']

        # params
        self.params = []

        # data
        self.data = cfg['data']
        self.n_tr = len(self.data['train'])
        self.n_te = len(self.data['test'])
        self.n_time = cfg['n_time']

        self.tr_T = self.data['tr_T']
        self.te_T = self.data['te_T']
        self.T = np.max([self.tr_T, self.te_T])

        # quadrature points and weights
        x, y = leggauss(self.n_time)
        self.quad_point = x
        self.quad_weight = y


        # dimension
        # self.nvec = cfg['nvec']
        self.nvec = self.data['nvec']
        self.nmod = len(self.nvec)

        # embedding rank for each mode
        self.rank = cfg['rank']

        # kernel
        self.k_log_ls = self.add_param(0.)
        # self.k2_log_ls = self.add_param(0.)

        self.k = KernelRBF(self.jitter)

        # beta dist
        self.log_alpha = self.add_param(np.log(2))
       
        self.v_hat = [self.add_param(logit(uniform(low=1e-6, high=1., size=(self.nvec[k], self.rank)))) for k in range(self.nmod)]
        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.nmod * self.rank)
        self.params += list(self.bn.parameters())

        # pseudo input
        self.npseudo = cfg['npseudo']
        # self.pseudo_input = self.add_param(rand(self.npseudo, self.rank * self.nmod))
        self.pseudo_input = self.add_param(self.init_pseudo_inputs())
        self.m = self.add_param(np.zeros((self.npseudo, 1)))
        self.L = self.add_param(np.eye(self.npseudo))

        # optimizer
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def init_pseudo_inputs(self):
        with torch.no_grad():
            ind = []
            for item in self.data['train']:
                ind.append(item[0])
            ind = np.array(ind)
            log_v = [logsigmoid(self.v_hat[k]) for k in range(self.nmod)]
            log_v_minus = [logsigmoid(-self.v_hat[k]) for k in range(self.nmod)]
            cumsum = [torch.cumsum(log_v_minus[k], dim=0) for k in range(self.nmod)]
            log_omega = [(cumsum[k] - log_v_minus[k] + log_v[k]) for k in range(self.nmod)] 

            x = torch.cat([log_omega[k][ind[:, k]].squeeze()  for k in range(self.nmod)], dim=1)
            x = self.bn(x)
            X = whiten(x)
            book = X[np.random.choice(ind.shape[0], size=self.npseudo, replace=False)]
            cluster_centers, _ = kmeans(X, book)
            if cluster_centers.shape[0] < self.npseudo:
                diff = self.npseudo - cluster_centers.shape[0]
                cluster_centers = np.concatenate((cluster_centers, cluster_centers[:diff, :]), 0)
            # cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=self.npseudo, distance='euclidean', device=self.device)
        return cluster_centers

    def add_param(self, init):
        p = torch.tensor(init, device=self.device, dtype=torch.float32, requires_grad=True)
        self.params.append(p)
        return p

    def get_data_tensor(self, data):
        ind = []
        event = []
        nevent = []
        N = len(data)
        n_batch = (N + self.batch_size - 1) // self.batch_size
        for i in range(n_batch):
            batch_data = data[i * self.batch_size : (i+1) * self.batch_size]
            batch_ind, batch_event, batch_nevent = self.generate_batch(batch_data)
            batch_ind = torch.tensor(batch_ind, device=self.device, dtype=torch.int64)
            batch_event = torch.tensor(batch_event, device=self.device, dtype=torch.float32)
            batch_nevent = torch.tensor(batch_nevent, device=self.device, dtype=torch.int64)
            # batch_T = torch.tensor(batch_T, device=self.device, dtype=torch.float32)

            ind.append(batch_ind)
            event.append(batch_event)
            nevent.append(batch_nevent)
        return ind, event, nevent
    
    def rate_func(self, f):
        return f ** 2
    def log_rate_func(self, f):
        return torch.log(f ** 2 + 1e-6) 

    def generate_batch(self, batch_data):
        batch_size = len(batch_data)
        batch_ind = []
        batch_event = []
        for i in range(batch_size):
            batch_ind.append(batch_data[i][0])
            batch_event.append(batch_data[i][1])
        # max_len = np.max([x.shape[0] for x in batch_event])
        # batch_event_padded = np.array([list(x) + [0] * (max_len - x.shape[0]) for x in batch_event])
        # batch_mask = np.array([[1] * x.shape[0] + [0] * (max_len - x.shape[0]) for x in batch_event])
        # batch_T = np.array([np.max(x) for x in batch_event]).reshape((-1, 1))
        batch_nevent = np.array([x.shape[0] for x in batch_event])
        batch_event = np.concatenate(batch_event)
        return batch_ind, batch_event, batch_nevent
    
    def batch_nELBO(self, batch_ind, batch_event, batch_nevent):
        batch_size = batch_ind.shape[0]
        # all unique indices
        log_v = [logsigmoid(self.v_hat[k]) for k in range(self.nmod)]
        log_v_minus = [logsigmoid(-self.v_hat[k]) for k in range(self.nmod)]
        cumsum = [torch.cumsum(log_v_minus[k], dim=0) for k in range(self.nmod)]
        log_omega = [(cumsum[k] - log_v_minus[k] + log_v[k]) for k in range(self.nmod)]

        # log_edge_prob Nx1xK NXR for one batch
        tmp = torch.stack([log_omega[k][batch_ind[:, k], :].squeeze() for k in range(self.nmod)], dim=2).sum(2)
        log_edge_prob = torch.logsumexp(tmp, dim=1).sum()

        # v log prior
        log_v_prior = 0
        for k in range(self.nmod):
            log_v_prior += log_v_minus[k].sum()
        log_v_prior = log_v_prior * (torch.exp(self.log_alpha) - 1)

        # KL divergence
        k_ls = torch.exp(self.k_log_ls)

        z = self.pseudo_input
        
        Ltril = torch.tril(self.L)
        Kzz = self.k.matrix(z, k_ls)
        KL = torch.trace(torch.solve(Ltril @ Ltril.T, Kzz)[0]) + (self.m.T @ torch.solve(self.m, Kzz)[0]).sum() + torch.logdet(Kzz) - torch.log(torch.diag(Ltril) ** 2).sum()
        KL = 0.5 * KL


        # pp log likelihood
        x = torch.cat([log_omega[k][batch_ind[:, k]].squeeze()  for k in range(self.nmod)], dim=1)
        x = self.bn(x)
        Kxx = self.k.matrix(x, k_ls)
        Kxz = self.k.cross(x, z, k_ls)

        f_z = self.m + Ltril @ torch.empty_like(self.m).normal_()
        f_m = Kxz @ torch.solve(f_z, Kzz)[0] 
        f_v = torch.diag(Kxx - Kxz @ torch.solve(Kxz.T, Kzz)[0]).view(-1, 1)
        f = f_m + torch.sqrt(f_v) * torch.empty_like(f_m).normal_()

        # log rate
        # log_rate = (batch_mask * self.log_rate_func(f)).sum()
        log_rate = (self.log_rate_func(f) * batch_nevent.view((-1, 1))).sum()

        # integral
        integral = -(self.tr_T * self.rate_func(f)).sum()

        ELBO = (integral + log_rate + log_edge_prob) * self.n_tr / batch_size + log_v_prior - KL

        return -ELBO

    def train(self):
        ll_list = []
        # generate training batch
        tr_ind, tr_event, tr_nevent = self.get_data_tensor(self.data['train'])
        te_ind, te_event, te_nevent = self.get_data_tensor(self.data['test'])
        tr_n_batch = (self.n_tr + self.batch_size - 1) // self.batch_size
        te_n_batch = (self.n_te + self.batch_size - 1) // self.batch_size
        for epoch in tqdm(range(self.nepoch)):
            for idx_batch in range(tr_n_batch):
                batch_ind = tr_ind[idx_batch]
                batch_event = tr_event[idx_batch]
                batch_nevent = tr_nevent[idx_batch]
                # batch_T = tr_T[idx_batch]

                self.optimizer.zero_grad()
                nELBO = self.batch_nELBO(batch_ind, batch_event, batch_nevent)
                nELBO.backward()
                self.optimizer.step()

            if epoch % self.test_every == 0:
                with torch.no_grad():
                    ll = 0
                    for idx_batch in range(te_n_batch):
                        batch_ind = te_ind[idx_batch]
                        batch_event = te_event[idx_batch]
                        batch_nevent = te_nevent[idx_batch]
                        # batch_T = te_T[idx_batch]

                        ll += self.test(batch_ind, batch_event, batch_nevent).item()
                    ll_list.append(ll)
                    with open(self.log_file, 'a') as f:
                        f.write('{}\n'.format(ll))
                    print('Loglikelihood: {}'.format(ll))
                    print('ELBO: {}'.format(nELBO.item()))

        return np.max(ll_list)

    def test(self, batch_ind, batch_event, batch_nevent):
        log_v = [logsigmoid(self.v_hat[k]) for k in range(self.nmod)]
        log_v_minus = [logsigmoid(-self.v_hat[k]) for k in range(self.nmod)]
        cumsum = [torch.cumsum(log_v_minus[k], dim=0) for k in range(self.nmod)]
        log_omega = [(cumsum[k] - log_v_minus[k] + log_v[k]) for k in range(self.nmod)]

        k_ls = torch.exp(self.k_log_ls)

        z = self.pseudo_input
        Kzz = self.k.matrix(z, k_ls)

        # construct embedding
        x = torch.cat([log_omega[k][batch_ind[:, k]].squeeze()  for k in range(self.nmod)], dim=1)
        x = self.bn(x)
        Kxz = self.k.cross(x, z, k_ls)

        # log rate
        f = Kxz @ torch.solve(self.m, Kzz)[0]
        # log_rate = self.log_rate_func(f).sum()
        # log_rate = (self.log_rate_func(f) * batch_mask).sum()
        log_rate = (self.log_rate_func(f) * batch_nevent.view((-1, 1))).sum()

        # integral sum
        # integral = -(batch_T * self.rate_func(f)).sum()
        integral = - (self.te_T * self.rate_func(f)).sum()
        ll = log_rate + integral
        return ll

