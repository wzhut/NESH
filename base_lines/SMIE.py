import torch as t
import numpy as np
import Util as util
from tqdm import tqdm

np.random.seed(0)
t.random.manual_seed(0)

class SMIE:
    def __init__(self, cfg):

        """ Load configuration """
        self.entity_dim = cfg['entity_dim']
        self.embedding_dim = cfg['embedding_dim']
        self.entity_typenb = len(self.entity_dim)

        self.induce_nb = cfg['induce_nb']
        self.cuda = cfg['cuda']
        
        self.entry_batch_sz = cfg['entry_batch_sz']
        self.outer_event_batch_sz = cfg['outer_event_batch_sz']
        self.inner_event_batch_sz = cfg['inner_event_batch_sz']
        self.t_batch_sz = cfg['t_batch_sz']
        self.te_batch_sz = cfg['te_batch_sz']

        self.jitter = cfg['jitter']
        self.lr = cfg['lr']
        self.epoch_nb = cfg['epoch_nb']
        self.test_every = cfg['test_every']

        self.sp_lower_threshold = cfg['sp_lower_threshold']
        self.sp_upper_threshold = cfg['sp_upper_threshold']

        self.tau_ratio = cfg['tau_ratio']
        self.sp_scale = cfg['sp_scale']

        self.log_file = cfg['log_file']

        # load data
        self.tr_event = t.from_numpy(cfg['tr_event'])
        self.tr_ind = t.from_numpy(cfg['tr_ind']).long()
        self.te_event = t.from_numpy(cfg['te_event'])
        self.te_ind = t.from_numpy(cfg['te_ind']).long()
        
        if self.cuda >= 0:
            self.tr_event = self.tr_event.cuda(self.cuda).float()
            self.tr_ind = self.tr_ind.cuda(self.cuda)
            self.te_event = self.te_event.cuda(self.cuda).float()
            self.te_ind = self.te_ind.cuda(self.cuda)
        
        # shift
        min_t = t.min(self.tr_event)
        self.tr_event = self.tr_event - min_t
        self.tr_T = t.max(self.tr_event)

        min_t = t.min(self.te_event)
        self.te_event = self.te_event - min_t
        self.te_T = t.max(self.te_event)


        self.tr_nb = self.tr_event.shape[0]
        self.tr_uind, self.tr_u2o = t.unique(self.tr_ind, dim=0, return_inverse=True)
        self.tr_uind_nb = self.tr_uind.shape[0]

        self.te_nb = self.te_event.shape[0]
        self.te_uind, self.te_u2o = t.unique(self.te_ind, dim=0, return_inverse=True)
        self.te_uind_nb = self.te_uind.shape[0]

        # split training data into batches (indices)
        self.entry_batch_idx = self.get_batch_idx(self.tr_uind_nb, self.entry_batch_sz)
        self.inner_event_batch_idx = self.get_batch_idx(self.tr_nb, self.inner_event_batch_sz)
        self.outer_event_batch_idx = self.get_batch_idx(self.tr_nb, self.outer_event_batch_sz)

        """ Register parameters and optimizer """       
        self.params = []
        # embedding
        self.embedding = []
        for i in range(self.entity_typenb):
            d = self.entity_dim[i]
            r = self.embedding_dim
            self.embedding.append(
                  self.register_param(t.rand((d, r)))
                                 )
        # kernel
        self.kernel_rbf = util.Kernel_RBF(self.jitter)
        # f
        self.f_log_ls = self.register_param(t.tensor([0.]))
        # g
        self.g_log_ls = self.register_param(t.tensor([0.]))
        # k
        self.k_log_ls = self.register_param(t.tensor([0.]))

        # self.log_dr = self.register_param(t.tensor([0.]))
        self.log_dr = self.register_param(t.log(self.tr_T.clone()) / self.tau_ratio)
        self.sp_log_ls = self.register_param(t.tensor([np.log(self.sp_scale)]))

        # posterior
        # f
        self.f_b = self.register_param(t.rand((self.induce_nb, 1)))
        self.f_L = self.register_param(t.eye(self.induce_nb))
        self.f_B = self.register_param(t.rand((self.induce_nb, self.embedding_dim * self.entity_typenb)))
        # g
        self.g_b = self.register_param(t.rand((self.induce_nb, 1)))
        self.g_L = self.register_param(t.eye(self.induce_nb))
        self.g_B = self.register_param(t.rand((self.induce_nb, self.embedding_dim * self.entity_typenb)))

        # optimizer
        self.optimizer = t.optim.Adam(self.params, lr=self.lr)
    
    def get_batch_idx(self, total_nb, batch_sz):
        indices = np.arange(total_nb)
        np.random.shuffle(indices)
        return [indices[i: i + batch_sz] for i in range(0, total_nb, batch_sz)]

    def train_mode(self, is_train=True):
        for v in self.params:
            v.requires_grad_(is_train)

    def register_param(self, v):
        if self.cuda >= 0:
            v = v.cuda(self.cuda)
        v.requires_grad_()
        self.params.append(v)
        return v
    
    def tr_mul(self, A, B):
        return (A * B.transpose(0, 1)).sum()
    
    def train(self):
        self.train_mode(True)

        iter_epoch = (self.tr_nb + self.outer_event_batch_sz - 1) // self.outer_event_batch_sz 
        test_every = self.test_every
        ll_list = []
        max_ll = -10000000000
        for epoch in range(self.epoch_nb):
            # for i in range(iter_epoch):
            for i in tqdm(range(iter_epoch), ascii=True, desc='Epoch: %d' % (epoch)):
                self.optimizer.zero_grad()
                nELBO = self.get_nELBO()
                nELBO.backward()
                for k, p in enumerate(self.params):
                    if t.isnan(p.grad).sum() > 0:
                        print(k, 'Nan grad')
                        pause = input('Pause.')
                self.optimizer.step()
                
            if epoch % test_every == 0:
                self.train_mode(False)
                ll = self.test()
                print('Epoch: %d Test LL: %e' % (epoch, ll))
                ll_list.append(ll)
                if max_ll < ll:
                    with open(self.log_file + 'params', 'wb') as f:
                        t.save(self.params, f)
                    for i in range(self.entity_typenb):
                        emb = self.embedding[i].detach().cpu().numpy()
                        np.savetxt(self.log_file + 'embedding{0}.txt'.format(i), emb)
                np.savetxt(self.log_file, ll_list)
                self.train_mode(True)
        return ll_list

    def test(self):
        ll = -self.te_T1() + self.te_T2()
        return ll

    def get_nELBO(self):

        # f KL
        fLtril = t.tril(self.f_L)
        fkBB = self.kernel_rbf.matrix(self.f_B, t.exp(self.f_log_ls))
        fKL = 0.5 * (
            self.tr_mul(t.solve(fLtril, fkBB)[0], fLtril.transpose(0, 1)) 
            + self.f_b.transpose(0, 1) @ t.solve(self.f_b, fkBB)[0]
            + t.logdet(fkBB) - t.log(t.diag(fLtril) ** 2).sum()
        )

        # g KL
        gLtril = t.tril(self.g_L)
        gkBB = self.kernel_rbf.matrix(self.g_B, t.exp(self.g_log_ls))
        gKL = 0.5 * (
            self.tr_mul(t.solve(gLtril, gkBB)[0], gLtril.transpose(0, 1))
            + self.g_b.transpose(0, 1) @ t.solve(self.g_b, gkBB)[0]
            + t.logdet(gkBB) - t.log(t.diag(gLtril) ** 2).sum()
        )

        # log F(S, U)
        F = 0
        for i in range(self.entity_typenb):
            F = F +  -0.5 * (self.embedding[i] ** 2).sum()
        T1 = self.tr_T1()
        T2 = self.tr_T2()
        F = F - T1 + T2
        ELBO = -fKL - gKL + F
        # print('T1', T1)
        # print('T2', T2)
        
        return -ELBO
    
    def tr_T1(self):
        # sample entry batch
        M = self.tr_uind.shape[0]
        k = np.random.randint(len(self.entry_batch_idx))
        idx = self.entry_batch_idx[k]
        ind1 = self.tr_uind[idx]
        time1 = self.tr_event[idx].view((-1, 1))
        entry_batch_sz = ind1.shape[0]
        # sample event batch
        k = np.random.randint(len(self.inner_event_batch_idx))
        idx = self.inner_event_batch_idx[k]
        ind2 = self.tr_ind[idx]
        time2 = self.tr_event[idx].view((-1, 1))
        event_batch_sz = ind2.shape[0]

        # sample time
        T1 = 0
        for _ in range(self.t_batch_sz):
            time1 = t.ones_like(time1) * np.random.rand() * self.tr_T
            X = self.X(ind1, ind2, time1, time2, ratio=self.tr_nb / event_batch_sz) # ind1 -> ind2
            hX = util.softplus(X, t.exp(self.sp_log_ls), self.sp_upper_threshold)
            T1 = T1 + hX.sum()
        T1 = self.tr_T * T1 * M / entry_batch_sz / self.t_batch_sz
        return T1
    
    def te_T1(self):
        # train event
        ind2 = self.te_ind
        time2 = self.te_event.view((-1, 1))
        # test batch by batch
        start = 0 
        T1 = 0
        M = self.te_uind.shape[0]
        total_iter = (M + self.te_batch_sz - 1) // self.te_batch_sz
        for i in tqdm(range(total_iter), ascii=True, desc='Test T1'):
            start = i * self.te_batch_sz
            end = np.min((start + self.te_batch_sz, M)) 

            ind1 = self.te_uind[start:end]
            time1 = self.te_event[start:end]
            # sample time
            res = 0
            for _ in range(self.t_batch_sz):
                time1 = t.ones_like(time1) * np.random.rand() * self.te_T
                X = self.X(ind1, ind2, time1, time2, False)
                # hX = t.nn.functional.softplus(X, beta=t.exp(self.sp_log_ls), threshold=self.sp_upper_threshold)
                hX = util.softplus(X, t.exp(self.sp_log_ls), self.sp_upper_threshold)
                res = res + hX.sum()
            # res = self.tr_T * res / self.t_batch_sz
            T1 = T1 + res
        T1 = T1 * self.te_T / self.t_batch_sz
        return T1
    
    def tr_T2(self):
        # sample first event batch
        k = np.random.randint(len(self.outer_event_batch_idx))
        idx = self.outer_event_batch_idx[k]
        ind1 = self.tr_ind[idx]
        time1 = self.tr_event[idx].view((-1, 1))
        outer_event_batch_sz = ind1.shape[0]
        # sample second event batch
        k = np.random.randint(len(self.inner_event_batch_idx))
        idx = self.inner_event_batch_idx[k]
        ind2 = self.tr_ind[idx]
        time2 = self.tr_event[idx].view((-1, 1))
        inner_event_batch_sz = ind2.shape[0]

        X = self.X(ind1, ind2, time1, time2, ratio=self.tr_nb / inner_event_batch_sz)
        loghX = util.log_softplus(X, self.sp_log_ls, self.sp_lower_threshold, self.sp_upper_threshold)
        T2 = loghX.sum() / outer_event_batch_sz * self.tr_nb
        return T2
    
    def te_T2(self):
        ind2 = self.te_ind
        time2 = self.te_event.view((-1, 1))

        # start = 0
        M = self.te_ind.shape[0]
        total_iter = (M + self.te_batch_sz - 1) // self.te_batch_sz

        T2 = 0
        for i in tqdm(range(total_iter), ascii=True, desc='Test T2'):
            start = i * self.te_batch_sz
            end = np.min((start + self.te_batch_sz, M)) 
            ind1 = self.te_ind[start:end]
            time1 = self.te_event[start:end]
            X = self.X(ind1, ind2, time1, time2, False)
            # loghX = self.log_softplus(X)
            loghX = util.log_softplus(X, self.sp_log_ls, self.sp_lower_threshold, self.sp_upper_threshold)
            T2 = T2 + loghX.sum()
        return T2



    def construct_x(self, ind):
        x = []
        # print(ind.shape)
        for i in range(self.entity_typenb):
            ind_i = ind[:,i].view(-1)
            x.append(self.embedding[i].index_select(0, ind_i))
        x = t.cat(x, 1)
        return x
    
    def X(self, ind1, ind2, time1, time2, sample=True, ratio=1.0):
        ind1_nb = ind1.shape[0]
        ind2_nb = ind2.shape[0]
        
        x1 = self.construct_x(ind1)
        x2 = self.construct_x(ind2)
        dim = x1.shape[1]
        # f
        fkXB = self.kernel_rbf.cross(x1, self.f_B, t.exp(self.f_log_ls))
        fkBB = self.kernel_rbf.matrix(self.f_B, t.exp(self.f_log_ls))
        if sample:
            # sample f_b
            fLtril = t.tril(self.f_L)
            epsilon = self.f_b.new(self.f_b.shape).normal_(mean=0, std=1)
            f_b = self.f_b + fLtril @ epsilon
            # sample f
            mean = fkXB @ t.solve(f_b, fkBB)[0]
            var  = 1.0 + self.jitter - (fkXB.transpose(0, 1) * t.solve(fkXB.transpose(0, 1), fkBB)[0]).sum(0) 
            std = t.sqrt(var).view(mean.shape)
            epsilon = mean.new(mean.shape).normal_(mean=0, std=1)
            f = mean + std * epsilon
        else:
            f_b = self.f_b
            mean = fkXB @ t.solve(f_b, fkBB)[0]
            f = mean

        # k
        K = self.kernel_rbf.cross(x1, x2, t.exp(self.k_log_ls)) # n1 x n2

        # g
        x = x2.view((1, ind2_nb, dim)) - x1.view((ind1_nb, 1, dim))
        x = x.view((-1, dim))
        gkXB = self.kernel_rbf.cross(x, self.g_B, t.exp(self.g_log_ls))
        gkBB = self.kernel_rbf.matrix(self.g_B, t.exp(self.g_log_ls))
        if sample:
            # sample g_b
            gLtril = t.tril(self.g_L)
            epsilon = self.g_b.new(self.g_b.shape).normal_(mean=0, std=1)
            g_b = self.g_b + gLtril @ epsilon
            # sample g
            mean = gkXB @ t.solve(g_b, gkBB)[0]
            var = 1.0 + self.jitter - (gkXB.transpose(0, 1) * t.solve(gkXB.transpose(0, 1), gkBB)[0]).sum(0)
            std = t.sqrt(var).view(mean.shape)
            epsilon = mean.new(mean.shape).normal_(mean=0, std=1)
            g = mean + std * epsilon
        else:
            g_b = self.g_b
            mean = gkXB @ t.solve(g_b, gkBB)[0]
            g = mean
            
        # mask
        mask = (time1 > time2.transpose(0, 1)).type(t.float32) # n1 x n2

        # reshape
        g = g.view((ind1_nb, ind2_nb))
        t_diff = (time1 - time2.transpose(0,1)) * mask
        effect = t.tanh(g) * K * t.exp(-1.0 * t_diff / t.exp(self.log_dr)) * mask
        effect = effect.sum(1).view(f.shape) * ratio
        X = f + effect
        return X







