import torch as t
import numpy as np
from tqdm import tqdm
import Util as util

np.random.seed(0)
t.random.manual_seed(0)

class SNETD:
    def __init__(self, cfg):
        """ Load configuration """
        self.entity_dim = cfg['entity_dim']
        self.embedding_dim = cfg['embedding_dim']
        self.entity_typenb = len(self.entity_dim)

        self.induce_nb = cfg['induce_nb']
        self.trig_window = cfg['trig_window']
        self.cuda = cfg['cuda']
        
        self.entry_batch_sz = cfg['entry_batch_sz']
        self.outer_event_batch_sz = cfg['outer_event_batch_sz']
        self.inner_event_batch_sz = cfg['inner_event_batch_sz']
        self.te_batch_sz = cfg['te_batch_sz']

        self.jitter = cfg['jitter']
        self.lr = cfg['lr']
        self.epoch_nb = cfg['epoch_nb']
        self.test_every = cfg['test_every']

        self.tau_ratio = cfg['tau_ratio']
        self.log_file = cfg['log_file']

        # load data
        self.tr_event = t.from_numpy(cfg['tr_event'])
        self.tr_ind = t.from_numpy(cfg['tr_ind']).long()
        self.te_event = t.from_numpy(cfg['te_event'])
        self.te_ind = t.from_numpy(cfg['te_ind']).long()
        if self.cuda >= 0:
            self.tr_event = self.tr_event.cuda(self.cuda).double()
            self.tr_ind = self.tr_ind.cuda(self.cuda)
            self.te_event = self.te_event.cuda(self.cuda).double()
            self.te_ind = self.te_ind.cuda(self.cuda)

        self.tr_first_idx = self.first_idx(cfg['tr_event'].reshape(-1))
        self.tr_last_idx = self.last_idx(cfg['tr_event'].reshape(-1))

        self.te_first_idx = self.first_idx(cfg['te_event'].reshape(-1))
        self.te_last_idx = self.last_idx(cfg['te_event'].reshape(-1))
        
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

        # k
        self.k_log_ls = self.register_param(t.tensor([0.]))

        # self.log_dr = self.register_param(t.tensor([0.]))
        # self.log_dr = self.register_param(t.log(self.tr_T.clone()))
        self.log_dr = self.register_param(t.log(self.tr_T.clone()) / self.tau_ratio)

        self.log_beta = self.register_param(t.tensor([0.]))

        # posterior
        # f
        self.f_b = self.register_param(t.rand((self.induce_nb, 1)))
        self.f_L = self.register_param(t.eye(self.induce_nb))
        self.f_B = self.register_param(t.rand((self.induce_nb, self.embedding_dim * self.entity_typenb)))

        # optimizer
        self.optimizer = t.optim.Adam(self.params, lr=self.lr)
    
    def first_idx(self, events):
        first_idx = np.zeros_like(events)
        for i in range(1, len(events)):
            if events[i] == events[i-1]:
                first_idx[i] = first_idx[i-1]
            else:
                first_idx[i] = i
        return first_idx
    
    def last_idx(self, events):
        last_idx = np.ones_like(events) * (len(events) - 1)
        for i in range(len(events) - 2, 0, -1):
            if events[i] == events[i+1]:
                last_idx[i] = last_idx[i+1]
            else:
                last_idx[i] = i
        return last_idx

    def get_batch_idx(self, total_nb, batch_sz):
        indices = np.arange(total_nb)
        np.random.shuffle(indices)
        return [indices[i: i + batch_sz] for i in range(0, total_nb, batch_sz)]

    def train_mode(self, is_train=True):
        for v in self.params:
            v.requires_grad_(is_train)

    def register_param(self, v):
        v = v.double()
        if self.cuda >= 0:
            v = v.cuda(self.cuda)
        v.requires_grad_()
        self.params.append(v)
        return v

    def construct_x(self, ind):
        x = []
        # print(ind.shape)
        for i in range(self.entity_typenb):
            ind_i = ind[:,i].view(-1)
            x.append(self.embedding[i].index_select(0, ind_i))
        x = t.cat(x, 1)
        return x
    
    def train(self):
        self.train_mode(True)

        iter_epoch = (self.tr_nb + self.outer_event_batch_sz - 1) // self.outer_event_batch_sz 
        test_every = self.test_every
        ll_list = []
        max_ll = -1000000000
        for epoch in range(self.epoch_nb):
            # for i in range(iter_epoch):
            for i in tqdm(range(iter_epoch), ascii=True, desc='Epoch: %d' % (epoch)):
                self.optimizer.zero_grad()
                nELBO = self.get_nELBO()
                # print(nELBO.item())
                nELBO.backward()
                for k, p in enumerate(self.params):
                    if t.isnan(p.grad).sum() > 0:
                        print(k, 'Nan grad')
                        print(nELBO)
                        # pause = input('Pause.')
                        return ll_list
                self.optimizer.step()
                
            if epoch % test_every == 0:
                self.train_mode(False)
                ll = self.test()
                print('Epoch: %d Test LL: %e' % (epoch, ll))
                ll_list.append(ll)
                # np.savetxt(self.log_file, ll_list)
                # if max_ll < ll:
                #     with open(self.log_file + 'params', 'wb') as f:
                #         t.save(self.params, f)
                #     for i in range(self.entity_typenb):
                #         emb = self.embedding[i].detach().cpu().numpy()
                #         np.savetxt(self.log_file + 'embedding{0}.txt'.format(i), emb)
                # np.savetxt(self.log_file, ll_list)
                self.train_mode(True)
        return ll_list

    def test(self):
        ll = -self.te_T1() + self.te_T2()
        return ll

    def tr_mul(self, A, B):
        return (A * B.transpose(0, 1)).sum()

    def get_nELBO(self):

        # f KL
        fLtril = t.tril(self.f_L)
        fkBB = self.kernel_rbf.matrix(self.f_B, t.exp(self.f_log_ls))
        fKL = 0.5 * (
            self.tr_mul(t.solve(fLtril, fkBB)[0], fLtril.transpose(0, 1)) 
            + self.f_b.transpose(0, 1) @ t.solve(self.f_b, fkBB)[0]
            + t.logdet(fkBB) - t.log(t.diag(fLtril) ** 2).sum()
        )

        # log F(S, U)
        F = 0
        for i in range(self.entity_typenb):
            F = F +  -0.5 * (self.embedding[i] ** 2).sum()
        T1 = self.tr_T1()
        T2 = self.tr_T2()

        # print('F', F)
        F = F - T1 + T2

        ELBO = -fKL + F
        # print('KL', fKL)
        
        # print(T1)
        # print(T2)
        # print(ELBO)
        
        return -ELBO
    
    def tr_T1(self):
        # sample entry batch
        k = np.random.randint(len(self.entry_batch_idx))
        idx = self.entry_batch_idx[k]
        ind1 = self.tr_uind[idx]
        entry_batch_sz = ind1.shape[0]

        x1 = self.construct_x(ind1)
        # base rate
        res = self.base_rate(ind1) * self.tr_T
        # calculate integral by batch
        for idx in self.inner_event_batch_idx:
            ind2 = self.tr_ind[idx]
            time_s = self.tr_event[idx]

            last_idx = self.tr_last_idx[idx]
            idx_e = np.minimum(last_idx + self.trig_window, self.tr_nb - 1)
            time_e = self.tr_event[idx_e]

            x2 = self.construct_x(ind2)

            K = self.kernel_rbf.cross(x1, x2, t.exp(self.k_log_ls))

            time_diff = time_e - time_s
            beta = t.exp(self.log_beta)
            dr = t.exp(self.log_dr)
            int_h = beta * dr * (1 - t.exp(-time_diff / dr))
            # int_h = dr * (1 - t.exp(-time_diff / dr)) 
            int_h = int_h.view((1, -1))
          
            int_h = (K * int_h).sum(1).view((-1, 1))
            res = res + int_h
        res = res.sum() * self.tr_uind_nb / entry_batch_sz
        return res
    
    def te_T1(self):

        M = self.te_uind.shape[0]
        total_iter = (M + self.te_batch_sz - 1) // self.te_batch_sz
        N = self.te_nb
        inner_total_iter = (N + self.inner_event_batch_sz - 1) // self.inner_event_batch_sz
        T1 = 0
        for i in tqdm(range(total_iter), ascii=True, desc='Test T1'):
            start = i * self.te_batch_sz
            end = np.min((start + self.te_batch_sz, M)) 

            ind1 = self.te_uind[start:end]
            x1 = self.construct_x(ind1)
            # base rate
            res = self.base_rate(ind1, False) * self.te_T
            # calculate integral by batch
            for j in range(inner_total_iter):
                start = j * self.inner_event_batch_sz
                end = np.min((start + self.inner_event_batch_sz, N)) 
                idx = np.arange(start, end)
                ind2 = self.te_ind[idx]
                time_s = self.te_event[idx]
                last_idx = self.te_last_idx[idx]
                idx_e = np.minimum(last_idx + self.trig_window, N - 1)
                time_e = self.te_event[idx_e]
                x2 = self.construct_x(ind2)

                K = self.kernel_rbf.cross(x1, x2, t.exp(self.k_log_ls))

                time_diff = time_e - time_s
                beta = t.exp(self.log_beta)
                dr = t.exp(self.log_dr)
                int_h = beta * dr * (1 - t.exp(-time_diff / dr))
                # int_h = dr * (1 - t.exp(-time_diff / dr))
                int_h = int_h.view((1, -1))

                int_h = (K * int_h).sum(1).view((-1, 1))
                res = res + int_h
            T1 = T1 + res.sum()
        return T1


    
    def tr_T2(self):
        # sample first event batch
        k = np.random.randint(len(self.outer_event_batch_idx))
        idx = self.outer_event_batch_idx[k]
        ind1 = self.tr_ind[idx]
        time_e = self.tr_event[idx].view((-1, 1))
        outer_event_batch_sz = ind1.shape[0]
        x1 = self.construct_x(ind1)
        base_rate = self.base_rate(ind1)

        # get trig window event
        first_idx = self.tr_first_idx[idx]
        idx_s = np.maximum(first_idx - self.trig_window, 0).reshape((-1, 1))
        idx = idx_s + np.arange(self.trig_window).reshape((1, -1))
        idx = np.minimum(idx, self.tr_nb - 1).reshape(-1)
        time_s = self.tr_event[idx].view((outer_event_batch_sz, self.trig_window))
        ind2 = self.tr_ind[idx]

        mask = (time_s < time_e).type(t.float64)

        x2 = self.construct_x(ind2)
        x1 = x1.repeat_interleave(self.trig_window, dim=0)
        K = self.kernel_rbf.pair(x1, x2, t.exp(self.k_log_ls)).view((outer_event_batch_sz, self.trig_window))

        beta = t.exp(self.log_beta)
        dr = t.exp(self.log_dr)
        time_diff = (time_e - time_s) * mask
        res = base_rate + (mask * K * beta * t.exp(-time_diff / dr)).sum(1).view((-1, 1))
        # res = base_rate + (mask * K * t.exp(-time_diff / dr)).sum(1).view((-1, 1))
        res = t.log(res).sum() * self.tr_nb / outer_event_batch_sz
        return res
    
    def te_T2(self):

        # start = 0
        M = self.te_nb
        total_iter = (M + self.te_batch_sz - 1) // self.te_batch_sz
        T2 = 0
        for i in tqdm(range(total_iter), ascii=True, desc='Test T2'):
            start = i * self.te_batch_sz
            end = np.min((start + self.te_batch_sz, M)) 
            idx = np.arange(start, end)
            # sample first event batch
            ind1 = self.te_ind[idx]
            time_e = self.te_event[idx].view((-1, 1))
            outer_event_batch_sz = ind1.shape[0]
            x1 = self.construct_x(ind1)
            base_rate = self.base_rate(ind1, False)

            # get trig window event
            first_idx = self.te_first_idx[idx]
            idx_s = np.maximum(first_idx - self.trig_window, 0).reshape((-1, 1))
            idx = idx_s + np.arange(self.trig_window).reshape((1, -1))
            idx = np.minimum(idx, self.te_nb - 1).reshape(-1)
            time_s = self.te_event[idx].view((outer_event_batch_sz, self.trig_window))
            ind2 = self.te_ind[idx]

            mask = (time_s < time_e).type(t.float64)

            x2 = self.construct_x(ind2)
            x1 = x1.repeat_interleave(self.trig_window, dim=0)
            K = self.kernel_rbf.pair(x1, x2, t.exp(self.k_log_ls)).view((outer_event_batch_sz, self.trig_window))

            beta = t.exp(self.log_beta)
            dr = t.exp(self.log_dr)
            time_diff = (time_e - time_s) * mask
            res = base_rate + (mask * K * beta * t.exp(-time_diff / dr)).sum(1).view((-1, 1))
            # res = base_rate + (mask * K * t.exp(-time_diff / dr)).sum(1).view((-1, 1))
            res = t.log(res).sum()
            T2 = T2 + res
        return T2

    def base_rate(self, ind, sample=True):
        x = self.construct_x(ind)
        fkXB = self.kernel_rbf.cross(x, self.f_B, t.exp(self.f_log_ls))
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
        base_rate = t.exp(f)
        return base_rate

