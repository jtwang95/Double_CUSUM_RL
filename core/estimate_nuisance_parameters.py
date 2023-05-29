import logging
import os
import sys

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.mixture
import torch
import torch.optim as optim
from core.neural_nets import *
from sklearn.model_selection import GridSearchCV
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("./")
from Utils.pytorch_utils import SimpleDataset

mylogger = logging.getLogger("testSA2")


class estimate_pt():
    def __init__(self, s, a, sp, r, lr, epoch_candidates, hidden_dims, s_test,
                 a_test, sp_test, r_test, **kwargs) -> None:
        # mylogger.warn(
        #     "you are using 'estimate_pt_alt' function that still be tested.")
        assert (s.shape[0] == a.shape[0]) & (a.shape[0] == sp.shape[0]) & (
            sp.shape[0] == r.shape[0])
        self.s_dim = s.shape[-1]
        self.s = np.array(s).reshape(-1, self.s_dim)
        self.a = np.array(a).reshape(-1, 1)
        self.sp = np.array(sp).reshape(-1, self.s_dim)
        self.r = np.array(r).reshape(-1, 1)
        self.spr = np.concatenate([self.sp, self.r], axis=1)
        self.lr = lr
        self.epoch_candidates = epoch_candidates
        self.max_epoches = max(epoch_candidates)
        self.l2_lambda = 0  #1e-4
        self.hidden_dims = hidden_dims
        # normalization
        self.mean_s, self.std_s = np.mean(self.s, axis=0), np.std(self.s,
                                                                  axis=0)
        self.mean_spr, self.std_spr = np.mean(self.spr,
                                              axis=0), np.std(self.spr, axis=0)
        self.s_n = (self.s - self.mean_s) / self.std_s
        self.spr_n = (self.spr - self.mean_spr) / self.std_spr
        self.s_test = s_test
        self.a_test = a_test
        self.sp_test = sp_test
        self.r_test = r_test
        self.fit()

    def fit(self):

        # data preparation
        idxs_a0 = (self.a == 0).flatten()
        idxs_a1 = (self.a == 1).flatten()
        s_n_a0 = self.s_n[idxs_a0]
        s_n_a1 = self.s_n[idxs_a1]
        spr_n_a0 = self.spr_n[idxs_a0]
        spr_n_a1 = self.spr_n[idxs_a1]

        # tensorization
        s_n_a0_t = torch.tensor(s_n_a0, dtype=torch.float32)
        s_n_a1_t = torch.tensor(s_n_a1, dtype=torch.float32)
        spr_n_a0_t = torch.tensor(spr_n_a0, dtype=torch.float32)
        spr_n_a1_t = torch.tensor(spr_n_a1, dtype=torch.float32)

        # training preparation

        # train action == 0
        input_a0 = s_n_a0_t
        output_a0 = spr_n_a0_t
        input_a1 = s_n_a1_t
        output_a1 = spr_n_a1_t
        # train_data_a0 = SimpleDataset(x=input_a0, y=output_a0)
        # train_data_a1 = SimpleDataset(x=input_a1, y=output_a1)
        # train_dataloader_a0 = DataLoader(train_data_a0,
        #                                  batch_size=len(train_data_a0),#128,
        #                                  shuffle=False)
        # train_dataloader_a1 = DataLoader(train_data_a1,
        #                                  batch_size=len(train_data_a1),#128,
        #                                  shuffle=False)

        self.model_a0_mean = smallNet(in_dim=self.s_dim,
                                      out_dim=self.s_dim + 1,
                                      hidden_dims=self.hidden_dims)
        self.model_a0_logvar = smallNet(in_dim=self.s_dim,
                                        out_dim=self.s_dim + 1,
                                        hidden_dims=self.hidden_dims)
        self.model_a1_mean = smallNet(in_dim=self.s_dim,
                                      out_dim=self.s_dim + 1,
                                      hidden_dims=self.hidden_dims)
        self.model_a1_logvar = smallNet(in_dim=self.s_dim,
                                        out_dim=self.s_dim + 1,
                                        hidden_dims=self.hidden_dims)

        optimizer_a0_mean = optim.Adam(self.model_a0_mean.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.l2_lambda)
        optimizer_a0_logvar = optim.Adam(self.model_a0_logvar.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.l2_lambda)
        optimizer_a1_mean = optim.Adam(self.model_a1_mean.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.l2_lambda)
        optimizer_a1_logvar = optim.Adam(self.model_a1_logvar.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.l2_lambda)
        loss_fn = nn.GaussianNLLLoss()
        self.model_a0_mean.train()
        self.model_a0_logvar.train()
        self.model_a1_mean.train()
        self.model_a1_logvar.train()
        self.epoch_candidate_losses = {i: 0 for i in self.epoch_candidates}
        # from torch.utils.tensorboard import SummaryWriter
        # ts_writer = SummaryWriter(
        #     os.path.join("/home/jitwang/TestSA2/ts_outs/"))
        for ep in range(self.max_epoches):
            optimizer_a0_mean.zero_grad()
            optimizer_a0_logvar.zero_grad()
            optimizer_a1_mean.zero_grad()
            optimizer_a1_logvar.zero_grad()
            # x0, y0 = next(iter(train_dataloader_a0))
            # x1, y1 = next(iter(train_dataloader_a1))
            x0, y0 = input_a0, output_a0
            x1, y1 = input_a1, output_a1
            train_loss_a0 = loss_fn(input=self.model_a0_mean.forward(x0),
                                    target=y0,
                                    var=torch.exp(
                                        self.model_a0_logvar.forward(x0)))
            train_loss_a1 = loss_fn(input=self.model_a1_mean.forward(x1),
                                    target=y1,
                                    var=torch.exp(
                                        self.model_a1_logvar.forward(x1)))
            train_loss = (train_loss_a0 * input_a0.shape[0] +
                          train_loss_a1 * input_a1.shape[0]) / (
                              input_a0.shape[0] + input_a1.shape[0])

            train_loss.backward()
            if ep % 2 == 0:
                optimizer_a0_mean.step()
                optimizer_a1_mean.step()
            else:
                optimizer_a0_logvar.step()
                optimizer_a1_logvar.step()
            if (ep + 1) % 10 == 0:
                mylogger.debug("epoch:{},train_loss:{},lr:{}".format(
                    ep + 1, train_loss.item(), self.lr))

            # if ep % 2 == 0:
            #     mylogger.info("{}".format(train_loss.item()))
            #     ts_writer.add_scalar('training loss_{}'.format(len(idxs_a0)),
            #                          train_loss.item(), ep)
        mylogger.debug("Finish training; epoch:{},train_loss:{}".format(
            ep + 1, train_loss.item()))

        mylogger.debug("Done training")

    def test_loss(self, s, a, sp, r):
        pass

    def sample(self, s, a, n=1):
        new_s = np.array(s).reshape([-1, self.s_dim])
        new_a = np.array(a).reshape([-1, 1])
        m = new_a.shape[0]

        new_s_n = (new_s - self.mean_s) / self.std_s
        idxs_a0 = (new_a == 0).flatten()
        idxs_a1 = (new_a == 1).flatten()
        sp_samples = np.zeros([m, n, self.s_dim])
        r_samples = np.zeros([m, n])
        self.model_a0_mean.eval()
        self.model_a1_mean.eval()
        self.model_a0_logvar.eval()
        self.model_a1_logvar.eval()
        # t0 = time.time()
        if np.sum(idxs_a0) > 0:
            m0 = np.sum(idxs_a0)
            with torch.no_grad():
                input_a0 = torch.tensor(new_s_n[idxs_a0], dtype=torch.float32)
                mean_spr_a0_n = self.model_a0_mean.forward(input_a0).numpy()
                var_spr_a0_n = torch.exp(
                    self.model_a0_logvar.forward(input_a0)).numpy()
            mean_spr_a0_n_flat = np.repeat(mean_spr_a0_n, repeats=n,
                                           axis=0).flatten()
            std_spr_a0_n_flat = np.sqrt(
                np.repeat(var_spr_a0_n, repeats=n, axis=0).flatten())
            samples_spr_a0_n = np.random.normal(mean_spr_a0_n_flat,
                                                std_spr_a0_n_flat).reshape(
                                                    m0 * n, self.s_dim + 1)
            samples_spr_a0 = samples_spr_a0_n * self.std_spr + self.mean_spr
            samples_spr_a0 = samples_spr_a0.reshape([m0, n, self.s_dim + 1])
            sp_samples[idxs_a0] = samples_spr_a0[:, :, :self.s_dim]
            r_samples[idxs_a0] = samples_spr_a0[:, :, self.s_dim]

        # t0 = time.time()
        if np.sum(idxs_a1) > 0:
            m1 = np.sum(idxs_a1)
            with torch.no_grad():
                input_a1 = torch.tensor(new_s_n[idxs_a1], dtype=torch.float32)
                mean_spr_a1_n = self.model_a1_mean.forward(input_a1).numpy()
                var_spr_a1_n = torch.exp(
                    self.model_a1_logvar.forward(input_a1)).numpy()
            mean_spr_a1_n_flat = np.repeat(mean_spr_a1_n, repeats=n,
                                           axis=0).flatten()
            std_spr_a1_n_flat = np.sqrt(
                np.repeat(var_spr_a1_n, repeats=n, axis=0).flatten())
            # t1 = time.time()
            samples_spr_a1_n = np.random.normal(
                mean_spr_a1_n_flat, std_spr_a1_n_flat).reshape(
                    m1 * n, self.s_dim + 1)  ## most time consuming part
            samples_spr_a1 = samples_spr_a1_n * self.std_spr + self.mean_spr
            samples_spr_a1 = samples_spr_a1.reshape([m1, n, self.s_dim + 1])
            sp_samples[idxs_a1] = samples_spr_a1[:, :, :self.s_dim]
            r_samples[idxs_a1] = samples_spr_a1[:, :, self.s_dim]

        return sp_samples, r_samples


class estimate_pt_original():
    def __init__(self, s, a, sp, r, lr, epoch_candidates, hidden_dims, s_test,
                 a_test, sp_test, r_test, **kwargs) -> None:
        assert (s.shape[0] == a.shape[0]) & (a.shape[0] == sp.shape[0]) & (
            sp.shape[0] == r.shape[0])
        self.s_dim = s.shape[-1]
        self.s = np.array(s).reshape(-1, self.s_dim)
        self.a = np.array(a).reshape(-1, 1)
        self.sp = np.array(sp).reshape(-1, self.s_dim)
        self.r = np.array(r).reshape(-1, 1)
        self.spr = np.concatenate([self.sp, self.r], axis=1)
        self.lr = lr
        self.epoch_candidates = epoch_candidates
        self.max_epoches = max(epoch_candidates)
        self.l2_lambda = 0  #1e-4
        self.hidden_dims = hidden_dims
        # normalization
        self.mean_s, self.std_s = np.mean(self.s, axis=0), np.std(self.s,
                                                                  axis=0)
        self.mean_spr, self.std_spr = np.mean(self.spr,
                                              axis=0), np.std(self.spr, axis=0)
        self.s_n = (self.s - self.mean_s) / self.std_s
        self.spr_n = (self.spr - self.mean_spr) / self.std_spr
        self.s_test = s_test
        self.a_test = a_test
        self.sp_test = sp_test
        self.r_test = r_test
        self.fit()

    def fit(self):

        # data preparation
        idxs_a0 = (self.a == 0).flatten()
        idxs_a1 = (self.a == 1).flatten()
        s_n_a0 = self.s_n[idxs_a0]
        s_n_a1 = self.s_n[idxs_a1]
        spr_n_a0 = self.spr_n[idxs_a0]
        spr_n_a1 = self.spr_n[idxs_a1]

        # tensorization
        s_n_a0_t = torch.tensor(s_n_a0, dtype=torch.float32)
        s_n_a1_t = torch.tensor(s_n_a1, dtype=torch.float32)
        spr_n_a0_t = torch.tensor(spr_n_a0, dtype=torch.float32)
        spr_n_a1_t = torch.tensor(spr_n_a1, dtype=torch.float32)

        # training preparation

        # train action == 0
        input_a0 = s_n_a0_t
        output_a0 = spr_n_a0_t
        input_a1 = s_n_a1_t
        output_a1 = spr_n_a1_t
        self.model_a0 = smallNet(in_dim=self.s_dim,
                                 out_dim=self.s_dim + 1,
                                 hidden_dims=self.hidden_dims)
        self.model_a1 = smallNet(in_dim=self.s_dim,
                                 out_dim=self.s_dim + 1,
                                 hidden_dims=self.hidden_dims)

        optimizer_a0 = optim.Adam(self.model_a0.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.l2_lambda)
        optimizer_a1 = optim.Adam(self.model_a1.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.l2_lambda)
        loss_fn = nn.MSELoss()
        self.model_a0.train()
        self.model_a1.train()
        self.epoch_candidate_losses = {i: 0 for i in self.epoch_candidates}
        for ep in range(self.max_epoches):
            self.model_a0.zero_grad()
            self.model_a1.zero_grad()
            train_loss_a0 = loss_fn(self.model_a0.forward(input_a0), output_a0)
            train_loss_a1 = loss_fn(self.model_a1.forward(input_a1), output_a1)
            train_loss = (train_loss_a0 * input_a0.shape[0] +
                          train_loss_a1 * input_a1.shape[0]) / (
                              input_a0.shape[0] + input_a1.shape[0])

            train_loss.backward()
            optimizer_a0.step()
            optimizer_a1.step()
            if (ep + 1) % 100 == 0:
                mylogger.debug("epoch:{},train_loss:{},lr:{}".format(
                    ep + 1, train_loss.item(), self.lr))
            if ((ep + 1) in self.epoch_candidates):
                self.epoch_candidate_losses[ep + 1] = self.test_loss(
                    s=self.s_test,
                    a=self.a_test,
                    sp=self.sp_test,
                    r=self.r_test)
        mylogger.debug("Finish training; epoch:{},train_loss:{}".format(
            ep + 1, train_loss.item()))

        # estimate cov of sp_n and r_n
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            self.model_a0.eval()
            self.model_a1.eval()
            nnout_a0 = self.model_a0.forward(s_n_a0_t)
            nnout_a1 = self.model_a1.forward(s_n_a1_t)
        self.cov_spr_a0_n = np.diag([
            loss_fn(nnout_a0[:, i], output_a0[:, i])
            for i in range(self.s_dim + 1)
        ])
        self.cov_spr_a1_n = np.diag([
            loss_fn(nnout_a1[:, i], output_a1[:, i])
            for i in range(self.s_dim + 1)
        ])

        mylogger.debug("Done training")

    def test_loss(self, s, a, sp, r):
        s = np.array(s).reshape(-1, self.s_dim)
        a = np.array(a).reshape(-1, 1)
        sp = np.array(sp).reshape(-1, self.s_dim)
        r = np.array(r).reshape(-1, 1)
        spr = np.concatenate([sp, r], axis=1)
        s_n = (s - self.mean_s) / self.std_s
        spr_n = (spr - self.mean_spr) / self.std_spr
        idxs_a0 = (a == 0).flatten()
        idxs_a1 = (a == 1).flatten()
        s_n_a0 = s_n[idxs_a0]
        s_n_a1 = s_n[idxs_a1]
        spr_n_a0 = spr_n[idxs_a0]
        spr_n_a1 = spr_n[idxs_a1]

        # tensorization
        s_n_a0_t = torch.tensor(s_n_a0, dtype=torch.float32)
        s_n_a1_t = torch.tensor(s_n_a1, dtype=torch.float32)
        spr_n_a0_t = torch.tensor(spr_n_a0, dtype=torch.float32)
        spr_n_a1_t = torch.tensor(spr_n_a1, dtype=torch.float32)

        # action == 0/1
        input_a0 = s_n_a0_t
        output_a0 = spr_n_a0_t
        input_a1 = s_n_a1_t
        output_a1 = spr_n_a1_t
        loss_fn = nn.MSELoss()
        self.model_a0.eval()
        self.model_a1.eval()
        with torch.no_grad():
            test_loss_a0 = loss_fn(self.model_a0.forward(input_a0), output_a0)
            test_loss_a1 = loss_fn(self.model_a1.forward(input_a1), output_a1)
            test_loss = (test_loss_a0 * input_a0.shape[0] +
                         test_loss_a1 * input_a1.shape[0]).numpy().item()
        return test_loss

    def sample(self, s, a, n=1):
        new_s = np.array(s).reshape([-1, self.s_dim])
        new_a = np.array(a).reshape([-1, 1])
        m = new_a.shape[0]

        new_s_n = (new_s - self.mean_s) / self.std_s
        idxs_a0 = (new_a == 0).flatten()
        idxs_a1 = (new_a == 1).flatten()
        sp_samples = np.zeros([m, n, self.s_dim])
        r_samples = np.zeros([m, n])
        self.model_a0.eval()
        self.model_a1.eval()
        # t0 = time.time()
        if np.sum(idxs_a0) > 0:
            m0 = np.sum(idxs_a0)
            with torch.no_grad():
                input_a0 = torch.tensor(new_s_n[idxs_a0], dtype=torch.float32)
                mean_spr_a0_n = self.model_a0.forward(input_a0).numpy()
            mean_spr_a0_n_flat = np.repeat(mean_spr_a0_n, repeats=n,
                                           axis=0).flatten()
            std_spr_a0_n_flat = np.sqrt(
                np.repeat(np.diag(self.cov_spr_a0_n).reshape(1, -1),
                          repeats=m0 * n,
                          axis=0).flatten())
            samples_spr_a0_n = np.random.normal(mean_spr_a0_n_flat,
                                                std_spr_a0_n_flat).reshape(
                                                    m0 * n, self.s_dim + 1)
            samples_spr_a0 = samples_spr_a0_n * self.std_spr + self.mean_spr
            samples_spr_a0 = samples_spr_a0.reshape([m0, n, self.s_dim + 1])
            sp_samples[idxs_a0] = samples_spr_a0[:, :, :self.s_dim]
            r_samples[idxs_a0] = samples_spr_a0[:, :, self.s_dim]

        # t0 = time.time()
        if np.sum(idxs_a1) > 0:
            m1 = np.sum(idxs_a1)
            with torch.no_grad():
                input_a1 = torch.tensor(new_s_n[idxs_a1], dtype=torch.float32)
                mean_spr_a1_n = self.model_a1.forward(input_a1).numpy()
            mean_spr_a1_n_flat = np.repeat(mean_spr_a1_n, repeats=n,
                                           axis=0).flatten()
            std_spr_a1_n_flat = np.sqrt(
                np.repeat(np.diag(self.cov_spr_a1_n).reshape(1, -1),
                          repeats=m1 * n,
                          axis=0).flatten())
            # t1 = time.time()
            samples_spr_a1_n = np.random.normal(
                mean_spr_a1_n_flat, std_spr_a1_n_flat).reshape(
                    m1 * n, self.s_dim + 1)  ## most time consuming part
            # mylogger.info("normal sample {} time usage:{}".format(
            # m1 * n,
            # time.time() - t1))
            samples_spr_a1 = samples_spr_a1_n * self.std_spr + self.mean_spr
            samples_spr_a1 = samples_spr_a1.reshape([m1, n, self.s_dim + 1])
            sp_samples[idxs_a1] = samples_spr_a1[:, :, :self.s_dim]
            r_samples[idxs_a1] = samples_spr_a1[:, :, self.s_dim]
            # mylogger.info("a1 time usage:{}".format(time.time() - t0))

        return sp_samples, r_samples


class estimate_w():
    def __init__(self, s, a, gmm_ncomponents=1) -> None:
        assert s.shape[0] == a.shape[0]
        self.s_dim = s.shape[-1]
        self.s = np.array(s).reshape([-1, self.s_dim])
        self.a = np.array(a).reshape([-1, 1])
        self.gmm_ncomponents = gmm_ncomponents
        self.fit()
        del self.s, self.a

    def fit(self):
        param_grid = {
            "n_components": range(1, 4),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }

        def gmm_bic_score(estimator, X):
            return -estimator.bic(X)

        grid_search = GridSearchCV(sklearn.mixture.GaussianMixture(),
                                   param_grid=param_grid,
                                   scoring=gmm_bic_score)
        grid_search.fit(self.s.reshape(-1, self.s_dim))
        df = pd.DataFrame(grid_search.cv_results_)[[
            "param_n_components", "param_covariance_type", "mean_test_score"
        ]]
        df = df.sort_values(by="mean_test_score",
                            ignore_index=True,
                            ascending=False)
        best_n_components = df.param_n_components[0]
        best_covariance_type = df.param_covariance_type[0]
        mylogger.debug(
            "w best_n_components:{}, best_covariance_type:{}".format(
                best_n_components, best_covariance_type))
        self.model_s = sklearn.mixture.GaussianMixture(
            n_components=best_n_components,
            covariance_type=best_covariance_type).fit(
                self.s.reshape(-1, self.s_dim))
        # self.model_s = sklearn.mixture.GaussianMixture(
        #     n_components=self.gmm_ncomponents,
        #     covariance_type="diag").fit(self.s.reshape(-1, self.s_dim))
        self.model_a_s = sklearn.linear_model.LogisticRegression().fit(
            self.s.reshape(-1, self.s_dim), self.a.reshape([
                -1,
            ]))
        # mylogger.debug("coefficients of logistic model:{}".format(
        #     str(self.model_a_s.coef_)))

    def density(self, s, a):
        new_s = np.array(s).reshape(-1, self.s_dim)
        new_a = np.array(a).reshape(-1, 1)
        p_s = np.exp(self.model_s.score_samples(new_s))
        probs = self.model_a_s.predict_proba(new_s)
        p_a_s = np.array(
            [probs[i, new_a.ravel()[i]] for i in range(len(probs))])
        mylogger.debug("p_a_s min:{}, max:{}".format(np.min(p_a_s),
                                                     np.max(p_a_s)))
        return (p_s * p_a_s)

    def sample(self, n=1):
        s_samples, _ = self.model_s.sample(n_samples=n)
        a_samples = np.random.binomial(
            1,
            self.model_a_s.predict_proba(s_samples)[:, 1])
        return (s_samples, a_samples)


if __name__ == "__main__":
    from gen_data import *
    torch.set_num_threads(1)
    S_DIM = 1
    T = 40
    N = 100
    TYPE = "hmoada"
    t = 20
    NUM_SAMPLES = 10000
    mySys = multiGaussionSys(S_DIM)
    S, A, R = mySys.simulate(T=T, N=N, type=TYPE)
    # w_models, sampler_sp_r = train_two_ml_models(S=S, A=A, R=R)

    # # w models check
    # S_fit, A_fit = w_models[t].sample(NUM_SAMPLES)
    # S_tmp, A_tmp, R_tmp = mySys.simulate(T=T, N=NUM_SAMPLES, type=TYPE)
    # S_real, A_real = S_tmp[:, t, :], A[:, t]

    # fig, ax = plt.subplots(nrows=S_DIM, ncols=2, figsize=[8, S_DIM * 4])
    # ax = ax.flatten()
    # for i in range(S_DIM):
    #     sns.kdeplot(S_fit[:, i], label="fit", ax=ax[2 * i])
    #     sns.kdeplot(S_real[:, i], label="real", ax=ax[2 * i])
    #     ax[2 * i].legend()
    # plt.savefig("est_w_check.png")

    # # sp_r_sampler check
    # t_var = np.repeat([t], repeats=NUM_SAMPLES, axis=0).reshape([-1, 1])
    # Sp_fit, R_fit = sampler_sp_r.sample(s=S_tmp[:, t, :],
    #                                     a=A_tmp[:, t],
    #                                     t=t_var,
    #                                     n=1)
    # Sp_real, R_real = S_tmp[:, t + 1, :], R_tmp[:, t]
    # fig, ax = plt.subplots(nrows=S_DIM, ncols=2, figsize=[8, S_DIM * 4])
    # ax = ax.flatten()
    # for i in range(S_DIM):
    #     sns.kdeplot(Sp_fit[:, 0, i], label="fit", ax=ax[2 * i])
    #     sns.kdeplot(Sp_real[:, i], label="real", ax=ax[2 * i])
    #     ax[2 * i].legend()
    #     sns.kdeplot(R_fit[:, 0], label="fit", ax=ax[2 * i + 1])
    #     sns.kdeplot(R_real, label="real", ax=ax[2 * i + 1])
    #     ax[2 * i + 1].legend()
    # plt.savefig("est_sampler_check.png")

    # linear model for pt
    pt_model = estimate_pt(s=S[:, :T, :].reshape([N, -1, 1]),
                           a=A[:, :T],
                           sp=S[:, 1:(T + 1), :].reshape([N, -1, 1]),
                           r=R[:, :T],
                           lr=0.001,
                           epoches=1000)
    print(pt_model.cov_spr_a0_n, pt_model.cov_spr_a1_n)
    print(pt_model.sample(s=[[1]], a=[[1]]))