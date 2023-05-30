import numpy as np
import sys, time, itertools, multiprocessing

sys.path.append("../")
from core.estimate_nuisance_parameters import estimate_w
from core.estimate_nuisance_parameters import estimate_pt
from core.random_h import *
from Utils.logger import *
from sklearn.model_selection import KFold
import pandas as pd
from scipy.stats import norm
from p_tqdm import p_imap


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def run_change_point_detection_one_repeat(S, A, R, M, B, ts, h_funs,
                                          learning_rate, pt_cv_selected,
                                          w_ncomponents, weight_clip_value,
                                          seed):
    # data preparation
    np.random.seed(seed)
    N, T = A.shape
    s_dim = S.shape[-1]
    S_n, R_n, _, _ = normalize_state_reward(S, R, S, R)
    folds = 2  # can not use dml since target parameter is not pathwise differentiable
    k_folds = KFold(n_splits=folds, shuffle=True, random_state=seed).split(S)
    # run change point detection
    idxs_train, idxs_test = next(k_folds)
    S_n1, A1, R_n1 = S_n[idxs_train], A[idxs_train], R_n[idxs_train]
    S_n2, A2, R_n2 = S_n[idxs_test], A[idxs_test], R_n[idxs_test]

    # allocate memory
    S2_normalized = np.zeros([len(ts), B])
    S2_res0s_normalized = [None for i in range(len(ts))]
    S2_res1s_normalized = [None for i in range(len(ts))]

    # first part is the same for two methods
    for i in range(len(ts)):
        t = ts[i]
        pt_hidden_dims = pt_cv_selected[i][
            0]  # [model0_topology,model1_topology]; examples of topology: [256,256]
        pt_epoches = pt_cv_selected[i][1]  # [model0_epochs,model1_epochs]
        w_models, pt_models = train_two_ml_models(
            S=S_n1,
            A=A1,
            R=R_n1,
            t=t,
            w_ncomponents=w_ncomponents,
            lr=learning_rate,
            pt_epoches=pt_epoches,
            pt_hidden_dims=pt_hidden_dims)
        g_fun = Mixture2Distribution(mixture1=w_models[0],
                                     mixture2=w_models[1],
                                     weight1=t / A1.shape[1],
                                     weight2=1 - t / A1.shape[1])
        S2_t_h_raw, S2_t_h_res0s, S2_t_h_res1s = calculate_S_t_h_doublerobust(
            S=S_n2,
            A=A2,
            R=R_n2,
            M=M,
            t=t,
            h_funs=h_funs,
            g_fun=g_fun,
            w_models=w_models,
            pt_models=pt_models,
            weight_clip_value=weight_clip_value)
        std_t_h = np.std(np.concatenate([S2_t_h_res0s, S2_t_h_res1s], axis=2),
                         axis=(1, 2))
        std_t_h_pooled = np.sqrt(
            np.var(S2_t_h_res0s, axis=(1, 2)) *
            (t - 1) + np.var(S2_t_h_res1s, axis=(1, 2)) *
            (T - t - 1)) / np.sqrt(T - 2)
        S2_t_h_normalized = np.mean(S2_t_h_raw, axis=1) / std_t_h
        S2_normalized[i] = S2_t_h_normalized * np.sqrt(t * (T - t) / T)
        S2_res0s_normalized[i] = S2_t_h_res0s / std_t_h[:, np.newaxis,
                                                        np.newaxis]
        S2_res1s_normalized[i] = S2_t_h_res1s / std_t_h[:, np.newaxis,
                                                        np.newaxis]

    # second part
    Gamma_normalized = np.max(S2_normalized)
    Gamma_bootstrap = calculate_boostrap(S2_res0s_normalized,
                                         S2_res1s_normalized, 5000)
    pvalue = np.mean(Gamma_normalized <= Gamma_bootstrap)

    return pvalue, Gamma_normalized


def run_change_point_detection_one_repeat_star(args):
    return run_change_point_detection_one_repeat(*args)


def run_change_point_detection(S,
                               A,
                               R,
                               M,
                               B,
                               ts,
                               htype,
                               learning_rate,
                               w_ncomponents,
                               weight_clip_value,
                               random_repeats,
                               cores,
                               seed,
                               pt_hidden_dims=None,
                               pt_epochs=None,
                               pvalue_combine_gamma=0.15):
    pt_cv_selected = {
        i: [[pt_hidden_dims, pt_hidden_dims], [pt_epochs, pt_epochs]]
        for i in range(len(ts))
    }

    if htype == "hybrid":
        h_funs = [
            random_h(s_dim=S.shape[-1], htype="nn") for _ in range(B - 1)
        ] + [random_h(s_dim=S.shape[-1], htype="reward")]
    else:
        h_funs = [random_h(s_dim=S.shape[-1], htype=htype) for _ in range(B)]

    if cores == 1:
        pvalues = []
        test_statistics = []
        for rep in range(random_repeats):
            out = run_change_point_detection_one_repeat(
                S=S,
                A=A,
                R=R,
                M=M,
                B=B,
                ts=ts,
                h_funs=h_funs,
                learning_rate=learning_rate,
                pt_cv_selected=pt_cv_selected,
                w_ncomponents=w_ncomponents,
                weight_clip_value=weight_clip_value,
                seed=seed + 10 * rep)
            pvalues.append(out[0])
            test_statistics.append(out[1])
    else:
        all_jobs = [(S, A, R, M, B, ts, h_funs, learning_rate, pt_cv_selected,
                     w_ncomponents, weight_clip_value, seed + rep)
                    for rep in range(random_repeats)]
        pvalues, test_statistics = zip(*list(
            p_imap(run_change_point_detection_one_repeat_star, all_jobs)))
    combined_pvalue = combine_multiple_p_values(
        pvalues=np.array(pvalues).flatten(), gamma=pvalue_combine_gamma)
    return combined_pvalue, pvalues, test_statistics


def train_two_ml_models(S, A, R, t, w_ncomponents, lr, pt_epoches,
                        pt_hidden_dims):
    torch.set_num_threads(1)
    N, T = A.shape
    s_dim = S.shape[2]
    state0 = S[:, :t].reshape([-1, s_dim])
    action0 = A[:, :t].reshape([-1, 1])
    next_state0 = S[:, 1:(t + 1)].reshape([-1, s_dim])
    reward0 = R[:, :t].reshape([-1, 1])
    state1 = S[:, t:T].reshape([-1, s_dim])
    action1 = A[:, t:T].reshape([-1, 1])
    next_state1 = S[:, (t + 1):(T + 1)].reshape([-1, s_dim])
    reward1 = R[:, t:T].reshape([-1, 1])
    w_models = [
        estimate_w(s=state0, a=action0, gmm_ncomponents=w_ncomponents),
        estimate_w(s=state1, a=action1, gmm_ncomponents=w_ncomponents)
    ]

    pt_model0 = estimate_pt(s=state0,
                            a=action0,
                            sp=next_state0,
                            r=reward0,
                            lr=lr,
                            epoch_candidates=[pt_epoches[0]],
                            hidden_dims=pt_hidden_dims[0],
                            s_test=state0,
                            a_test=action0,
                            sp_test=next_state0,
                            r_test=reward0)
    pt_model1 = estimate_pt(s=state1,
                            a=action1,
                            sp=next_state1,
                            r=reward1,
                            lr=lr,
                            epoch_candidates=[pt_epoches[1]],
                            hidden_dims=pt_hidden_dims[1],
                            s_test=state1,
                            a_test=action1,
                            sp_test=next_state1,
                            r_test=reward1)
    pt_models = [pt_model0, pt_model1]

    return (w_models, pt_models)


class Mixture2Distribution():

    def __init__(self, mixture1, mixture2, weight1, weight2) -> None:
        self.mixture1 = mixture1
        self.mixture2 = mixture2
        self.weight1 = weight1
        self.weight2 = weight2
        self.s_dim = self.mixture1.s_dim
        mylogger.debug("weight1:{},weight2:{}".format(self.weight1,
                                                      self.weight2))

    def density(self, s, a):
        new_s = np.array(s).reshape(-1, self.s_dim)
        new_a = np.array(a).reshape(-1, 1)

        # mixture component 1
        p1 = self.mixture1.density(new_s, new_a)

        # mixuture component 2
        p2 = self.mixture2.density(new_s, new_a)

        return self.weight1 * p1 + self.weight2 * p2

    def sample(self, n=1):
        idxs = np.random.binomial(1, p=self.weight2, size=[n])
        s_samples = np.zeros([n, self.s_dim])
        a_samples = np.zeros([n], dtype=int)

        if np.sum(idxs == 0) > 0:
            s_samples[idxs == 0], a_samples[idxs == 0] = self.mixture1.sample(
                n=np.sum(idxs == 0))

        if np.sum(idxs == 1) > 0:
            s_samples[idxs == 1], a_samples[idxs == 1] = self.mixture2.sample(
                n=np.sum(idxs == 1))
        return (s_samples, a_samples)


# define function to calculate delta(t;a,s)
def calculate_delta(s, a, h, pt_models, m):
    s_dim = s.shape[-1]
    s_samples = np.array(s).reshape([-1, s_dim])
    a_samples = np.array(a).reshape([-1, 1])
    N = s_samples.shape[0]
    t0 = time.time()
    sp_samples1, r_samples1 = pt_models[1].sample(s=s_samples,
                                                  a=a_samples,
                                                  n=m)  # N*m*sdim, N*m
    t0 = time.time()
    hs1 = h.call(sp_samples1.reshape([-1, s_dim]),
                 r_samples1.reshape([-1, 1])).reshape([-1, m])  #N*m

    del sp_samples1, r_samples1
    sp_samples0, r_samples0 = pt_models[0].sample(s=s_samples,
                                                  a=a_samples,
                                                  n=m)  # N*m*sdim, N*m
    hs0 = h.call(sp_samples0.reshape([-1, s_dim]),
                 r_samples0.reshape([-1, 1])).reshape([-1, m])  #N*m

    del sp_samples0, r_samples0
    hs1s = np.mean(hs1, axis=1)
    hs0s = np.mean(hs0, axis=1)
    return hs1s - hs0s, hs0s, hs1s


def calculate_S_t_h_doublerobust(S,
                                 A,
                                 R,
                                 M,
                                 t,
                                 h_funs,
                                 g_fun,
                                 w_models=None,
                                 pt_models=None,
                                 weight_clip_value=None,
                                 **kwargs):

    N, T = A.shape
    B = len(h_funs)
    s_dim = S.shape[2]
    if not weight_clip_value:
        weight_clip_value = 1e10

    g_s_a = g_fun

    S_raw = np.zeros([B, N])
    res0s = np.zeros([B, N, t])
    res1s = np.zeros([B, N, T - t])

    # calculate weight0 and weight1
    gs = g_s_a.density(s=S[:, :t, :].reshape([-1, s_dim]),
                       a=A[:, :t].reshape([-1, 1])).reshape([N, -1])  # N*t
    ws = w_models[0].density(s=S[:, :t, :].reshape([-1, s_dim]),
                             a=A[:, :t].reshape([-1, 1])).reshape([N,
                                                                   -1])  # N*t
    weight0 = gs / ws
    weight0 = np.clip(weight0, a_min=None, a_max=weight_clip_value)

    gs = g_s_a.density(s=S[:, t:T, :].reshape([-1, s_dim]),
                       a=A[:, t:T].reshape([-1, 1])).reshape([N,
                                                              -1])  # N*(T-t)
    ws = w_models[1].density(s=S[:, t:T, :].reshape([-1, s_dim]),
                             a=A[:, t:T].reshape([-1,
                                                  1])).reshape([N,
                                                                -1])  # N*(T-t)
    weight1 = gs / ws
    weight1 = np.clip(weight1, a_min=None, a_max=weight_clip_value)
    # mylogger.warning("{},{}".format(np.min(weight1), np.max(weight1)))
    s_samples_int, a_samples_int = g_s_a.sample(n=int(np.max([t, T - t]) * N))

    # estimate S_t_h for each h and t
    for i in range(B):
        h_fun = h_funs[i]
        deltas, _, _ = calculate_delta(s=s_samples_int,
                                       a=a_samples_int,
                                       h=h_fun,
                                       pt_models=pt_models,
                                       m=M)
        tmp = np.mean(np.abs(deltas))
        integral = np.repeat(tmp, repeats=N)
        del tmp

        # 2. Estimate first sign part
        # a) calculate sign delta
        deltas, hs0s, hs1s = calculate_delta(s=S[:,
                                                 t:T, :].reshape([-1, s_dim]),
                                             a=A[:, t:T].reshape([-1, 1]),
                                             h=h_fun,
                                             pt_models=pt_models,
                                             m=M)
        deltas = deltas.reshape([N, -1])
        sgn_deltas = np.sign(deltas)  # N*(T-t)

        # b) calculate h_sp_r
        h_sp_r = h_fun.call(sp=S[:, (t + 1):(T + 1), :].reshape([-1, s_dim]),
                            r=R[:, t:T]).reshape([N, -1])  # N*(T-t)

        # c) calculate exp_h_sp_r
        exp_h_sp_r = hs1s.reshape([N, T - t])  # N*(T-t)
        del hs0s, hs1s

        # e) calculate aug1 and res1
        res1 = sgn_deltas * (h_sp_r - exp_h_sp_r) * weight1
        aug1 = np.mean(res1, axis=1)

        # 3. Estimate second sign part
        # a) calculate sign delta
        deltas, hs0s, hs1s = calculate_delta(s=S[:, :t, :].reshape([-1,
                                                                    s_dim]),
                                             a=A[:, :t].reshape([-1, 1]),
                                             h=h_fun,
                                             pt_models=pt_models,
                                             m=M)
        deltas = deltas.reshape([N, -1])
        sgn_deltas = np.sign(deltas)  # N*t

        # b) calculate h_sp_r
        h_sp_r = h_fun.call(sp=S[:, 1:(t + 1), :].reshape([-1, s_dim]),
                            r=R[:, :t]).reshape([N, -1])  # N*t

        # c) calculate exp_h_sp_r
        exp_h_sp_r = hs0s.reshape([N, t])  # N*t
        del hs0s, hs1s

        # e) calculate aug0 and res0
        res0 = sgn_deltas * (h_sp_r - exp_h_sp_r) * weight0
        aug0 = np.mean(res0, axis=1)  # [N,]

        # 4. estimate S given h and t
        S_raw_b = integral + (aug1 - aug0)  # N,

        S_raw[i] = S_raw_b
        res0s[i, :, :] = res0
        res1s[i, :, :] = res1

    return S_raw, res0s, res1s


def calculate_boostrap(res0s, res1s, J):
    len_ts = len(res0s)
    B = res0s[0].shape[0]
    N = res0s[0].shape[1]
    T = res0s[0].shape[-1] + res1s[0].shape[-1]

    statistic_boots = np.zeros([J])
    for j in range(J):
        tmp = np.zeros([len_ts, B])
        e = np.random.normal(0, 1, size=[N, T])
        for i in range(len_ts):
            t = res0s[i].shape[-1]
            eB = np.repeat(e[np.newaxis, :, :], B, axis=0)
            res0b = res0s[i] * eB[:, :, :t]
            res1b = res1s[i] * eB[:, :, t:T]
            augB = np.mean(res1b, axis=2) - np.mean(res0b, axis=2)
            tmp[i, :] = np.mean(augB, axis=1) * np.sqrt(t * (T - t) / T)
        statistic_boots[j] = np.max(tmp)
    return statistic_boots


def combine_multiple_p_values(pvalues, gamma=0.15):
    if len(pvalues) == 1:
        return pvalues[0]
    else:
        return np.min([1, np.quantile(np.array(pvalues) / gamma, gamma)])


def normalize_state_reward(S1, R1, S2, R2):
    N1, T = R1.shape
    sdim = S1.shape[-1]
    N2 = R2.shape[1]
    state_mean = np.mean(S1, axis=(0, 1))
    state_sd = np.std(S1, axis=(0, 1))
    reward_mean = np.mean(R1, axis=(0, 1))
    reward_std = np.std(R1, axis=(0, 1))

    S1_n = (S1 - state_mean) / state_sd
    S2_n = (S2 - state_mean) / state_sd
    R1_n = (R1 - reward_mean) / reward_std
    R2_n = (R2 - reward_mean) / reward_std

    return S1_n, R1_n, S2_n, R2_n


if __name__ == "__main__":
    pass
