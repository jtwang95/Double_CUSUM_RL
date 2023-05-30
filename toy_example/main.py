import numpy as np
import pandas as pd
import time

P1, P2 = 0.8, 0.8
P3, P4 = 0.7, 0.7


def gen_data(T, N, type, seed):
    np.random.seed(seed)
    S, A, R = np.zeros([N, T + 1],
                       dtype=int), np.zeros([N, T],
                                            dtype=int), np.zeros([N, T],
                                                                 dtype=int)
    S[:, 0] = np.random.binomial(n=1, p=0.5, size=N)
    if type == "hmo":
        for t in range(T):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=P1,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=P2, size=N)) + 0
    if type == "pwc2_reward":
        CHGPT = int(T / 2)
        for t in range(CHGPT):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=0.8,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=0.8, size=N)) + 0
        for t in range(CHGPT, T):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=0.8,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=0.6, size=N)) + 0
    if type == "pwc2_trans":
        CHGPT = int(T / 2)
        for t in range(CHGPT):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=0.8,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=0.8, size=N)) + 0
        for t in range(CHGPT, T):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=0.6,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=0.8, size=N)) + 0
    if type == "pwc2_both":
        CHGPT = int(T / 2)
        for t in range(CHGPT):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=P1,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=P2, size=N)) + 0
        for t in range(CHGPT, T):
            A[:, t] = np.random.binomial(n=1, p=0.5, size=N)
            real_action = (A[:, t] == np.random.binomial(n=1, p=P3,
                                                         size=N)) + 0
            S[:, t + 1] = (S[:, t] == real_action) + 0
            R[:, t] = (S[:, t] == np.random.binomial(n=1, p=P4, size=N)) + 0
    return S, A, R


def random_h(seed):
    np.random.seed(seed)
    return np.random.normal(0, 1, size=[2, 2])


def estimate_pt(S, A, R, t):
    N, T = A.shape
    d0 = {
        0: {
            0: np.zeros([2, 2]),
            1: np.zeros([2, 2])
        },
        1: {
            0: np.zeros([2, 2]),
            1: np.zeros([2, 2])
        }
    }
    for j in range(t):
        for i in range(N):
            d0[S[i, j]][A[i, j]][S[i, j + 1], R[i, j]] += 1
    d1 = {
        0: {
            0: np.zeros([2, 2]),
            1: np.zeros([2, 2])
        },
        1: {
            0: np.zeros([2, 2]),
            1: np.zeros([2, 2])
        }
    }
    for j in range(t, T):
        for i in range(N):
            d1[S[i, j]][A[i, j]][S[i, j + 1], R[i, j]] += 1

    ## normalize
    for i in [0, 1]:
        for j in [0, 1]:
            d0[i][j] = d0[i][j] / np.sum(d0[i][j])
            d1[i][j] = d1[i][j] / np.sum(d1[i][j])
    return [d0, d1]


def estimate_w(S, A, R, t):
    N, T = A.shape
    a0 = np.zeros([2, 2])
    for j in range(t):
        for i in range(N):
            a0[S[i, j]][A[i, j]] += 1
    a1 = np.zeros([2, 2])
    for j in range(t, T):
        for i in range(N):
            a1[S[i, j]][A[i, j]] += 1

    a0 = a0 / np.sum(a0)
    a1 = a1 / np.sum(a1)

    return [a0, a1]


def calculate_exphs(s, a, pt_model, h):
    exphs = 0.0
    for sp in [0, 1]:
        for r in [0, 1]:
            exphs += h[sp, r] * pt_model[s][a][sp, r]
    return exphs


def calculate_test_statistic(S, A, R, t, pt_models, w_models, h):
    N, T = A.shape
    g_s_a = np.array([[0.4, 0.3], [0.1, 0.2]])

    exphs0_table = np.zeros([2, 2])  # function of (s,a)
    for s, a in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        exphs0_table[s, a] = calculate_exphs(s, a, pt_models[0], h)
    exphs1_table = np.zeros([2, 2])  # function of (s,a)
    for s, a in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        exphs1_table[s, a] = calculate_exphs(s, a, pt_models[1], h)
    delta_table = exphs1_table - exphs0_table  # function of (s,a)

    # integral
    integral = 0.0
    for s, a in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        integral += np.abs(delta_table[s, a]) * g_s_a[s, a]

    # first aug part
    res1 = np.zeros([N, T - t])
    for n in range(N):
        for i in range(t, T):
            s, a, sp, r = S[n, i], A[n, i], S[n, i + 1], R[n, i]
            res1[n, i - t] = np.sign(delta_table[s, a] +
                                     np.random.normal(0, 0.00000001, 1)) * (
                                         h[sp, r] - exphs1_table[s, a]
                                     ) * g_s_a[s, a] / w_models[1][s, a]
    aug1 = np.mean(res1, axis=1)

    # second aug part
    res0 = np.zeros([N, t])
    for n in range(N):
        for i in range(0, t):
            s, a, sp, r = S[n, i], A[n, i], S[n, i + 1], R[n, i]
            res0[n, i] = np.sign(delta_table[s, a] +
                                 np.random.normal(0, 0.00000001, 1)) * (
                                     h[sp, r] - exphs0_table[s, a]
                                 ) * g_s_a[s, a] / w_models[0][s, a]
    aug0 = np.mean(res0, axis=1)

    # average over all samples
    raw_S = integral + aug1 - aug0
    scaled_S = raw_S / np.std(raw_S, ddof=1) * np.sqrt(t * (T - t) / T)
    return np.mean(scaled_S), res0, res1


def calculate_test_statistic_bootstrap(res0s, res1s, J=1000):
    B, N, t = res0s.shape
    T = res0s.shape[-1] + res1s.shape[-1]
    S_t_boots = np.zeros([J]) * 1.0
    for j in range(J):
        S_h = np.zeros([B]) * 1.0
        e0 = np.random.normal(0, 1, size=res0s[0].shape)
        e1 = np.random.normal(0, 1, size=res1s[0].shape)
        e0B = np.repeat(e0[np.newaxis, :, :], B, axis=0)
        e1B = np.repeat(e1[np.newaxis, :, :], B, axis=0)
        aug0B = np.mean(res0s * e0B, axis=2)
        aug1B = np.mean(res1s * e1B, axis=2)
        raw_SB = aug1B - aug0B
        S_h = np.mean(raw_SB, axis=1) / np.std(
            raw_SB, axis=1, ddof=1) * np.sqrt(t * (T - t) / T)
        S_t_boots[j] = np.max(S_h)
    return S_t_boots


def misspecify_pt_model(pt_model, alpha=0.5, noise_type="s1"):
    if noise_type == "true":
        return pt_model
    if noise_type == "s1":
        noise_model = {
            0: {
                0: np.array([[0.45, 0.05], [0.45, 0.05]]),
                1: np.array([[0.45, 0.05], [0.45, 0.05]])
            },
            1: {
                0: np.array([[0.45, 0.05], [0.45, 0.05]]),
                1: np.array([[0.25, 0.25], [0.25, 0.25]])
            }
        }
    if noise_type == "s2":
        noise_model = {
            0: {
                0: np.array([[0.01, 0.49], [0.05, 0.45]]),
                1: np.array([[0.05, 0.45], [0.05, 0.45]])
            },
            1: {
                0: np.array([[0.05, 0.45], [0.05, 0.45]]),
                1: np.array([[0.05, 0.15], [0.05, 0.75]])
            }
        }
    noisy_pt_model = {
        0: {
            0: np.array([[0, 0], [0, 0]]),
            1: np.array([[0, 0], [0, 0]])
        },
        1: {
            0: np.array([[0, 0], [0, 0]]),
            1: np.array([[0, 0], [0, 0]])
        }
    }
    for s, a in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        noisy_pt_model[s][a] = alpha * noise_model[s][a] + (
            1 - alpha) * pt_model[s][a]
    return noisy_pt_model


def misspecify_w_model(w_model, alpha=0.5, noise_type="s1"):
    if noise_type == "true":
        return w_model
    if noise_type == "s1":
        noise_model = np.array([[0.2, 0.3], [0.3, 0.2]])
    if noise_type == "s2":
        noise_model = np.array([[0.4, 0.2], [0.2, 0.2]])
    noisy_w_model = np.zeros_like(noise_model)
    for s, a in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        noisy_w_model[s][a] = alpha * noise_model[s][a] + (
            1 - alpha) * w_model[s][a]
    return noisy_w_model


def combine_multiple_p_values(pvalues, gamma=0.1):
    if len(pvalues) == 1:
        return pvalues[0]
    else:
        return np.min([
            1,
            np.quantile(np.array(pvalues) / gamma,
                        gamma,
                        interpolation="nearest")
        ])


from joblib import Parallel, delayed
import multiprocessing
from functools import partial


def run_dr(type, T, N, B, TS, J, CROSS_FIT, CROSS_FOLD, REPS, PVALUE_THRESHOLD,
           ALPHA, PT_NOISE_TYPES, W_NOISE_TYPES, N_THRESHOLDS):
    if type == "hmo":
        true_pt_model = {
            0: {
                0:
                np.array([[(1 - P1) * P2, (1 - P1) * (1 - P2)],
                          [P1 * P2, P1 * (1 - P2)]]),
                1:
                np.array([[P1 * P2, P1 * (1 - P2)],
                          [(1 - P1) * P2, (1 - P1) * (1 - P2)]])
            },
            1: {
                0:
                np.array([[P1 * (1 - P2), P1 * P2],
                          [(1 - P1) * (1 - P2), (1 - P1) * P2]]),
                1:
                np.array([[(1 - P1) * (1 - P2), (1 - P1) * P2],
                          [P1 * (1 - P2), P1 * P2]])
            }
        }
        true_w_model = np.array([[0.25, 0.25], [0.25, 0.25]])
        true_pt_models = [true_pt_model, true_pt_model]
        true_w_models = [true_w_model, true_w_model]
    if type == "pwc2_reward":
        true_pt_models = [{
            0: {
                0: np.array([[0.16, 0.04], [0.64, 0.16]]),
                1: np.array([[0.64, 0.16], [0.16, 0.04]])
            },
            1: {
                0: np.array([[0.16, 0.64], [0.04, 0.16]]),
                1: np.array([[0.04, 0.16], [0.16, 0.64]])
            }
        }, {
            0: {
                0: np.array([[0.10, 0.10], [0.40, 0.40]]),
                1: np.array([[0.40, 0.40], [0.10, 0.10]])
            },
            1: {
                0: np.array([[0.40, 0.40], [0.10, 0.10]]),
                1: np.array([[0.10, 0.10], [0.40, 0.40]])
            }
        }]
        true_w_models = [
            np.array([[0.25, 0.25], [0.25, 0.25]]),
            np.array([[0.25, 0.25], [0.25, 0.25]])
        ]
    if type == "pwc2_both":
        true_pt_models = [{
            0: {
                0:
                np.array([[(1 - P1) * P2, (1 - P1) * (1 - P2)],
                          [P1 * P2, P1 * (1 - P2)]]),
                1:
                np.array([[P1 * P2, P1 * (1 - P2)],
                          [(1 - P1) * P2, (1 - P1) * (1 - P2)]])
            },
            1: {
                0:
                np.array([[P1 * (1 - P2), P1 * P2],
                          [(1 - P1) * (1 - P2), (1 - P1) * P2]]),
                1:
                np.array([[(1 - P1) * (1 - P2), (1 - P1) * P2],
                          [P1 * (1 - P2), P1 * P2]])
            }
        }, {
            0: {
                0:
                np.array([[(1 - P3) * P4, (1 - P3) * (1 - P4)],
                          [P3 * P4, P3 * (1 - P4)]]),
                1:
                np.array([[P3 * P4, P3 * (1 - P4)],
                          [(1 - P3) * P4, (1 - P3) * (1 - P4)]])
            },
            1: {
                0:
                np.array([[P3 * (1 - P4), P3 * P4],
                          [(1 - P3) * (1 - P4), (1 - P3) * P4]]),
                1:
                np.array([[(1 - P3) * (1 - P4), (1 - P3) * P4],
                          [P3 * (1 - P4), P3 * P4]])
            }
        }]
        true_w_models = [
            np.array([[0.25, 0.25], [0.25, 0.25]]),
            np.array([[0.25, 0.25], [0.25, 0.25]])
        ]
    if type == "pwc2_trans":
        true_pt_models = [{
            0: {
                0: np.array([[0.16, 0.04], [0.64, 0.16]]),
                1: np.array([[0.64, 0.16], [0.16, 0.04]])
            },
            1: {
                0: np.array([[0.16, 0.64], [0.04, 0.16]]),
                1: np.array([[0.04, 0.16], [0.16, 0.64]])
            }
        }, {
            0: {
                0: np.array([[0.32, 0.08], [0.48, 0.12]]),
                1: np.array([[0.48, 0.12], [0.32, 0.08]])
            },
            1: {
                0: np.array([[0.12, 0.48], [0.08, 0.32]]),
                1: np.array([[0.08, 0.32], [0.12, 0.48]])
            }
        }]
        true_w_models = [
            np.array([[0.25, 0.25], [0.25, 0.25]]),
            np.array([[0.25, 0.25], [0.25, 0.25]])
        ]

    input = list(range(REPS))  # arbitrary list
    job = partial(one_run_dr,
                  type=type,
                  T=T,
                  N=N,
                  B=B,
                  TS=TS,
                  J=J,
                  CROSS_FIT=CROSS_FIT,
                  CROSS_FOLD=CROSS_FOLD,
                  true_pt_models=true_pt_models,
                  true_w_models=true_w_models,
                  ALPHA=ALPHA,
                  PT_NOISE_TYPES=PT_NOISE_TYPES,
                  W_NOISE_TYPES=W_NOISE_TYPES)
    # arg1 is being fetched from input list
    output = Parallel(n_jobs=N_THRESHOLDS)(delayed(job)(i * 42) for i in input)
    pvalues = np.array([v[0] for v in output])

    return np.mean(pvalues < PVALUE_THRESHOLD)


def one_run_dr(seed, type, T, N, B, TS, J, CROSS_FIT, CROSS_FOLD,
               true_pt_models, true_w_models, ALPHA, PT_NOISE_TYPES,
               W_NOISE_TYPES):
    # print(psudo_arg)
    S, A, R = gen_data(T, 2 * N, type, seed)
    idxs = np.array(range(N * 2))
    pvalues = []
    t_stars = []
    stat_stars = []
    for fold in range(CROSS_FOLD):
        np.random.shuffle(idxs)
        idxs_train = idxs[:N]
        idxs_test = idxs[N:]

        S1, A1, R1 = S[idxs_train], A[idxs_train], R[idxs_train]
        S2, A2, R2 = S[idxs_test], A[idxs_test], R[idxs_test]
        S2, A2, R2 = S, A, R
        h_funs = [random_h(seed=b) for b in range(B)]
        S_t = np.zeros_like(TS) * 1.0
        S_t_boots = np.zeros([len(TS), J]) * 1.0
        for i in range(len(TS)):
            t = TS[i]
            S_h = np.zeros([B])
            res0s = np.zeros([B, S2.shape[0], t])
            res1s = np.zeros([B, S2.shape[0], T - t])
            for b in range(B):
                h = h_funs[b]
                pt_models = [
                    misspecify_pt_model(true_pt_models[0],
                                        alpha=ALPHA,
                                        noise_type=PT_NOISE_TYPES[0]),
                    misspecify_pt_model(true_pt_models[1],
                                        alpha=ALPHA,
                                        noise_type=PT_NOISE_TYPES[1])
                ]
                w_models = [
                    misspecify_w_model(true_w_models[0],
                                       alpha=ALPHA,
                                       noise_type=W_NOISE_TYPES[0]),
                    misspecify_w_model(true_w_models[1],
                                       alpha=ALPHA,
                                       noise_type=W_NOISE_TYPES[1])
                ]
                # calculate test statistic
                stat, res0, res1 = calculate_test_statistic(
                    S2, A2, R2, t, pt_models, w_models, h)
                S_h[b] = stat
                res0s[b] = res0
                res1s[b] = res1
            S_t[i] = np.max(S_h)
            S_t_boots[i] = calculate_test_statistic_bootstrap(res0s, res1s, J)
        # take maximum over all t in TS
        S_stat = np.max(S_t)
        S_boots = np.max(S_t_boots, axis=0)
        pvalue = np.mean(S_stat < S_boots)
        pvalues.append(pvalue)
        t_stars.append(TS[np.argmax(S_t)])
        stat_stars.append(S_stat)
        if not CROSS_FIT:
            break
    combined_pvalue = combine_multiple_p_values(pvalues, gamma=0.15)
    return combined_pvalue, ",".join([str(i) for i in t_stars])


def batch_run(type, pt_noise_types, w_noise_types, alpha):
    TYPE = type
    T = 10
    N = 30
    B = 25
    TS = [3, 4, 5, 6, 7, 8]
    J = 1000
    CROSS_FIT = False
    CROSS_FOLD = 1
    REPS = 500
    PVALUE_THRESHOLD = 0.05
    PT_NOISE_TYPES = pt_noise_types
    W_NOISE_TYPES = w_noise_types
    ALPHA = [alpha]
    N_THRESHOLDS = 1

    res = pd.DataFrame({})
    for alpha in ALPHA:
        t0 = time.time()
        out = run_dr(TYPE, T, N, B, TS, J, CROSS_FIT, CROSS_FOLD, REPS,
                     PVALUE_THRESHOLD, alpha, PT_NOISE_TYPES, W_NOISE_TYPES,
                     N_THRESHOLDS)
        t1 = time.time()
    return out


if __name__ == "__main__":

    import itertools
    np.random.seed(21)

    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    res_hmo = expand_grid({
        "type": ["hmo"],
        "pt_noise_type": ["true,true", "s1,s2"],
        "w_noise_type": ["true,true", "s1,s2"],
        "alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
    })

    res_pwc2 = expand_grid({
        "type": ["pwc2_both"],
        "pt_noise_type": ["true,true", "s1,s1"],
        "w_noise_type": ["true,true", "s1,s1"],
        "alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
    })

    res = pd.concat([res_pwc2, res_hmo]).reset_index()
    print(res)

    for index, row in res.iterrows():
        t0 = time.time()
        print("{}/{} --- Running configuration {}".format(
            index + 1, res.shape[0], str(row.to_dict())))
        pt_noise_type = row["pt_noise_type"].split(",")
        w_noise_type = row["w_noise_type"].split(",")
        rej_prob = batch_run(row["type"],
                             pt_noise_type,
                             w_noise_type,
                             alpha=row["alpha"])
        t1 = time.time()
        res.at[index, "rej_prob"] = rej_prob
        res.at[index, "time"] = "{:.2f}".format(t1 - t0)
        try:
            print("Done;Reject probability:{};Time used:{}.".format(
                res.at[index, "rej_prob"].values[0], res.at[index,
                                                            "time"].values[0]))
        except:
            print("Done;Reject probability:{};Time used:{}.".format(
                res.at[index, "rej_prob"], res.at[index, "time"]))
    res.to_csv("res.csv")