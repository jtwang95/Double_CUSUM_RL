import numpy as np
import itertools
from sklearn.model_selection import KFold
from p_tqdm import p_imap


class estimate_pt_discrete:

    def __init__(self, s, a, r, sp, num_states, num_actions,
                 num_rewards) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.probability_matrix = self._count_proportion(s, a, r, sp)

    def _state_action_to_index(self, s, a):
        return a * self.num_states + s

    def _state_reward_to_index(self, sp, r):
        return r * self.num_states + sp

    def _count_proportion(self, s, a, r, sp):
        occurence_matrix = np.zeros([
            self.num_states * self.num_actions,
            self.num_states * self.num_rewards
        ])
        s = np.array(s).reshape([-1])
        a = np.array(a).reshape([-1])
        r = np.array(r).reshape([-1])
        sp = np.array(sp).reshape([-1])

        for i in range(len(s)):
            occurence_matrix[self._state_action_to_index(s[i], a[i]),
                             self._state_reward_to_index(sp[i], r[i])] += 1
        normalization_matrix = np.repeat(np.sum(occurence_matrix,
                                                axis=1)[:, np.newaxis],
                                         repeats=occurence_matrix.shape[1],
                                         axis=1)
        normalization_matrix[normalization_matrix == 0] = 1
        probability_matrix = occurence_matrix / normalization_matrix
        return probability_matrix

    def get_probability_vector(self, s, a):
        return self.probability_matrix[self._state_action_to_index(s, a), :]


class estimate_w_discrete:

    def __init__(self, s, a, num_states, num_actions, epsilon=0.001) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.probability_vector = self._count_proportion(s, a, epsilon)

    def _state_action_to_index(self, s, a):
        return a * self.num_states + s

    def _count_proportion(self, s, a, epsilon):
        occurence_vector = np.zeros([self.num_states * self.num_actions])
        s = np.array(s).reshape([-1])
        a = np.array(a).reshape([-1])

        for i in range(len(s)):
            occurence_vector[self._state_action_to_index(s[i], a[i])] += 1
        probability_vector = occurence_vector / np.sum(occurence_vector)
        probability_vector[probability_vector == 0.0] = epsilon
        return probability_vector

    def get_probability(self, s, a):
        return self.probability_vector[self._state_action_to_index(s, a)]

    def get_probability_vector(self):
        return self.probability_vector


class Mixture2Distribution():

    def __init__(self, mixture1, mixture2, weight1, weight2) -> None:
        self.mixture1 = mixture1
        self.mixture2 = mixture2
        self.weight1 = weight1
        self.weight2 = weight2

    def get_probability(self, s, a):
        p1 = self.mixture1.get_probability(s, a)
        p2 = self.mixture2.get_probability(s, a)
        return self.weight1 * p1 + self.weight2 * p2

    def get_probability_vector(self):
        return self.weight1 * self.mixture1.get_probability_vector(
        ) + self.weight2 * self.mixture2.get_probability_vector()


class random_h:

    def __init__(self, num_states, num_rewards) -> None:
        self.num_states = num_states
        self.num_rewards = num_rewards
        self.h_vector = np.random.normal(0, 1, size=[num_states * num_rewards])

    def _state_reward_to_index(self, sp, r):
        return r * self.num_states + sp

    def get_hvalue(self, sp, r):
        return self.h_vector[self._state_reward_to_index(sp, r)]

    def get_hvalue_vector(self):
        return self.h_vector


def calculate_exphs(s, a, pt_model, h):
    probability_vector = pt_model.get_probability_vector(s=s, a=a)
    h_vector = h.get_hvalue_vector()
    return np.sum(probability_vector * h_vector)


def train_nuisance_models(S, A, R, t, num_states, num_actions, num_rewards):
    N, T = A.shape
    state0 = S[:, :t].reshape([-1, 1])
    action0 = A[:, :t].reshape([-1, 1])
    next_state0 = S[:, 1:(t + 1)].reshape([-1, 1])
    reward0 = R[:, :t].reshape([-1, 1])
    state1 = S[:, t:T].reshape([-1, 1])
    action1 = A[:, t:T].reshape([-1, 1])
    next_state1 = S[:, (t + 1):(T + 1)].reshape([-1, 1])
    reward1 = R[:, t:T].reshape([-1, 1])
    w_models = [
        estimate_w_discrete(s=state0,
                            a=action0,
                            num_states=num_states,
                            num_actions=num_actions),
        estimate_w_discrete(s=state1,
                            a=action1,
                            num_states=num_states,
                            num_actions=num_actions)
    ]
    pt_models = [
        estimate_pt_discrete(s=state0,
                             a=action0,
                             r=reward0,
                             sp=next_state0,
                             num_states=num_states,
                             num_actions=num_actions,
                             num_rewards=num_rewards),
        estimate_pt_discrete(s=state1,
                             a=action1,
                             r=reward1,
                             sp=next_state1,
                             num_states=num_states,
                             num_actions=num_actions,
                             num_rewards=num_rewards)
    ]

    return (w_models, pt_models)


def calculate_S_one_t_all_h(S, A, R, t, pt_models, w_models, g_fun, h_funs,
                            num_states, num_actions, num_rewards,
                            weight_clip_value):

    def state_action_to_index(s, a):
        return a * num_states + s

    N, T = A.shape
    B = len(h_funs)

    weight0_vector = np.zeros([num_states * num_actions])
    weight1_vector = np.zeros([num_states * num_actions])
    for s, a in itertools.product(range(num_states), range(num_actions)):
        weight0_vector[state_action_to_index(s, a)] = g_fun.get_probability(
            s, a) / w_models[0].get_probability(s, a)
        weight1_vector[state_action_to_index(s, a)] = g_fun.get_probability(
            s, a) / w_models[1].get_probability(s, a)
    if not weight_clip_value:
        weight_clip_value = 1e10
    weight0_vector = np.clip(weight0_vector,
                             a_min=None,
                             a_max=weight_clip_value)
    weight1_vector = np.clip(weight1_vector,
                             a_min=None,
                             a_max=weight_clip_value)

    res0 = np.zeros([B, N, t])
    res1 = np.zeros([B, N, T - t])
    scaled_res0 = np.zeros_like(res0)
    scaled_res1 = np.zeros_like(res1)
    scaled_S = np.zeros([B])
    for b in range(B):
        h = h_funs[b]
        exphs0_vector = np.ones([num_states * num_actions
                                 ])  # function of (s,a)
        for s, a in itertools.product(range(num_states), range(num_actions)):
            exphs0_vector[state_action_to_index(s, a)] = calculate_exphs(
                s, a, pt_models[0], h)
        exphs1_vector = np.ones([num_states * num_actions
                                 ])  # function of (s,a)
        for s, a in itertools.product(range(num_states), range(num_actions)):
            exphs1_vector[state_action_to_index(s, a)] = calculate_exphs(
                s, a, pt_models[1], h)
        delta_vector = exphs1_vector - exphs0_vector  # function of (s,a)

        # integral
        integral = np.sum(
            np.abs(delta_vector) * g_fun.get_probability_vector())

        # first aug part
        for n in range(N):
            s_vector, a_vector, sp_vector, r_vector = S[n, t:T].flatten(), A[
                n,
                t:T].flatten(), S[n,
                                  (t + 1):(T + 1)].flatten(), R[n,
                                                                t:T].flatten()
            sgn_delta_vector = np.sign(delta_vector[state_action_to_index(
                s_vector, a_vector)])  # (T-t) * 1
            h_values_vector = h.get_hvalue(sp_vector, r_vector)
            exphs_values_vector = exphs1_vector[state_action_to_index(
                s_vector, a_vector)]
            weight_values_vector = weight1_vector[state_action_to_index(
                s_vector, a_vector)]
            res1[b, n] = sgn_delta_vector * (
                h_values_vector - exphs_values_vector) * weight_values_vector

        # second aug part
        for n in range(N):
            s_vector, a_vector, sp_vector, r_vector = S[n, 0:t].flatten(), A[
                n, 0:t].flatten(), S[n, 1:(t + 1)].flatten(), R[n,
                                                                0:t].flatten()
            sgn_delta_vector = np.sign(delta_vector[state_action_to_index(
                s_vector, a_vector)])
            h_values_vector = h.get_hvalue(sp_vector, r_vector)
            exphs_values_vector = exphs0_vector[state_action_to_index(
                s_vector, a_vector)]
            weight_values_vector = weight0_vector[state_action_to_index(
                s_vector, a_vector)]
            res0[b, n] = sgn_delta_vector * (
                h_values_vector - exphs_values_vector) * weight_values_vector

        normalization_scale = np.std(np.concatenate([res0[b], res1[b]],
                                                    axis=1),
                                     ddof=1)
        raw_S = integral + np.mean(res1[b], axis=1) - np.mean(res0[b], axis=1)
        # scaled_S = raw_S / np.std(aug1-aug0, ddof=1) * np.sqrt(t * (T - t) / T)
        scaled_S[b] = np.mean(raw_S / normalization_scale *
                              np.sqrt(t * (T - t) / T))
        scaled_res0[b] = res0[b] / normalization_scale
        scaled_res1[b] = res1[b] / normalization_scale
    return scaled_S, scaled_res0, scaled_res1


def calculate_test_statistics(S, A, R, ts, h_funs, num_states, num_actions,
                              num_rewards, weight_clip_value, seed):
    np.random.seed(seed)
    N, T = A.shape
    B = len(h_funs)
    k_folds = KFold(n_splits=2, shuffle=True, random_state=seed).split(S)
    idxs_train, idxs_test = next(k_folds)
    S1, A1, R1 = S[idxs_train], A[idxs_train], R[idxs_train]
    S2, A2, R2 = S[idxs_test], A[idxs_test], R[idxs_test]

    scaled_S_all = np.zeros([len(ts), B])
    scaled_res0s = [None for i in range(len(ts))]
    scaled_res1s = [None for i in range(len(ts))]
    for i in range(len(ts)):
        w_models, pt_models = train_nuisance_models(S1, A1, R1, ts[i],
                                                    num_states, num_actions,
                                                    num_rewards)
        g_fun = Mixture2Distribution(mixture1=w_models[0],
                                     mixture2=w_models[1],
                                     weight1=ts[i] / A1.shape[1],
                                     weight2=1 - ts[i] / A1.shape[1])
        scaled_S_all[i], scaled_res0s[i], scaled_res1s[
            i] = calculate_S_one_t_all_h(S2, A2, R2, ts[i], pt_models,
                                         w_models, g_fun, h_funs, num_states,
                                         num_actions, num_rewards,
                                         weight_clip_value)
    test_statistic = np.max(scaled_S_all)
    test_statistic_bootstrap = calculate_test_statistic_boostrap(
        scaled_res0s, scaled_res1s)
    pvalue = np.mean(test_statistic <= test_statistic_bootstrap)
    return pvalue, test_statistic, test_statistic_bootstrap


def calculate_test_statistic_boostrap(res0s, res1s, J=3000):
    len_ts = len(res0s)
    B = res0s[0].shape[0]
    N = res0s[0].shape[1]
    T = res0s[0].shape[-1] + res1s[0].shape[-1]

    test_statistic_boots = np.zeros([J])
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
        test_statistic_boots[j] = np.max(tmp)
    return test_statistic_boots


def combine_multiple_p_values(pvalues, gamma=0.15):
    if len(pvalues) == 1:
        return pvalues[0]
    else:
        return np.min([1, np.quantile(np.array(pvalues) / gamma, gamma)])


def calculate_test_statistics_star(args):
    return calculate_test_statistics(*args)


def test_stationarity_mdp(S,
                          A,
                          R,
                          B,
                          ts,
                          num_states,
                          num_actions,
                          num_rewards,
                          weight_clip_value,
                          random_repeats,
                          cores,
                          seed,
                          pvalue_combine_gamma=0.15):
    h_funs = [
        random_h(num_states=num_states, num_rewards=num_rewards)
        for i in range(B)
    ]

    if cores == 1:
        pvalues = []
        for rep in range(random_repeats):
            pvalue, _, _ = calculate_test_statistics(
                S=S,
                A=A,
                R=R,
                ts=ts,
                h_funs=h_funs,
                num_states=num_states,
                num_actions=num_actions,
                num_rewards=num_rewards,
                weight_clip_value=weight_clip_value,
                seed=seed + rep)
            pvalues.append(pvalue)
    else:
        all_jobs = [(S, A, R, ts, h_funs, num_states, num_actions, num_rewards,
                     weight_clip_value, seed + rep)
                    for rep in range(random_repeats)]
        pvalues, _, _ = list(
            zip(*p_imap(calculate_test_statistics_star, all_jobs)))
    combined_pvalue = combine_multiple_p_values(
        pvalues=np.array(pvalues).flatten(), gamma=pvalue_combine_gamma)
    return combined_pvalue, pvalues
