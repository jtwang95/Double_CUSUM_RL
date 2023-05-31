from core.calculate_test_statistic import test_stationarity_mdp
from core.calculate_test_statisitc_discrete import test_stationarity_mdp as test_stationarity_mdp_discrete

from simulation.gen_data import multiGaussionSys
from grid_world.a01_generate_trajectories import generate_grid_world_trajectories
from Utils.logger import *
import numpy as np


# continuous case
def simple_continuous_case():
    # generate simulation data
    mySys = multiGaussionSys(s_dim=1)
    S, A, R = mySys.simulate(mean0=0,
                             cov0=1,
                             N=300,
                             T=10,
                             type="pwc2ada_reward",
                             mean=0,
                             cov=1,
                             change_pt=5)

    # key is kappa defined in the paper; kappa means select time interval [T-kappa,T] where T is the length of horizon
    # value is the time point(s) selected to test within the interval [T-kappa, T].
    kappas_dict = {
        3: [8],
        4: [7, 8],
        5: [6, 7, 8],
        6: [5, 6, 7, 8],
        7: [4, 5, 6, 7, 8],
        8: [3, 4, 5, 6, 7, 8]
    }
    # store the (kappa,pvalue) pair
    pvalue_dict = {}

    for kappa, ts in kappas_dict.items():
        mylogger.info("Running kappa:{}, ts:{}".format(kappa, ts))
        T0, T1 = 10 - kappa, 10
        ts = [i - T0 for i in ts]
        S_ = S[:, T0:(T1 + 1), :]
        R_ = R[:, T0:T1]
        A_ = A[:, T0:T1]
        pvalue, _, _ = test_stationarity_mdp(S=S_,
                                             A=A_,
                                             R=R_,
                                             M=20,
                                             B=20,
                                             ts=ts,
                                             learning_rate=0.01,
                                             w_ncomponents=2,
                                             weight_clip_value=100,
                                             random_repeats=4,
                                             cores=4,
                                             seed=10,
                                             pt_hidden_dims=[8, 8],
                                             pt_epochs=10)
        pvalue_dict[kappa] = pvalue

    detected_chgpt = 10 - list(pvalue_dict.keys())[np.where(
        (np.array(list(pvalue_dict.values())) < 0.05))[0][0] - 1]

    print("pvalue_dict:{},detected_chgpt:{}".format(pvalue_dict,
                                                    detected_chgpt))


def simple_discrete_case():
    S, A, R = generate_grid_world_trajectories(num=300,
                                               size_grid=4,
                                               change_point=5,
                                               censor_time=10,
                                               seed=10)
    kappas_dict = {
        3: [8],
        4: [7, 8],
        5: [6, 7, 8],
        6: [5, 6, 7, 8],
        7: [4, 5, 6, 7, 8],
        8: [3, 4, 5, 6, 7, 8]
    }
    pvalue_dict = {}

    for kappa, ts in kappas_dict.items():
        mylogger.info("Running kappa:{}, ts:{}".format(kappa, ts))
        T0, T1 = 10 - kappa, 10
        ts = [i - T0 for i in ts]
        S_ = S[:, T0:(T1 + 1), :]
        R_ = R[:, T0:T1]
        A_ = A[:, T0:T1]
        pvalue, _ = test_stationarity_mdp_discrete(S=S_,
                                                   A=A_,
                                                   R=R_,
                                                   B=20,
                                                   ts=ts,
                                                   num_states=4 * 4,
                                                   num_actions=4,
                                                   num_rewards=4,
                                                   weight_clip_value=10,
                                                   random_repeats=4,
                                                   cores=4,
                                                   seed=10,
                                                   pvalue_combine_gamma=0.15)
        pvalue_dict[kappa] = pvalue

    detected_chgpt = 10 - list(pvalue_dict.keys())[np.where(
        (np.array(list(pvalue_dict.values())) < 0.05))[0][0] - 1]

    print("pvalue_dict:{},detected_chgpt:{}".format(pvalue_dict,
                                                    detected_chgpt))


if __name__ == "__main__":
    simple_continuous_case()  # ~20s on 4-core pc
    # simple_discrete_case() # ~5s on 4-core pc
