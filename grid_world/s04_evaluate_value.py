import sys, argparse, os, tqdm, copy, glob

sys.path.append("../")
import numpy as np
from Utils.logger import *
from a01_generate_trajectories import generate_grid_world_trajectories
from Utils.help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocess import Pool

# parse arguments
parser = argparse.ArgumentParser(description='Value evaluation')
parser.add_argument("--folder", type=str, help="the folder to store output")
parser.add_argument("--grid-size", type=int)
parser.add_argument("--num-states", type=int)
parser.add_argument("--num-actions", type=int)
parser.add_argument("--change-point", type=int)
parser.add_argument("--n-eval", type=int)
parser.add_argument("--t-eval", type=int)
parser.add_argument('--cores',
                    type=int,
                    help="number of cores to use",
                    default=1)
args = parser.parse_args()

WORK_FOLDER = args.folder
GRID_SIZE = args.grid_size
NUM_STATES = args.num_states
NUM_ACTIONS = args.num_actions
CHANGE_POINT = args.change_point
N_EVAL = args.n_eval
T_EVAL = args.t_eval
CORES = args.cores

if not os.path.exists(os.path.join(WORK_FOLDER, "value_evaluation")):
    os.mkdir(os.path.join(WORK_FOLDER, "value_evaluation"))
output_filename = "values.out"

mylogger.info("Output filename {}".format(output_filename))


class QTable:

    def __init__(self, num_states, num_actions) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros([self.num_states, self.num_actions])

    def act(self, s):
        return np.argmax(self.q_table[s, :])

    def update(self, s, a, sp, r, lr, gamma):
        new_value = (1 - lr) * self.q_table[s, a] + lr * (
            r + gamma * np.max(self.q_table[sp]))
        self.q_table[s, a] = new_value


def table_offline_q_learning(data, model_parameters, train_parameters):
    '''
    # data={S:,A:,R:}
    # model_parameters={num_states,num_actions}
    # train_parameters={lr:,epochs}
    '''
    S, A, R = data["S"], data["A"], data["R"]
    N, T = A.shape
    q_table = QTable(num_states=model_parameters["num_states"],
                     num_actions=model_parameters["num_actions"])
    for ep in range(train_parameters["epochs"]):
        old_q_table = copy.deepcopy(q_table)
        for n in range(N):
            for t in range(T):
                s, a, sp, r = S[n, t], A[n, t], S[n, t + 1], R[n, t]
                q_table.update(s=s,
                               a=a,
                               sp=sp,
                               r=r,
                               lr=train_parameters["lr"],
                               gamma=0.9)
        if ep % 10 == 0:
            mylogger.debug("epoch:{}, old new difference:{}".format(
                ep, np.mean((old_q_table.q_table - q_table.q_table)**2)))
    return q_table


def value_evaluation(S, A, R, N_eval, T_eval, seed):
    # information of data
    N, T = A.shape
    data = {"S": S, "A": A, "R": R}
    model_parameters = {"num_states": NUM_STATES, "num_actions": NUM_ACTIONS}
    train_parameters = {"lr": 0.1, "epochs": 100}

    ql_policy = table_offline_q_learning(data=data,
                                         model_parameters=model_parameters,
                                         train_parameters=train_parameters)

    class MyPolicy():

        def __init__(self, policy, policy_switch_t) -> None:
            self.policy = policy
            self.policy_switch_t = policy_switch_t

        def act(self, s, t):
            if t < self.policy_switch_t:
                return np.random.choice(NUM_ACTIONS)
            else:
                return self.policy.act(s)

    S_eval, A_eval, R_eval = generate_grid_world_trajectories(
        N_eval,
        size_grid=GRID_SIZE,
        change_point=CHANGE_POINT,
        censor_time=CHANGE_POINT + T_eval,
        seed=seed,
        policy=MyPolicy(ql_policy, CHANGE_POINT))

    expected_reward = calculate_expected_discounted_reward_MC(
        R_eval[:, CHANGE_POINT:], gamma=0.9)
    return expected_reward


def evaluate_one_dataset(id,
                         seed,
                         S,
                         A,
                         R,
                         detected_last_chgpt,
                         N_eval,
                         T_eval,
                         overall=True,
                         detected=True,
                         random=False):
    N, T = A.shape
    mylogger.debug("Training dataset id:{},N:{};T:{}".format(id, N, T))
    np.random.seed(seed)
    # overall
    if overall:
        expected_reward_overall = value_evaluation(S=S,
                                                   A=A,
                                                   R=R,
                                                   N_eval=N_eval,
                                                   T_eval=T_eval,
                                                   seed=seed)
    else:
        expected_reward_overall = float("NaN")

    # detected change point
    if detected:
        S1, A1, R1 = S[:,
                       detected_last_chgpt:], A[:,
                                                detected_last_chgpt:], R[:,
                                                                         detected_last_chgpt:]
        expected_reward_detected = value_evaluation(S=S1,
                                                    A=A1,
                                                    R=R1,
                                                    N_eval=N_eval,
                                                    T_eval=T_eval,
                                                    seed=seed)
    else:
        expected_reward_detected = float("NaN")

    # random chage point
    if random:
        EPSILON = 0.1
        REP = 10
        tmp = np.zeros([REP])
        for _ in range(REP):
            random_chgpt = np.random.choice(
                range(int(T * EPSILON), int(T - T * EPSILON)))
            S1, A1, R1 = S[:,
                           random_chgpt:], A[:,
                                             random_chgpt:], R[:,
                                                               random_chgpt:]
            tmp[_] = value_evaluation(S=S1,
                                      A=A1,
                                      R=R1,
                                      N_eval=N_eval,
                                      T_eval=T_eval,
                                      seed=seed)
        expected_reward_random = np.mean(tmp)
    else:
        expected_reward_random = float("NaN")

    return {
        "id": id,
        "overall": expected_reward_overall * 0.1,
        "detected": expected_reward_detected * 0.1,
        "random": expected_reward_random * 0.1
    }


def evaluate_one_dataset_star(args):
    return (evaluate_one_dataset(*args))


def calculate_expected_discounted_reward_MC(reward, gamma):
    N, T = reward.shape
    s = np.zeros([N])
    for t in range(T - 1, -1, -1):
        s = gamma * s + reward[:, t]
    return np.mean(s)


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(WORK_FOLDER, "detected_chgpts.out"), sep="\t")
    data_folder = os.path.join(WORK_FOLDER, "data")
    data_files = glob.glob(os.path.join(data_folder, "*.npz"))
    num_files = len(data_files)
    mylogger.info("Create all jobs with {} dataset".format(num_files))
    all_jobs = []  # cross_fold * num_files
    for _, filename in enumerate(data_files):
        id = int(
            os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
        data = np.load(filename)
        S, A, R = data["S"], data["A"], data["R"]
        seed = len(all_jobs)
        all_jobs.append(
            (id, seed, S, A, R, d.loc[d.id == id,
                                      "detected_chgpt"].to_numpy()[0], N_EVAL,
             T_EVAL))

    mylogger.info("Create {} jobs".format(len(all_jobs)))
    mylogger.info("Running MP version with {} cores and {} jobs".format(
        CORES, len(all_jobs)))
    with Pool(CORES) as pool:
        output = list(
            tqdm.tqdm(pool.imap(evaluate_one_dataset_star, all_jobs),
                      total=len(all_jobs)))

    res = pd.DataFrame(output)
    print(res)
    res.to_csv(os.path.join(WORK_FOLDER, "value_evaluation", "values.out"),
               index=False,
               sep="\t")

    ## plot
    d = pd.melt(frame=res, value_vars=["overall", "detected", "random"])
    sns.boxplot(x="variable", y="value", data=d)
    plt.savefig(
        os.path.join(WORK_FOLDER, "value_evaluation", "values_boxplot.png"))
