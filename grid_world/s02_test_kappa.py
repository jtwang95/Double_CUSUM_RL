import numpy as np
import os, argparse, tqdm, json, glob, sys

sys.path.append("../")
from Utils.logger import *
from core.calculate_test_statisitc_discrete import run_change_point_detection
import pandas as pd
from multiprocessing import Pool, cpu_count

# parse arguments
parser = argparse.ArgumentParser(
    description='test for non-stationarity given data, kappa and ts')
parser.add_argument('--num-states', help="number of states", type=int)
parser.add_argument('--num-actions', help="number of actions", type=int)
parser.add_argument('--num-rewards', help="number of rewards", type=int)
parser.add_argument('-B',
                    type=int,
                    help='Number of arbitary g or h functions',
                    default=2)
parser.add_argument('-p',
                    '--p-threshold',
                    type=float,
                    help='Threshold of p value',
                    default=0.05)
parser.add_argument('--num-random-repeats',
                    type=int,
                    help="number of random repeats",
                    default=1)
parser.add_argument("--out-folder",
                    type=str,
                    help="the working folder to store output",
                    default="./outs/test")
parser.add_argument("--kappa",
                    type=int,
                    help="kappa value, test non-stationarity on [T-kappa,T]")
parser.add_argument("--ts",
                    type=str,
                    help="testing time points, sepearted by ','")
parser.add_argument('--seed', type=int, help="random seed")
parser.add_argument('--weight-clip-value',
                    type=float,
                    default=None,
                    help="Threshold for weight clipping for augmented term")
parser.add_argument('--cores',
                    type=int,
                    default=1,
                    help="Number of cores to used.")
parser.add_argument('--gamma',
                    type=float,
                    default=0.15,
                    help="gamma for pvalue combination")
args = parser.parse_args()

NUM_STATES = args.num_states
NUM_ACTIONS = args.num_actions
NUM_REWARDS = args.num_rewards
B = args.B
PVALUE_THRESHOLD = args.p_threshold
NUM_RANDOM_REPEATS = args.num_random_repeats
OUT_FOLDER = args.out_folder
KAPPA = args.kappa
TS = [int(i) for i in args.ts.split(",")]
SEED = args.seed if args.seed > 0 else None
WEIGHT_CLIP_VALUE = args.weight_clip_value
CORES = min(args.cores, cpu_count())
GAMMA = args.gamma


def test_one_dataset(id, S, A, R, seed, ts):
    pvalue, pvalues = run_change_point_detection(
        S=S,
        A=A,
        R=R,
        B=B,
        ts=ts,
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        weight_clip_value=WEIGHT_CLIP_VALUE,
        random_repeats=NUM_RANDOM_REPEATS,
        cores=1,
        seed=seed,
        pvalue_combine_gamma=GAMMA)
    return {"id": id, "pvalue": pvalue, "pvalues": pvalues}


def test_one_dataset_star(args):
    return test_one_dataset(*args)


if __name__ == "__main__":
    # save configuration
    with open(os.path.join(OUT_FOLDER, 'settings.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # load data
    if not os.path.exists(os.path.join(OUT_FOLDER, "outputs")):
        os.mkdir(os.path.join(OUT_FOLDER, "outputs"))
    data_folder = os.path.join(OUT_FOLDER, "data")
    data_files = glob.glob(os.path.join(data_folder, "*.npz"))
    num_files = len(data_files)
    mylogger.info("{} files with kappa {} and ts {}".format(
        num_files,
        KAPPA,
        str(TS),
    ))
    all_jobs = []  # cross_fold * num_files
    for _, filename in enumerate(data_files):
        idx = int(
            os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
        data = np.load(filename)
        S, A, R = data["S"], data["A"], data["R"]
        N, T = A.shape

        T0, T1 = T - KAPPA, T
        ts = TS
        # set time 0 to T-KAPPA
        ts = [i - T0 for i in ts]
        S = S[:, T0:(T1 + 1), :]
        R = R[:, T0:T1]
        A = A[:, T0:T1]
        all_jobs.append((idx, S, A, R, SEED, ts))

    mylogger.info("Running all jobs with {} cores and {} jobs".format(
        CORES, len(all_jobs)))
    with Pool(CORES) as pool:
        output = list(
            tqdm.tqdm(pool.imap(test_one_dataset_star, all_jobs),
                      total=len(all_jobs)))

    res = pd.DataFrame(output)
    res["reject"] = (res["pvalue"] <= PVALUE_THRESHOLD).astype(int)
    ########### output ###################
    res.to_csv(os.path.join(
        OUT_FOLDER, "outputs",
        "kappa_{}_ts_{}.out".format(KAPPA, ",".join([str(i) for i in TS]))),
               index=False,
               sep="\t")
