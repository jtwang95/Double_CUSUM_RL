import numpy as np
import os, sys, argparse, time, tqdm, json, glob

sys.path.append("../")
from Utils.logger import *
from core.calculate_test_statistic import *
from Utils.help_functions import *
import pandas as pd
from multiprocessing import Pool, cpu_count

# parse arguments
parser = argparse.ArgumentParser(
    description='test for non-stationarity given data, kappa and ts')
parser.add_argument('-B',
                    type=int,
                    help='Number of arbitary g or h functions',
                    default=2)
parser.add_argument('-p',
                    '--p-threshold',
                    type=float,
                    help='Threshold of p value',
                    default=0.05)
parser.add_argument('-M',
                    type=int,
                    help='Number of Monte Carlo samples',
                    default=100)
parser.add_argument('--htype',
                    type=str,
                    help="Type of random h function",
                    default="hybrid")
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
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help="learning rate for pt models")
parser.add_argument('--pt-hidden-dims',
                    type=str,
                    default=None,
                    help="hidden layers of pt models")
parser.add_argument('--pt-epochs',
                    type=int,
                    default=None,
                    help="number of epochs for pt models")
parser.add_argument('--pt-cv', type=int, default=0, help="pt cv or not")
parser.add_argument('--pt-cv-folds',
                    type=int,
                    default=5,
                    help="number of folds for cv")
parser.add_argument('--pt-cv-cores',
                    type=int,
                    default=1,
                    help="number of cores for cv")
parser.add_argument('--pt-nn-candidates',
                    type=str,
                    default=["32,32"],
                    help="pt nn candidates, seperated by semicolon")
parser.add_argument('--pt-epochs-candidates',
                    type=str,
                    default="50,100",
                    help="pt epoch candidates")
parser.add_argument('--num-wcomponents',
                    type=int,
                    default=2,
                    help="number of mixed gaussian to estimate omega")
parser.add_argument('--gamma',
                    type=float,
                    default=0.15,
                    help="gamma for pvalue combination")
args = parser.parse_args()

B = args.B
M = args.M
PVALUE_THRESHOLD = args.p_threshold
H_TYPE = args.htype
NUM_RANDOM_REPEATS = args.num_random_repeats
OUTPUT_FOLDER = args.out_folder
KAPPA = args.kappa
TS = [int(i) for i in args.ts.split(",")]
SEED = args.seed if args.seed > 0 else None
WEIGHT_CLIP_VALUE = args.weight_clip_value
CORES = min(args.cores, cpu_count())
LEARNING_RATE = args.lr
PT_CV = False if args.pt_cv == 0 else True
NUM_WCOMPONENTS = args.num_wcomponents
GAMMA = args.gamma

PT_CV_FOLDS = args.pt_cv_folds if PT_CV else None
PT_CV_CORES = args.pt_cv_cores if PT_CV else None
PT_NN_CANDIDATES = args.pt_nn_candidates.split(";") if PT_CV else None
PT_EPOCHS_CANDIDATES = [int(i) for i in args.pt_epochs_candidates.split(",")
                        ] if PT_CV else None
PT_HIDDEN_DIMS = [int(i) for i in args.pt_hidden_dims.split(",")
                  ] if not PT_CV else None
PT_EPOCHS = args.pt_epochs if not PT_CV else None


def test_one_dataset(id, S, A, R, seed, ts):
    pvalue, pvalues, test_statistics = run_change_point_detection(
        S=S,
        A=A,
        R=R,
        M=M,
        B=B,
        ts=ts,
        htype=H_TYPE,
        learning_rate=LEARNING_RATE,
        pt_cv=PT_CV,
        pt_cv_folds=PT_CV_FOLDS,
        w_ncomponents=NUM_WCOMPONENTS,
        weight_clip_value=WEIGHT_CLIP_VALUE,
        random_repeats=NUM_RANDOM_REPEATS,
        cores=1,
        seed=seed,
        pt_hidden_dims=PT_HIDDEN_DIMS,
        pt_epochs=PT_EPOCHS,
        pt_cv_cores=PT_CV_CORES,
        pt_nn_candidates=PT_NN_CANDIDATES,
        pt_epochs_candidates=PT_EPOCHS_CANDIDATES,
        pvalue_combine_gamma=GAMMA)
    return {
        "id": id,
        "pvalue": pvalue,
        "pvalues": pvalues,
        "test_statistic_mean": np.mean(test_statistics)
    }


def test_one_dataset_star(args):
    return test_one_dataset(*args)


if __name__ == "__main__":
    # save configuration
    with open(os.path.join(OUTPUT_FOLDER, 'settings.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # load data
    if not os.path.exists(os.path.join(OUTPUT_FOLDER, "outputs")):
        os.mkdir(os.path.join(OUTPUT_FOLDER, "outputs"))
    data_folder = os.path.join(OUTPUT_FOLDER, "data")
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
        S_DIM = S.shape[-1]

        T0, T1 = T - KAPPA, T
        ts = choose_time_points(T0=T0, T1=T1 - 1, forced_time_points=TS)
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
        OUTPUT_FOLDER, "outputs",
        "kappa_{}_ts_{}.out".format(KAPPA, ",".join([str(i) for i in TS]))),
               index=False,
               sep="\t")
