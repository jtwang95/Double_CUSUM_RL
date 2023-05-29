import numpy as np
import os, sys

sys.path.append("../")
from gen_data import multiGaussionSys
from Utils.logger import *
import argparse
import os

# parse arguments
parser = argparse.ArgumentParser(description='generate simulation data')
parser.add_argument('-T', type=int, help='Length of horizon', default=40)
parser.add_argument('-N', type=int, help='Number of trajctories', default=10)
parser.add_argument('--type',
                    type=str,
                    help='Type of simulation',
                    default="hmo")
parser.add_argument('--nrep',
                    type=int,
                    help="number of independent datasets",
                    default=1)
parser.add_argument('--chgpt',
                    type=lambda x: None if x == 'None' else int(x),
                    help="change point",
                    default=None)
parser.add_argument(
    '--sdim',
    type=int,
    help="the demension of states",
    default=1,
)
parser.add_argument("--outfolder",
                    type=str,
                    help="the folder to store output",
                    default="./outs/test")
parser.add_argument('--loginfo', action=argparse.BooleanOptionalAction)
parser.add_argument('--debug',
                    action='store_true',
                    help='print debug messages to stderr')
parser.add_argument('--seed', type=int, help="random seed", default=-1)
args = parser.parse_args()

T = args.T
N = args.N
SIMULATION_TYPE = args.type
NREP = args.nrep
CHGPT = args.chgpt
S_DIM = args.sdim
OUTPUT_FOLDER = args.outfolder
LOGINFO = False if args.loginfo == False else True
DEBUG = args.debug
SEED = args.seed if args.seed > 0 else None
np.random.seed(SEED)

# set logger and folder
mylogger = logging.getLogger("simulate_data")
log_level = logging.INFO if LOGINFO else logging.WARN
log_level = logging.DEBUG if DEBUG else log_level
coloredlogs.install(level=log_level,
                    fmt=FORMAT,
                    datefmt=DATEF,
                    level_styles=LEVEL_STYLES,
                    logger=mylogger)

data_folder = os.path.join(OUTPUT_FOLDER, "data")
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# generate dataset
mySys = multiGaussionSys(s_dim=S_DIM)
for num in range(NREP):
    S, A, R = mySys.simulate(mean0=0,
                             cov0=1,
                             N=N,
                             T=T,
                             type=SIMULATION_TYPE,
                             mean=0,
                             cov=1,
                             change_pt=CHGPT)
    np.savez(os.path.join(data_folder, "sim_data_{}.npz".format(num)),
             S=S,
             A=A,
             R=R)
with open(os.path.join(OUTPUT_FOLDER, "real_chgpt.out"), "w") as f:
    f.write(SIMULATION_TYPE + "\t" + str(CHGPT))

mylogger.info(
    "Done! Have generated {} datasets, type: {}, chgpt: {}, N: {}, T: {}, r00:{:0.3f},seed:{}"
    .format(NREP, SIMULATION_TYPE, CHGPT, N, T, R[0, 0], args.seed))
