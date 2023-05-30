import numpy as np
import argparse, os, sys

sys.path.append("../")
from Utils.logger import *
from a01_generate_trajectories import generate_grid_world_trajectories

# parse arguments
parser = argparse.ArgumentParser(description='generate grid world dataset')
parser.add_argument('-T', type=int, help='Length of horizon', default=10)
parser.add_argument('--change-point',
                    type=lambda x: None if x == 'None' else int(x),
                    help="change point",
                    default=None)
parser.add_argument('-N', type=int, help='Number of trajctories', default=1)
parser.add_argument('--grid-size',
                    type=int,
                    help="size of the grid world",
                    default=3)
parser.add_argument('--nrep',
                    type=int,
                    help="number of independent datasets",
                    default=1)
parser.add_argument("--out-folder",
                    type=str,
                    help="the folder to store output",
                    default="./outs/test")
parser.add_argument('--seed', type=int, help="random seed", default=-1)
args = parser.parse_args()

T = args.T
N = args.N
NREP = args.nrep
CHANGE_POINT = args.change_point
GRID_SIZE = args.grid_size
OUT_FOLDER = args.out_folder
SEED = args.seed if args.seed > 0 else None
np.random.seed(SEED)

if not os.path.exists("./outs/"):
    os.mkdir("./outs/")

# set logger and folder
mylogger = logging.getLogger("simulate_data")
coloredlogs.install(level=logging.INFO,
                    fmt=FORMAT,
                    datefmt=DATEF,
                    level_styles=LEVEL_STYLES,
                    logger=mylogger)

data_folder = os.path.join(OUT_FOLDER, "data")
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

for i in range(NREP):
    S, A, R = generate_grid_world_trajectories(num=N,
                                               size_grid=GRID_SIZE,
                                               change_point=CHANGE_POINT,
                                               censor_time=T,
                                               seed=i)
    np.savez(os.path.join(data_folder, "sim_data_{}.npz".format(i)),
             S=S,
             A=A,
             R=R)
with open(os.path.join(OUT_FOLDER, "real_chgpt.out"), "w") as f:
    f.write(str(CHANGE_POINT))

mylogger.info(
    "Generated {} grid world datasets,  chgpt: {}, N: {}, T: {},seed:{}".
    format(NREP, CHANGE_POINT, N, T, args.seed))
