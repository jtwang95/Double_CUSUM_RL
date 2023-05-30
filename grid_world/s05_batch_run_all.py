import os, shutil, time, sys

sys.path.append("../")
import logging
from Utils.logger import *
import subprocess
import pandas as pd
import itertools, glob
from datetime import datetime


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def generate_kappa_dict(kappa0, delta, t_max, T):
    kappa, t = kappa0, t_max
    d = {kappa: [t]}
    kappa = kappa + delta
    t = t - delta
    while kappa <= T:
        # print(d)
        d[kappa] = d[kappa - delta] + [t]
        kappa = kappa + delta
        t = t - delta
    return d


######################################
############## setting ###############
######################################
JOB_ID = datetime.now().strftime("%y%m%d%H%M%S")
CHGPT = 25
REP = 100
N = 100
T = 50
P = 0.05
B = 100
GRID_SIZE = 4
NUM_STATES = GRID_SIZE**2
NUM_ACTIONS = 4
NUM_REWARDS = 4
NUM_RANDOM_REPEATS = 10
WEIGHT_CLIP_VALUE = 100
CORES = 8
SEED = 42
GAMMA = 0.15

## value evaluation
N_EVAL = 100
T_EVAL = 50

JOB_NAME = "simulation_gridworld_gridsize_{}_rep_{}_chgpt_{}_N_{}_T_{}_{}".format(
    GRID_SIZE, NUM_RANDOM_REPEATS, CHGPT, N, T, JOB_ID)
OUT_FOLDER = "./outs/" + JOB_NAME + "/"

KAPPAS_DICT = generate_kappa_dict(10, 5, 45, 40)

res = expand_grid({
    "N": [N],
    "T": [T],
    "kappa_setting": [
        str(key) + "_" + ",".join([str(i) for i in value])
        for key, value in KAPPAS_DICT.items()
    ]
})

if not os.path.exists("./outs/"):
    os.mkdir("./outs/")

######################################
############# set env ################
######################################
mylogger.info("create " + OUT_FOLDER)
if os.path.exists(OUT_FOLDER):
    raise FileExistsError("Folder" + OUT_FOLDER + " exists.")
else:
    os.mkdir(OUT_FOLDER)
    os.mkdir(OUT_FOLDER + "scripts/")
    os.mkdir(OUT_FOLDER + "outputs/")
    os.mkdir(OUT_FOLDER + "bash_files/")
    files_to_save = list(glob.iglob("./*.py")) + list(
        glob.iglob("./*.sh")) + list(glob.iglob("../core/*.py"))
    _ = [
        shutil.copyfile(f, OUT_FOLDER + "scripts/" + f.split("/")[-1])
        for f in files_to_save
    ]
    handler = logging.FileHandler(OUT_FOLDER + JOB_NAME + ".log", mode="a")
    handler.setFormatter(logging.Formatter(FORMAT))
    mylogger.addHandler(handler)

######################################
##### step 1: simulate dataset #######
######################################
mylogger.info("Create {} simulated datasets".format(REP))

tpl1 = '''
# SETTING
REP={}
N={}
T={}
CHGPT={}
GRIDSIZE={}
OUTFOLDER={}
SEED={}

python -u s01_simulate_dataset.py -T $T -N $N --nrep $REP --grid-size $GRIDSIZE --change-point $CHGPT --out-folder $OUTFOLDER --seed $SEED
'''
sbatch_filename = os.path.join(OUT_FOLDER, "bash_files/",
                               "01_simulate_dataset.sh")
with open(sbatch_filename, "w") as f:
    f.write(tpl1.format(REP, N, T, CHGPT, GRID_SIZE, OUT_FOLDER, SEED))
p = subprocess.Popen("bash " + sbatch_filename,
                     shell=True,
                     stdout=subprocess.PIPE)
p.wait()

######################################
##### step 2: chgpt detection ########
######################################
tpl2 = '''
#SETTING
B={}
P={}
num_states={}
num_actions={}
num_rewards={}
num_random_repeats={}
out_folder={}
kappa={}
ts={}
cores={}
weight_clip_value={}
gamma={}
seed={}


# RUN
python -u s02_test_kappa.py --num-states $num_states --num-actions $num_actions \
     --num-rewards $num_rewards -B $B --num-random-repeats $num_random_repeats \
     -p $P --out-folder $out_folder --kappa $kappa --ts $ts --seed $seed \
     --weight-clip-value $weight_clip_value --cores $cores
'''
for index, row in res.iterrows():
    t0 = time.time()
    mylogger.info(
        "{}/{} --- Submit jobs for configuration {} using {} cores".format(
            index + 1, res.shape[0], str(row.to_dict()), CORES))
    kappa, ts = row["kappa_setting"].split("_")
    kappa_setting = row["kappa_setting"]
    sbatch_filename = os.path.join(
        OUT_FOLDER, "bash_files/",
        "02_batch_run_kappa_{}_{}.sh".format(kappa, ts))
    with open(sbatch_filename, "w") as f:
        f.write(
            tpl2.format(B, P, NUM_STATES, NUM_ACTIONS, NUM_REWARDS,
                        NUM_RANDOM_REPEATS, OUT_FOLDER, kappa, ts, CORES,
                        WEIGHT_CLIP_VALUE, GAMMA, SEED))
    p = subprocess.Popen("bash " + sbatch_filename,
                         shell=True,
                         stdout=subprocess.PIPE)
    p.wait()
    output_filename = os.path.join(OUT_FOLDER, "outputs",
                                   "kappa_{}_ts_{}.out".format(kappa, ts))
    out = pd.read_csv(output_filename, sep="\t", header=0)
    rej_prob = out.reject.mean()
    res.loc[res["kappa_setting"] == kappa_setting, "rej_prob"] = rej_prob
    t1 = time.time()
    res.loc[res["kappa_setting"] == kappa_setting,
            "time"] = "{:.2f}".format(t1 - t0)
    mylogger.info("Done {}; Rej prob:{}; Time used:{}.".format(
        kappa_setting, rej_prob, t1 - t0))

res.to_csv(OUT_FOLDER + "res.csv", index=False, sep="\t")

############################
####### kappa plot #########
############################
subprocess.check_output([
    "python", OUT_FOLDER + "scripts/s04_plot_kappa.py", "-f",
    OUT_FOLDER + "res.csv", "-o", OUT_FOLDER
],
                        stderr=subprocess.STDOUT)

#####################################
#### step 3: determine chgpt ########
#####################################
mylogger.info("STEP 3: determine change point")

tpl3 = '''
# SETTING
FOLDER={}

python -u s03_determine_chgpts.py  --folder $FOLDER
'''
sbatch_filename = os.path.join(OUT_FOLDER, "bash_files/",
                               "03_determine_chgpts.sh")
with open(sbatch_filename, "w") as f:
    f.write(tpl3.format(OUT_FOLDER))
p = subprocess.Popen("bash " + sbatch_filename,
                     shell=True,
                     stdout=subprocess.PIPE)
p.wait()

######################################
##### step 4: value evaluation #######
######################################
mylogger.info("STEP 4: value evaluation")
if not os.path.exists(os.path.join(OUT_FOLDER, "value_evaluation")):
    os.mkdir(os.path.join(OUT_FOLDER, "value_evaluation"))

tpl4 = ''' 
#SETTING
folder={}
grid_size={}
num_states={}
num_actions={}
change_point={}
cores={}
n_eval={}
t_eval={}

# RUN
python s04_evaluate_value.py --folder $folder --grid-size $grid_size --num-states $num_states --num-actions $num_actions\
     --change-point $change_point --cores $cores --n-eval $n_eval --t-eval $t_eval
'''
mylogger.info("Submit value evaluation jobs using {} cores".format(
    min(REP, CORES)))
sbatch_filename = os.path.join(OUT_FOLDER, "bash_files/",
                               "04_value_evaluation.sh")
with open(sbatch_filename, "w") as f:
    f.write(
        tpl4.format(OUT_FOLDER, GRID_SIZE, NUM_STATES, NUM_ACTIONS, CHGPT,
                    CORES, N_EVAL, T_EVAL))
p = subprocess.Popen("bash " + sbatch_filename,
                     shell=True,
                     stdout=subprocess.PIPE)
p.wait()
