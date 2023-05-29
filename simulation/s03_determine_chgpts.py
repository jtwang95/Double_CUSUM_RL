import pandas as pd
import argparse, os
import numpy as np
import glob

# parse arguments
parser = argparse.ArgumentParser(
    description='Determine chgpts given kappa files')
parser.add_argument("--folder", type=str, help="the folder to store output")
args = parser.parse_args()

WORK_FOLDER = args.folder

data = np.load(os.path.join(WORK_FOLDER, "data", "sim_data_0.npz"))
S, A, R = data["S"], data["A"], data["R"]
N, T = A.shape
REP = len(glob.glob(os.path.join(WORK_FOLDER, "data", "*.npz")))

kappa_files = [
    os.path.basename(i)
    for i in glob.glob(os.path.join(WORK_FOLDER, "outputs", "kappa*"))
]
kappa_settings = [[
    int(os.path.splitext(file)[0].split('_')[1]),
    os.path.splitext(file)[0].split('_')[3]
] for file in kappa_files]
kappa_settings = sorted(kappa_settings, key=lambda x: x[0])

kappa_chgpt = [T] * REP
has_rejected = [0] * REP

for kappa, ts in kappa_settings:
    d = pd.read_csv(os.path.join(WORK_FOLDER, "outputs",
                                 "kappa_{}_ts_{}.out".format(kappa, ts)),
                    sep="\t")
    d = d.sort_values(by="id", ascending=True)
    rejects = list(d.reject)
    for i in d.id:
        if rejects[i] == 0 and has_rejected[i] == 0:
            kappa_chgpt[i] = int(kappa)
        if rejects[i] == 1:
            if kappa == kappa_settings[0][0]:
                kappa_chgpt[i] = int(kappa)
            has_rejected[i] = 1

detected_chgpts = [T - i for i in kappa_chgpt]
res = pd.DataFrame({
    "id": range(len(detected_chgpts)),
    "detected_chgpt": detected_chgpts
}).sort_values(by="id", ascending=True)
res.to_csv(os.path.join(WORK_FOLDER, "detected_chgpts.out"),
           index=False,
           sep="\t")
