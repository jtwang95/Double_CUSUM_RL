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

data_files = list(glob.iglob(os.path.join(WORK_FOLDER, "data", "*.npz")))
data = np.load(data_files[0])
S, A, R = data["S"], data["A"], data["R"]
N, T = A.shape
num_files = len(data_files)

kappa_files = [
    os.path.basename(i)
    for i in glob.glob(os.path.join(WORK_FOLDER, "outputs", "kappa*"))
]
kappa_settings = [[
    int(os.path.splitext(file)[0].split('_')[1]),
    os.path.splitext(file)[0].split('_')[3]
] for file in kappa_files]
kappa_settings = sorted(kappa_settings, key=lambda x: x[0])

kappa_chgpt = {
    int(os.path.basename(key).replace(".", "_").split("_")[-2]): T
    for key in data_files
}
has_rejected = {
    int(os.path.basename(key).replace(".", "_").split("_")[-2]): 0
    for key in data_files
}
prev_kappa = kappa_settings[0][0]
for kappa, ts in kappa_settings:
    d = pd.read_csv(os.path.join(WORK_FOLDER, "outputs",
                                 "kappa_{}_ts_{}.out".format(kappa, ts)),
                    sep="\t")
    for id in d.id:
        if d.loc[d.id == id,
                 "reject"].values[0] == 1 and has_rejected[id] == 0:
            kappa_chgpt[id] = int(prev_kappa)
            has_rejected[id] = 1
    prev_kappa = kappa

detected_chgpts = [(key, T - value) for key, value in kappa_chgpt.items()]

res = pd.DataFrame(detected_chgpts,
                   columns=["id",
                            "detected_chgpt"]).sort_values(by="id",
                                                           ascending=True)
res.to_csv(os.path.join(WORK_FOLDER, "detected_chgpts.out"),
           index=False,
           sep="\t")
