import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os.path

parser = argparse.ArgumentParser(
    description='plot kappa plot for simulation dataset')
parser.add_argument('-f',
                    '--filename',
                    type=str,
                    help='csv file',
                    required=True)
parser.add_argument('-o',
                    '--outfolder',
                    type=str,
                    help="output folder of figure file",
                    default="./")
args = parser.parse_args()

FILENAME = args.filename
OUTFOLDER = args.outfolder

if not os.path.isfile(FILENAME):
    raise FileExistsError("Cannot open {}".format(FILENAME))

d = pd.read_csv(FILENAME, header=0, sep="\t")
d["kappa"] = [int(i.split("_")[0]) for i in d["kappa_setting"]]

## make plot
fig, ax = plt.subplots()
ax.plot(d["kappa"], d["rej_prob"], '-o')
ax.axhline(y=0.05, linestyle="--", color="black")
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel("Reject prob")
ax.set_xticks(range(min(d["kappa"]), max(d["kappa"]) + 1, 5))
ax.set_yticks([0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim([-0.05, 1.05])
fig.savefig(os.path.join(OUTFOLDER, "rej_prob_vs_kappa.png"))
print("Saved kappa file to {}".format(
    os.path.join(OUTFOLDER, "rej_prob_vs_kappa.png")))
