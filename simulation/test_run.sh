# for TEST only
## generate dataset
python -u s01_simulate_dataset.py -T 50 -N 100 --type pwc2ada_state --nrep 5 --chgpt 25 --sdim 10 --outfolder ./outs/test/ --seed 10

## test for one specific kappa
python -u s02_test_kappa.py -B 25  -p 0.05 -M 100 --htype hybrid  --out-folder ./outs/test/ --kappa 40 --ts 20 --seed 1234 --weight-clip-value 100 --cores 5 --pt-hidden-dims "128,128" --pt-epochs 100

## determine chgpts 
python -u s03_determine_chgpts.py --folder ./outs/test

# run a full experiment
## EST ~4 mins
python -u s04_batch_run_all.py