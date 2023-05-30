# simple example
## generate dataset
python -u s01_simulate_dataset.py -T 50 -N 100 --nrep 1 --grid-size 4 --change-point 25 --out-folder ./outs/test/ --seed 10

## test for one specific kappa
python -u s02_test_kappa.py --num-states 16 --num-actions 4 --num-rewards 4 -B 2 --num-random-repeats 1 -p 0.05 --out-folder ./outs/test/ --kappa 30 --ts 25 --seed 1234 --weight-clip-value 10 --cores 1

## determine chgpts 
python s03_determine_chgpts.py --folder ./outs/test

## value evaluation
python s04_evaluate_value.py --folder ./outs/test/ --grid-size 4 --num-states 16 --num-actions 4 --change-point 25 --cores 1 --n-eval 10 --t-eval 20