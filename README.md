# RBC reinforcement experiment

This is an experiment to examine the use of reinforcement learning to learn an optimal policy in a real-business cycle model. It is part of an effort to apply reinforcement learning to the broader class of macroeconomic DSGE-type models.


# Usage
This is still quite rough. 

1. Setup the hyperparameter grid-space in `experiments_params.yaml`.
2. Run `Snakemake INIT` to build a hyperparameter grid
3. Run `Snakemake TRAIN` to train each model in the hyperparameter grid

I'm still tweaking this, but in a short-episode-length setting, a PPO agent can learn a policy that outperforms the optimum constant-value policy. 

# Requirements:
This model was developed in the conda environment descsribed in conda.yaml

Most crucial dependencies are: 

- tensorforce (latest)
- tensorflow 1.4
- snakemake (latest)
- gym (latest)


