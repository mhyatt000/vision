# Usage

To use run:

```bash
./run.sh <experiment_name>
```

or 

```bash
torchrun <your_torchrun_options> general/master.py <experiment_name>
```

## Experiment Config

See `MLEX/config` on defining an experiment config file. Place the file into the `configs` folder before running the experiment.

## Experiment Results

Experiment results will be placed into `experiments/<name>`. Results include related figures, logs, and model weights.
