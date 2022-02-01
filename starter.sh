#!/bin/bash
export WORKING_DIR=".";
export PYTHONPATH=$WORKING_DIR:$PYTHONPATH;
export LUIGI_CONFIG_PATH="./luigi.cfg"; # use luigi.cfg file.

### Prepare and plot data
# python3 -m luigi --module tasks.prep_data PrepareData --workingDir $WORKING_DIR --local-scheduler
# python3 -m luigi --module tasks.data_plots DataPlots --workingDir $WORKING_DIR --local-scheduler

### Train model
## No-lock enables starting two trainings in parallel.
## For parallel training simply call this script twice in the background (with different params in luigi.cfg or starter.sh)
python3 -m luigi --module tasks.train_model TrainModel --workingDir $WORKING_DIR --local-scheduler

### Successively train model
## For example when want to train stages successively by preloading weights from previous stage.
## At the end of TrainModel the path to the weights of this training is printed (as only thing) to stdout.
## For consecutive training, save the path in a var and pass it as `preloadPath` to the next training.
# WEIGHTS_A=$(python3 -m luigi --no-lock --module tasks.train_model TrainModel --stageMask 100 --workingDir $WORKING_DIR --local-scheduler)
# WEIGHTS_B=$(python3 -m luigi --no-lock --module tasks.train_model TrainModel --stageMask 010 --preloadPath $WEIGHTS_A --workingDir $WORKING_DIR --local-scheduler)
# WEIGHTS_C=$(python3 -m luigi --no-lock --module tasks.train_model TrainModel --stageMask 001 --preloadPath $WEIGHTS_B --workingDir $WORKING_DIR --local-scheduler)

### Evaluate model
# python3 -m luigi --module tasks.eval_plots EvalPlots --workingDir $WORKING_DIR --local-scheduler
