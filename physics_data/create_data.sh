#!/bin/bash

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
OUTPUT_SCENES_DIR=ours_default_scenes
OUTPUT_DATA_DIR=ours_default_data

mkdir $OUTPUT_SCENES_DIR

# This script is purely sequential but it is recommended to parallelize the
# following loop, which generates the simulation data.
for seed in `seq 3001 3150`; do
        python create_physics_data.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed

done

for seed in `seq 3151 3240`; do
        python create_physics_data.py --output $OUTPUT_SCENES_DIR \
                                      --seed $seed \
                                      --obstacle
done


# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
python create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR
