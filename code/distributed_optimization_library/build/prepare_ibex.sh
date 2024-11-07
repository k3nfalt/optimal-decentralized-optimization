#!/bin/bash
module load singularity
singularity build ../python_ml.sif docker://k3nfalt/python_ml
singularity shell ../python_ml.sif
