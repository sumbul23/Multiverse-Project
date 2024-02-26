#!/bin/bash

cd /dss/work/nola7251/scripts

module load hpc-env/12.2
module load Python/3.10.8-GCCcore-12.2.0

python -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install numpy pandas scikit-learn PyPI PyWavelets Networkx

echo "Virtual environment activated and packages installed."

