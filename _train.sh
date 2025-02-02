#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Train tf 
print_header "Training network"
cd $DIR

# Begin experiment
python3.6 main.py \
--env-name "IPD-v0" \
--n-lookahead 0

python3.6 main.py \
--env-name "IPD-v0" \
--n-lookahead 1

python3.6 main.py \
--env-name "IPD-v0" \
--n-lookahead 2
