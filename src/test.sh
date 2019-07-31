#!/bin/bash

set -e
set -x

python demo.py ctdet --demo $1 --load_model $2 --debug 4
