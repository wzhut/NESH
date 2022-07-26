#!/bin/bash

# dataset crash
python test_SNETD.py -d crash -r 2 -f 1
python test_SNETD.py -d crash -r 2 -f 2
python test_SNETD.py -d crash -r 2 -f 3
python test_SNETD.py -d crash -r 2 -f 4
python test_SNETD.py -d crash -r 2 -f 5

python test_SNETD.py -d crash -r 5 -f 1
python test_SNETD.py -d crash -r 5 -f 2
python test_SNETD.py -d crash -r 5 -f 3
python test_SNETD.py -d crash -r 5 -f 4
python test_SNETD.py -d crash -r 5 -f 5

python test_SNETD.py -d crash -r 8 -f 1
python test_SNETD.py -d crash -r 8 -f 2
python test_SNETD.py -d crash -r 8 -f 3
python test_SNETD.py -d crash -r 8 -f 4
python test_SNETD.py -d crash -r 8 -f 5

python test_SNETD.py -d crash -r 10 -f 1
python test_SNETD.py -d crash -r 10 -f 2
python test_SNETD.py -d crash -r 10 -f 3
python test_SNETD.py -d crash -r 10 -f 4
python test_SNETD.py -d crash -r 10 -f 5

