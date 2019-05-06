import fileinput
import glob
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='runs CI test from a folder')
    # Add arguments
    parser.add_argument(
        '-dr', '--direction', type=int, help='1 for True to False and 0 for reverse', required=True)

    args = parser.parse_args()
    direction = args.direction

    return direction


direction = get_args()

for line in fileinput.input(['CI_base.py','MIMIFY_REG.py','MIMIFY_GAN.py','classifier.py','utilities.py'], inplace=True):
    if direction == 0:
        sys.stdout.write(line.replace('use_cuda = False', 'use_cuda = True'))
    else:
        sys.stdout.write(line.replace('use_cuda = True', 'use_cuda = False'))
