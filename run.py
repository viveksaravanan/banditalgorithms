#!/usr/bin/env python

from matplotlib import pyplot as plt
import sys
import os 

sys.path.insert(0, 'src')
from etc import etc_plot

def main(targets):
    if 'test' in targets:
        mu2s = [0.01 * d for d in range(0,101)]
        etc_plot(0, mu2s, 25, False, 1000, 1000)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)