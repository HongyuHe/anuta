from multiprocess import Pool, Manager, managers
import numpy as np
import psutil
import sys
from rich import print as pprint
import pandas as pd
from tqdm import tqdm
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import *
from copy import deepcopy
from time import perf_counter

from grammar import AnutaMilli
from constructor import Constructor, Millisampler, Cidds001
from miner import miner, miner_waterfall
from utils import log, save_constraints
import json


anuta : AnutaMilli = None

    
def main(constructor: Constructor, limit: int):
    miner_waterfall(constructor, limit)
    

if __name__ == '__main__':
    # boundsfile = f"./data/meta_bounds.json"
    # filepath = f"./data/meta_w10_s5_{sys.argv[1]}.csv"
    # millisampler = Millisampler(filepath)
    # main(millisampler, sys.argv[1])
    
    filepath = f"data/cidds_wk3_processed.csv"
    cidds = Cidds001(filepath)
    limit = 0
    if sys.argv[1] == 'all':
        limit = cidds.df.shape[0]
        # main(cidds, sys.argv[1])
        # sys.exit(0)
    
    # cidds.df = cidds.df.sample(n=int(sys.argv[1]), random_state=42)
    limit = int(sys.argv[1])
    main(cidds, limit)
    sys.exit(0)