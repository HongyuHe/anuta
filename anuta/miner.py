from collections import defaultdict
from rich import print as pprint
from time import perf_counter
from multiprocess import Pool
from tqdm import tqdm
from typing import *
import pandas as pd
import numpy as np
import sympy as sp
import psutil

from grammar import AnutaMilli, Anuta
from constructor import Constructor, DomainCounter
from model import Model
from utils import log, save_constraints


anuta : Anuta = None
window = 10

def test_millisampler_constraints(worker_idx: int, dfpartition: pd.DataFrame) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    # var_bounds = anuta.bounds
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.initial_kb))]
    
    for i, sample in tqdm(dfpartition.iterrows(), total=len(dfpartition)):
        # print(f"Testing with sample {j+1}/{len(dfpartition)}.", end='\r')
        canary_max = sample.iloc[-window:].max()
        canary_premise = i % 2 #* 1: old index, 0: even index
        canary_conclusion = anuta.constants['burst_threshold'] + 1 if canary_premise else 0
        assignments = {}
        #* Assign the canary variables
        for cannary in anuta.canary_vars:
            if 'max' in cannary.name:
                assignments[cannary] = canary_max
            elif 'premise' in cannary.name:
                assignments[cannary] = canary_premise
            elif 'conclusion' in cannary.name:
                assignments[cannary] = canary_conclusion
        for name, val in sample.items():
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
            
            # bounds = var_bounds.get(name)
            # if bounds and (val < bounds.lb or val > bounds.ub):
            #     #* Learn the bounds
            #     # mutex.acquire()
            #     # print(f"Updating bounds for {name}: [{bounds.lb}, {bounds.ub}]")
            #     var_bounds[name] = IntBounds(
            #         min(bounds.lb, val), max(bounds.ub, val))
            #     # print(f"Updated bounds for {name}: [{var_bounds[name].lb}, {var_bounds[name].ub}]")
            #     # mutex.release()
        
        for k, constraint in enumerate(anuta.initial_kb):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.subs(assignments | anuta.constants)
            if not sat:
                violations[k] = 1
    log.info(f"Worker {worker_idx+1} finished.")
    return violations

def test_cidds_constraints(
        worker_idx: int, dfpartition: pd.DataFrame, 
        indexset: dict[str, dict[str, np.ndarray]], 
        fcount: dict[str, dict[str, DomainCounter]], 
        limit: int,
        #^ limit ≤ len(dfpartition)
    ) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    # var_bounds = anuta.bounds
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.initial_kb))]
    exhausted_values = defaultdict(list)
    exhausted_values[f"Worker {worker_idx}"] = 'Exhausted Domain Values'
    
    for i in tqdm(range(limit), total=limit):
        '''Domain Counting'''
        indexed_vars = list(fcount.keys())
        #^ Get the vars at every iteration to account for the changes in the indexset.
        #* Cycle through the vars, treating them equally (no bias).
        nxt_var = indexed_vars[i % len(indexed_vars)]
        #* Find the least frequent value of the next variable.
        least_freq_val = min(fcount[nxt_var], key=fcount[nxt_var].get)
        #& Get the 1st from the indices of least frequent value (inductive bias).
        #TODO: Choose randomly from the indices?
        indices = indexset[nxt_var][least_freq_val]
        #^ Somehow ndarray passes by value (unlike list) ...
        index, indexset[nxt_var][least_freq_val] = indices[0], indices[1: ]
        if indexset[nxt_var][least_freq_val].size == 0:
            #* Remove the corresponding counter if the value is exhausted 
            #* to prevent further sampling (from empty sets).
            # log.info(f"Exhausted {nxt_var}={least_freq_val}.")
            del fcount[nxt_var][least_freq_val]
            # del indexset[nxt_var][least_freq_val]
            exhausted_values[nxt_var].append(least_freq_val)
            if not fcount[nxt_var]:
                # log.info(f"Exhausted all values of {nxt_var}.")
                del fcount[nxt_var]
                # del indexset[nxt_var]
        
        sample = dfpartition.iloc[index]
        assignments = {}
        for name, val in sample.items():
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
            
            #* Increment the frequency count of the value of the var.
            if name in fcount and val in fcount[name]:
                fcount[name][val].count += 1
        
        # pprint(assignments)
        for k, constraint in enumerate(anuta.initial_kb):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            sat = constraint.subs(assignments)
            if not sat:
                violations[k] = 1
        
    log.info(f"Worker {worker_idx+1} finished.")
    # pprint(fcount)
    pprint(exhausted_values)
    return violations


def miner(constructor: Constructor, limit: int = 0):
    global anuta
    label = str(limit)
    anuta = constructor.anuta
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    log.info(f"Spawning {core_count} workers ...")
    # mutex = Manager().Lock()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    indexsets, fcounts = zip(*[constructor.get_indexset_and_counter(df) for df in dfpartitions])
    args = [(i, df, indexset, fcount, limit//core_count) 
            for i, (df, indexset, fcount) in enumerate(zip(dfpartitions, indexsets, fcounts))]
    pprint(fcounts[0])
    
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    # violation_indices, bounds_array = pool.starmap(test_constraints, args)
    # violation_indices = pool.starmap(test_millisampler_constraints, args)
    violation_indices = pool.starmap(test_cidds_constraints, args)
    # violation_indices = [r[0] for r in results]
    # bounds_array = [r[1] for r in results]
    # pool.close()

    log.info(f"All workers finished.")
    
    aggregated_violations = np.logical_or.reduce(violation_indices)
    # aggregated_bounds = {k: IntBounds(sys.maxsize, 0) for k in anuta.bounds.keys()}
    learned_kb = []
    log.info(f"Aggregating violations ...")
    #* Update learned_kb based on the violated constraints
    for index, is_violated in tqdm(enumerate(aggregated_violations), total=len(aggregated_violations)):
        if not is_violated:
            learned_kb.append(anuta.initial_kb[index])
    
    end = perf_counter()
    # log.info(f"Aggregating bounds ...")
    #* Update the bounds based on the learned bounds
    # for bounds in bounds_array:
    #     for k, v in bounds.items():
    #         if v.lb < aggregated_bounds[k].lb:
    #             aggregated_bounds[k].lb = v.lb
    #         if v.ub > aggregated_bounds[k].ub:
    #             aggregated_bounds[k].ub = v.ub

    removed_count = len(anuta.initial_kb) - len(learned_kb)
    # pprint(aggregated_bounds)
    print(f"{len(learned_kb)=}, {len(anuta.initial_kb)=} ({removed_count=})")
    Model.save_constraints(learned_kb + anuta.prior_kb, f'learned_{label}')
    print(f"Learning time: {end-start:.2f}s\n\n")
    
    if len(learned_kb) > 200: 
        #* Skip pruning if the number of constraints is too large
        return
    
    start = perf_counter()
    log.info(f"Pruning redundant constraints ...")
    # assumptions = [v >= 0 for v in anuta.variables.values()]
    assumptions = []
    cnf = sp.And(*(learned_kb + assumptions))
    simplified_logic = cnf.simplify()
    reduced_kb = list(simplified_logic.args) \
        if isinstance(simplified_logic, sp.And) else [simplified_logic]
    filtered_count = len(learned_kb) - len(reduced_kb)
    end = perf_counter()
    print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    print(f"Pruning time: {end-start:.2f}s\n\n")
    
    anuta.learned_kb = reduced_kb + anuta.prior_kb
    Model.save_constraints(anuta.learned_kb, f'reduced_{label}')
    
def miner_waterfall(constructor: Constructor, limit: int = 0):
    global anuta
    label = str(limit)
    anuta = constructor.anuta
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count()
    log.info(f"Spawning {core_count} workers ...")
    # mutex = Manager().Lock()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    indexsets, fcounts = zip(*[constructor.get_indexset_and_counter(df) for df in dfpartitions])
    args = [(i, df, indexset, fcount, limit//core_count) 
            for i, (df, indexset, fcount) in enumerate(zip(dfpartitions, indexsets, fcounts))]
    pprint(fcounts[0])
    
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    # violation_indices, bounds_array = pool.starmap(test_constraints, args)
    # violation_indices = pool.starmap(test_millisampler_constraints, args)
    violation_indices = pool.starmap(test_cidds_constraints, args)
    # violation_indices = [r[0] for r in results]
    # bounds_array = [r[1] for r in results]
    # pool.close()

    log.info(f"All workers finished.")
    
    aggregated_violations = np.logical_or.reduce(violation_indices)
    # aggregated_bounds = {k: IntBounds(sys.maxsize, 0) for k in anuta.bounds.keys()}
    learned_kb = []
    log.info(f"Aggregating violations ...")
    #* Update learned_kb based on the violated constraints
    for index, is_violated in tqdm(enumerate(aggregated_violations), total=len(aggregated_violations)):
        if not is_violated:
            learned_kb.append(anuta.initial_kb[index])
    
    end = perf_counter()
    # log.info(f"Aggregating bounds ...")
    #* Update the bounds based on the learned bounds
    # for bounds in bounds_array:
    #     for k, v in bounds.items():
    #         if v.lb < aggregated_bounds[k].lb:
    #             aggregated_bounds[k].lb = v.lb
    #         if v.ub > aggregated_bounds[k].ub:
    #             aggregated_bounds[k].ub = v.ub

    removed_count = len(anuta.initial_kb) - len(learned_kb)
    # pprint(aggregated_bounds)
    print(f"{len(learned_kb)=}, {len(anuta.initial_kb)=} ({removed_count=})")
    Model.save_constraints(learned_kb + anuta.prior_kb, f'learned_{label}.rule')
    print(f"Learning time: {end-start:.2f}s\n\n")
    
    if len(learned_kb) > 200: 
        #* Skip pruning if the number of constraints is too large
        return
    
    start = perf_counter()
    log.info(f"Pruning redundant constraints ...")
    # assumptions = [v >= 0 for v in anuta.variables.values()]
    assumptions = []
    cnf = sp.And(*(learned_kb + assumptions))
    simplified_logic = cnf.simplify()
    reduced_kb = list(simplified_logic.args) \
        if isinstance(simplified_logic, sp.And) else [simplified_logic]
    filtered_count = len(learned_kb) - len(reduced_kb)
    end = perf_counter()
    print(f"{len(learned_kb)=}, {len(reduced_kb)=} ({filtered_count=})\n")
    print(f"Pruning time: {end-start:.2f}s\n\n")
    
    anuta.learned_kb = reduced_kb + anuta.prior_kb
    Model.save_constraints(anuta.learned_kb, f'reduced_{label}.rule')