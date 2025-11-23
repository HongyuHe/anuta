from collections import defaultdict, Counter
from rich import print as pprint
from time import perf_counter
from multiprocess import Pool
from copy import deepcopy
from tqdm import tqdm
from typing import *
import pandas as pd
import numpy as np
import sympy as sp
import z3
import psutil
import warnings
warnings.filterwarnings("ignore")

from anuta.grammar import AnutaMilli, Anuta, DomainType
from anuta.constructor import Constructor, DomainCounter
from anuta.theory import Theory, Constraint
from anuta.utils import log, clausify, save_constraints, z3evalmap
from anuta.cli import FLAGS


anuta : Anuta = None
# #* Load configurations.
# cfg = FLAGS.config
def detector(
    constructor: Constructor,
    path_to_rules: str,
    label: str | int = 0,
    limit: int = 0,
    save: bool = True,
) -> Tuple[float, int, float, float]:
    df = constructor.df
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    domain_kinds: Dict[str, DomainType] = {}
    for varname, domain in constructor.anuta.domains.items():
        domain_kinds[varname] = domain.kind
    
    core_count = psutil.cpu_count() if not FLAGS.cores else FLAGS.cores
    dfpartitions = [partition.reset_index(drop=True) for partition in np.array_split(df, core_count)]
    nworkers = core_count if len(df) > core_count else 1
    
    log.info(f"Spawning {nworkers} workers for detection ...")
    if nworkers > 1:
        args = [(i, partition, path_to_rules, domain_kinds) for i, partition in enumerate(dfpartitions)]
        pool = Pool(core_count)
        log.info("Detecting violations in parallel ...")
        worker_outputs = pool.starmap(detect_partition, args)
        pool.close()
    else:
        worker_outputs = [detect_partition(0, df, path_to_rules, domain_kinds)]
    
    sample_partitions, worker_times = zip(*worker_outputs)
    sample_violations = np.concatenate(sample_partitions)
    total_invalid = int(sample_violations.sum())
    sample_violation_rate = (total_invalid / len(df)) if len(df) else 0.0
    min_runtime = min(worker_times)
    max_runtime = max(worker_times)
    
    log.info(f"Invalid samples detected: {total_invalid}/{len(df)} ({sample_violation_rate:.3%})")
    log.info(f"Detection worker runtime: {min_runtime:.2f}s (min), {max_runtime:.2f}s (max)\n")
    
    if save:
        np.save(f"detector_sample_violations_{constructor.label}_{label}.npy", sample_violations)
    
    return sample_violation_rate, len(df), min_runtime, max_runtime

def _coerce_z3_value(varname: str, value: Any, domaintype: DomainType) -> Optional[z3.ExprRef]:
    if pd.isna(value):
        log.warning(f"[Detector] Encountered NaN for {varname}; marking sample as violation.")
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        if domaintype == DomainType.REAL:
            return z3.RealVal(float(value))
        if isinstance(value, float):
            if not float(value).is_integer():
                log.warning(f"[Detector] Non-integer value {value} for integer var {varname}.")
                return None
            value = int(round(value))
        return z3.IntVal(int(value))
    except (TypeError, ValueError) as exc:
        log.warning(f"[Detector] Failed to coerce {varname}={value}: {exc}")
        return None

def _build_evalmap(domain_kinds: Dict[str, DomainType]) -> Dict[str, Any]:
    evalmap = z3evalmap.copy()
    for varname, domaintype in domain_kinds.items():
        if domaintype == DomainType.REAL:
            evalmap[varname] = z3.Real(varname)
        else:
            evalmap[varname] = z3.Int(varname)
    return evalmap

def detect_partition(
    worker_idx: int,
    dfpartition: pd.DataFrame,
    rulepath: str,
    domain_kinds: Dict[str, DomainType],
) -> Tuple[np.ndarray, float]:
    print(f"Detector worker {worker_idx+1} started.", end='\r')
    evalmap = _build_evalmap(domain_kinds)
    theory = Theory(rulepath, evalmap)
    worker_start = perf_counter()
    sample_violations = np.zeros(len(dfpartition), dtype=int)
    z3varmap = {name: theory.evalmap.get(name) for name in domain_kinds if name in theory.evalmap}
    
    for local_idx, (_, sample) in enumerate(tqdm(
        dfpartition.iterrows(), 
        total=len(dfpartition),
        desc=f"Detector worker {worker_idx+1} processing samples"
    )):
        assignment: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        invalid_assignment = False
        for varname, value in sample.items():
            if varname not in z3varmap:
                continue
            domaintype = domain_kinds.get(varname, DomainType.INTEGER)
            z3value = _coerce_z3_value(varname, value, domaintype)
            if z3value is None:
                invalid_assignment = True
                break
            assignment.append((z3varmap[varname], z3value))
        
        if invalid_assignment:
            sample_violations[local_idx] = 1
            continue
        
        substituted = z3.simplify(z3.substitute(theory._z3theory, assignment))
        if z3.is_false(substituted):
            sample_violations[local_idx] = 1
        elif z3.is_true(substituted):
            continue
        else:
            #* Fallback to solver check
            solver = z3.Solver()
            solver.add(substituted)
            if solver.check() != z3.sat:
                sample_violations[local_idx] = 1
    
    runtime = perf_counter() - worker_start
    log.info(f"Detector worker {worker_idx+1} finished in {runtime:.2f}s.")
    return sample_violations, runtime
    

def validate(
    worker_idx: int, 
    dfpartition: pd.DataFrame,
    rules: List[sp.Expr]
) -> List[int]:
    log.info(f"Worker {worker_idx+1} started.")
    #* >0: Violated, 0: Not violated
    rule_violations = [0 for _ in range(len(rules))]
    sample_violations = [0 for _ in range(len(dfpartition))]
    invalid_samples = set()
    
    for i, sample in tqdm(dfpartition.iterrows(), total=len(dfpartition)):
        assignments = sample.to_dict()
        for k, rule in enumerate(rules):
            # if violations[k]: 
            #     #* This constraint has already been violated.
            #     continue
            #* Evaluate the constraint with the given assignments
            if isinstance(rule, bool):
                #* If the rule is a boolean, it is already evaluated.
                sat = rule
            else:
                sat = rule.subs(assignments)
            try:
                if not sat:
                    rule_violations[k] += 1
                    sample_violations[i] += 1
                    invalid_samples.add(i)
            except Exception as e:
                log.error(f"Error evaluating {rule}:\n{e}")
                pprint("Assignments:", assignments)
                exit(1)
    #* The last value is the number of invalid samples.
    rule_violations.append(len(invalid_samples))
    log.info(f"Worker {worker_idx+1} finished.")
    return rule_violations, sample_violations

def validator(
    constructor: Constructor, 
    rules: List[sp.Expr], 
    label: int=0, 
    save=True
) -> Tuple[float, float]:
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    nworkers = core_count = psutil.cpu_count() if not FLAGS.cores else FLAGS.cores
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    
    if len(constructor.df) <= core_count:
        #* Avoid parallel overhead if the number of samples is small.
        nworkers = 1
    
    log.info(f"Spawning {nworkers} workers for validation ...")
    if nworkers > 1:
        #* Prepare arguments for parallel processing
        args = [(i, df, rules) for i, df in enumerate(dfpartitions)]
        pool = Pool(core_count)
        log.info(f"Validating constraints in parallel ...")
        violations = pool.starmap(validate, args)
        log.info(f"All workers finished.")
        pool.close()
    else:
        violations = validate(0, constructor.df, rules)
    end = perf_counter()
    
    rule_violations, sample_violations = zip(*violations) if nworkers > 1 else violations
    #* Concatenate sample violations from all workers
    sample_violations = np.concatenate(sample_violations) if nworkers > 1 \
        else sample_violations
    log.info(f"Aggregating violations ...")
    aggregated_violations = np.logical_or.reduce(rule_violations) if nworkers > 1 \
        else np.logical_or.reduce([rule_violations, np.zeros(len(rule_violations))])
    aggregated_violations = aggregated_violations[:-1] #* Remove the last value
    aggregated_counts = np.sum(rule_violations, axis=0) if nworkers > 1 \
        else np.sum([rule_violations, np.zeros(len(rule_violations))], axis=0)
    total_invalid_samples = aggregated_counts[-1]
    aggregated_counts = aggregated_counts[:-1] #* Remove the last value
    assert len(aggregated_violations) == len(rules), \
        f"{len(aggregated_violations)=} != {len(rules)=}"
    
    violation_record = Counter(aggregated_violations)
    pprint(violation_record)
    print(f"Violated rules: {violation_record[True]}/{len(aggregated_violations)}")
    print(f"Invalid samples: {total_invalid_samples}/{len(constructor.df)}")
    
    rule_violation_rate = violation_record[True] / len(aggregated_violations)
    sample_violation_rate = total_invalid_samples / len(constructor.df)
    
    violated_rules = [rules[i] for i, is_violated in enumerate(aggregated_violations) if is_violated]
    valid_rules = [rules[i] for i, is_violated in enumerate(aggregated_violations) if not is_violated]
    if save:
        #* Save violated rules
        Theory.save_constraints(violated_rules, f"violated_{constructor.label}_{label}.pl")
        #* Save aggregated violation counts as an array
        np.save(f"violation_counts_{constructor.label}_{label}.npy", aggregated_counts)   
        np.save(f"sample_violations_{constructor.label}_{label}.npy", sample_violations)     
    
    log.info(f"Rule violatioin rate: {rule_violation_rate:.3%}")
    log.info(f"Invalid samples: {sample_violation_rate:.3%}")
    log.info(f"Runtime time: {end-start:.2f}s\n\n")
    
    return rule_violation_rate, sample_violation_rate

def validate_candidates(
        constructor: Constructor,
        test_size: int = 1000,
        iteration_limit: int = 10,
) -> List[Constraint]:
    global anuta
    assert anuta, "anuta is not initialized."
    start = perf_counter()
    
    refdf = constructor.df.copy()
    rules = [constraint.expr for constraint in anuta.kb]
    before_nrules = len(rules)
    
    it = 0
    while it < iteration_limit:
        it += 1
        log.info(f"Validation Iteration: {it}")
        #* Sample a subset of the data
        df = refdf.sample(n=test_size, replace=False, random_state=42)
        refdf.drop(df.index, inplace=True)
        
        constructor.df = df
        violation_rate, valid_rules = validator(constructor, rules, save=False)
        log.info(f"(iteration {it}) Violation rate: {violation_rate:.3%}")
        if violation_rate == 0:
            log.info(f"Finished validating all candidates.")
            break
        else:
            log.info(f"Validating remaining candidates ...")
            rules = valid_rules
    #> End of while loop
    end = perf_counter()
    log.info(f"Validation time (total {it} iters): {end-start:.2f}s\n\n")
    log.info(f"Removed # candidates: {before_nrules - len(rules)}")
    return set([Constraint(rule) for rule in rules])

def test_candidates(
        worker_idx: int, dfpartition: pd.DataFrame, 
        indexset: dict[str, dict[str, np.ndarray]], 
        fcount: dict[str, dict[str, DomainCounter]], 
        limit: int, #* limit ≤ len(dfpartition)
) -> List[int]:
    global anuta
    assert anuta, "anuta is not initialized."
    
    log.info(f"Worker {worker_idx+1} started.")
    #* 1: Violated, 0: Not violated
    violations = [0 for _ in range(len(anuta.candidates))]
    exhausted_values = defaultdict(list)
    exhausted_values[f"Worker {worker_idx}"] = 'Exhausted Domain Values'
    
    for i in tqdm(range(limit), total=limit):
        if FLAGS.config.DOMAIN_COUNTING:
            '''Domain Counting'''
            visited = set()
            index = None
            while True:
                #* Get the vars at every iteration to account for the changes in the indexset.
                indexed_vars = list(fcount.keys())
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
                    del fcount[nxt_var][least_freq_val]
                    exhausted_values[nxt_var].append(least_freq_val)
                    if not fcount[nxt_var]:
                        del fcount[nxt_var]
                if index not in visited:
                    visited.add(index)
                    break
            sample: pd.Series = dfpartition.iloc[index]
        else:
            '''Random Sampling''' 
            sample: pd.Series = dfpartition.sample(random_state=i).iloc[0]
        
        assignments = {}
        for name, val in sample.items():
            var = anuta.variables.get(name)
            if not var: continue
            assignments[var] = val
            
            #* Increment the frequency count of the value of the var.
            if name in fcount:
                if val in fcount[name]:
                    fcount[name][val].count += 1
                elif 'neq' in fcount[name]:
                    #* For numerical values with 'var!=val' index.
                    fcount[name]['neq'].count += 1
        
        for k, constraint in enumerate(anuta.candidates):
            if violations[k]: 
                #* This constraint has already been violated.
                continue
            #* Evaluate the constraint with the given assignments
            if isinstance(constraint, Constraint):
                constraint = constraint.expr
            try:
                sat = constraint.subs(assignments)
                if not sat:
                    violations[k] = 1
            except Exception as e:
                if FLAGS.baseline:
                    #! Temporary fix for the issue with the baseline method 
                    #! which has negation on literals.
                    log.error(f"Error evaluating {constraint}:\n{e}")
                    violations[k] = 1
                else:
                    raise e
            # sat = constraint.subs(assignments)
            # if not sat:
            #     # log.info(f"Violated: {constraint}")
            #     violations[k] = 1
        
    log.info(f"Worker {worker_idx+1} finished.")
    pprint(dict(exhausted_values))
    return violations

def miner_versionspace(constructor: Constructor, refconstructor: Constructor, limit: int):
    global anuta
    #* Use a global var to prevent passing the object to each worker.
    anuta = constructor.anuta
    label = str(limit)
    FLAGS.config.ARITY_LIMIT = 3
    start = perf_counter()
    
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count() if not FLAGS.cores else FLAGS.cores
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    indexsets, fcounts = zip(*[constructor.get_indexset_and_counter(df, anuta.domains) for df in dfpartitions])
    fullindexsets, fullfcounts = constructor.get_indexset_and_counter(constructor.df, anuta.domains)
    
    assert fullindexsets; assert fullfcounts
    pprint(fcounts[0])
    
    while anuta.search_arity <= FLAGS.config.ARITY_LIMIT:
        anuta.propose_new_candidates()
        
        log.info(f"Started testing arity-{anuta.search_arity} constraints.")
        if not anuta.candidates:
            log.warn(f"No new candidates found.")
            break
        
        log.info(f"Testing {len(anuta.candidates)} arity-{anuta.search_arity} candidates ...")
        nworkers = core_count
        if limit <= core_count or len(anuta.candidates) <= core_count:
            #* Avoid parallel overhead if the number of candidates is small.
            nworkers = 1
        log.info(f"Spawning {nworkers} workers ...")
        if nworkers > 1:
            #* Prepare arguments for parallel processing
            args = [(i, df, indexset, fcount, limit//nworkers) 
                    for i, (df, indexset, fcount) in enumerate(
                        # zip(dfpartitions, deepcopy(indexsets), deepcopy(fcounts)))]
                        zip(dfpartitions, deepcopy(indexsets), deepcopy(fcounts)))]
                        #? Create new DC counters or keep the counts from the level?
                        #^ I think we should NOT keep the counts from the previous level,
                        #^ because an unviolated example could eliminate a candidate on the next level.
            pool = Pool(core_count)
            log.info(f"Testing arity-{anuta.search_arity} constraint in parallel ...")
            violation_indices = pool.starmap(test_candidates, args)
            log.info(f"All workers finished.")
            pool.close()
        else:
            violation_indices = test_candidates(0, constructor.df, fullindexsets, fullfcounts, limit)
        
        log.info(f"Aggregating violations ...")
        aggregated_violations = np.logical_or.reduce(violation_indices) \
            if nworkers > 1 else violation_indices
        assert len(aggregated_violations) == len(anuta.candidates), \
            f"{len(aggregated_violations)=} != {len(anuta.candidates)=}"
        pprint(Counter(aggregated_violations))
        
        log.info(f"Rejecting candidates ...")
        old_size = len(anuta.kb)
        new_candidates = set()
        for idx, is_violated in enumerate(aggregated_violations):
            candidate = anuta.candidates[idx]
            if is_violated:
                #* If the more specific stem is violated, use them to 
                #* construct longer (more general) constraints.
                new_candidates.add(candidate)
                anuta.num_candidates_rejected += 1
            else:
                #* Learn a valid constraint (dedupe is handled by set).
                anuta.kb.add(candidate)

        # log.info(f"Removed {len(anuta.candidates)-len(new_candidates)} candidates.")
        #TODO: Don't reuse `candidates`, use `rejected`.
        #! Convert to a list to preserve order!!!
        anuta.candidates = list(new_candidates)
        new_size = len(anuta.kb)
        log.info(f"Learned {new_size-old_size} candidates.")
    #> End of while loop
    end = perf_counter()
    log.info(f"Finished mining all constraints up to arity {anuta.search_arity}.")

    print(f"Total proposed: {anuta.num_candidates_proposed}")
    print(f"Total rejected: {anuta.num_candidates_rejected} ({anuta.num_candidates_rejected/anuta.num_candidates_proposed:.2%})")
    print(f"Total prior: {len(anuta.prior)}")
    initial_learned = len(anuta.kb)
    Theory.save_constraints(anuta.kb | anuta.prior, f'learned_{constructor.label}_{label}.pl')
    
    anuta.kb = validate_candidates(refconstructor)
    print(f"Total proposed: {anuta.num_candidates_proposed}")
    print(f"Initial learned: {initial_learned} ({initial_learned/anuta.num_candidates_proposed:.2%})")
    print(f"Final learned: {len(anuta.kb)} ({len(anuta.kb)/anuta.num_candidates_proposed:.2%})")
    
    #* Prior: [(X!=2 & X!=3 & ...), (Y=500 | Y=400 | ...)]
    Theory.save_constraints(anuta.kb | anuta.prior, f'learned_{constructor.label}_{label}_checked.pl')
    print(f"Runtime: {end-start:.2f}s\n\n")
    
    # if len(anuta.kb) <= 200: 
    #     #^ Skip pruning if the number of constraints is too large
    #     log.info(f"Pruning redundant constraints ...")
    #     start = perf_counter()
    #     # assumptions = [v >= 0 for v in anuta.variables.values()]
    #     assumptions = anuta.prior
    #     cnf = sp.And(*(anuta.kb | set(assumptions)))
    #     simplified_logic = sp.to_cnf(clausify(cnf))
    #     reduced_kb = list(simplified_logic.args) \
    #         if isinstance(simplified_logic, sp.And) else [simplified_logic]
    #     pruned_count = len(anuta.kb) - len(reduced_kb)
    #     end = perf_counter()
    #     print(f"{len(anuta.kb)=}, {len(reduced_kb)=} ({pruned_count=})\n")
    #     print(f"Pruning time: {end-start:.2f}s\n\n")
        
    #     Theory.save_constraints(anuta.kb, f'pruned_{label}_a{FLAGS.config.ARITY_LIMIT}.pl')

def miner_valiant(constructor: Constructor, limit: int = 0):
    global anuta
    label = str(limit)
    anuta = constructor.anuta
    #* Generate all possible candidates in one go.
    anuta.populate_kb()
    
    start = perf_counter()
    #* Prepare arguments for parallel processing
    core_count = psutil.cpu_count() if not FLAGS.cores else FLAGS.cores
    # core_count = 1
    log.info(f"Spawning {core_count} workers ...")
    # mutex = Manager().Lock()
    dfpartitions = [df.reset_index(drop=True) for df in np.array_split(constructor.df, core_count)]
    indexsets, fcounts = zip(*[constructor.get_indexset_and_counter(df, anuta.domains) for df in dfpartitions])
    args = [(i, df, indexset, fcount, limit//core_count) 
            for i, (df, indexset, fcount) in enumerate(zip(dfpartitions, indexsets, fcounts))]
    pprint(fcounts[0])
    
    print(f"Testing constraints in parallel ...")
    pool = Pool(core_count)
    # violation_indices, bounds_array = pool.starmap(test_constraints, args)
    # violation_indices = pool.starmap(test_millisampler_constraints, args)
    violation_indices = pool.starmap(test_candidates, args)
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
            learned_kb.append(anuta.candidates[index])
    
    end = perf_counter()
    # log.info(f"Aggregating bounds ...")
    #* Update the bounds based on the learned bounds
    # for bounds in bounds_array:
    #     for k, v in bounds.items():
    #         if v.lb < aggregated_bounds[k].lb:
    #             aggregated_bounds[k].lb = v.lb
    #         if v.ub > aggregated_bounds[k].ub:
    #             aggregated_bounds[k].ub = v.ub

    removed_count = len(anuta.candidates) - len(learned_kb)
    # pprint(aggregated_bounds)
    print(f"{len(learned_kb)=}, {len(anuta.candidates)=} ({removed_count=})")
    Theory.save_constraints(learned_kb, f'learned_{constructor.label}_{label}.pl')
    print(f"Learning time: {end-start:.2f}s\n\n")
    
    if len(learned_kb) <= 200: 
        #* Skip pruning if the number of constraints is too large
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
        Theory.save_constraints(anuta.learned_kb, f'reduced_{constructor.label}_{label}.pl')
        

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
