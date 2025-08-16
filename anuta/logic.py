from collections import defaultdict
from itertools import combinations
from time import perf_counter
from typing import *
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import psutil
import sympy as sp
from dataclasses import dataclass
from enum import Enum, auto
from rich import print as pprint
import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore")

from anuta.grammar import TYPE_DOMIAN, ConstantType, DomainType, VariableType, group_variables_by_type
from anuta.utils import log
from anuta.theory import Constraint, Theory
from anuta.constructor import Constructor


class LogicLearner(object):
    
    def __init__(self, constructor: Constructor, limit: int = None):
        if limit and limit < constructor.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            constructor.df = constructor.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = 'all'
        
        self.dataset = constructor.label
        self.examples: pd.DataFrame = constructor.df
        self.variables: List[str] = [
            var for var in self.examples.columns
        ]
        self.domains = constructor.anuta.domains
        self.constants = constructor.anuta.constants
        
        self.categoricals = [] # constructor.categoricals
        self.prior: Set[Constraint] = set()
    
    def learn(
        self,
        max_size: Optional[int] = 10,
        max_solutions: Optional[int] = 50_000,
    ):
        """
        1) Create predicate space (already intra-tuple).
        2) Build intra-tuple evidence sets: for each tuple t, SAT(t) = { P | t |= P }.
        3) Enumerate all minimal hitting sets with bounded DFS/BnB.
        4) Return rules: Or(p in cover)
        """
        predicates: Set[Constraint] = self.generate_predicates()

        start = perf_counter()
        evidence_sets: List[frozenset[Constraint]] = self.build_evidence_set(predicates)
        end = perf_counter()
        log.info(f"\nBuilt {len(evidence_sets)} evidence sets in {end - start:.2f} seconds.")

        if not evidence_sets:
            log.warning("No non-empty evidence sets were produced; returning only prior.")

        start = perf_counter()
        covers = self.enumerate_minimal_hitting_sets(
            evidence_sets=evidence_sets,
            max_size=max_size,
            max_solutions=max_solutions,
        )
        end = perf_counter()
        log.info(f"Enumerated {len(covers)} minimal hitting sets in {end - start:.2f} seconds.")
        
        learned_rules: Set[Constraint] = set()
        num_trivial = 0
        for cover in tqdm(covers, desc="... Collecting learned rules"):
            rule = sp.Or(*[c.expr for c in cover])
            #* Remove trivial rules (True/False).
            if rule not in [sp.true, sp.false]:
                learned_rules.add(Constraint(rule))
            else:
                num_trivial += 1
                
        num_learned = len(learned_rules)
        log.info(f"Learned {num_learned} rules.")
        log.info(f"Removed {num_trivial} trivial rules (True/False).")
        
        learned_rules |= self.prior
        log.info(f"Total {len(learned_rules)} rules (including prior).")
        Theory.save_constraints(learned_rules, f'denial_{self.dataset}_{self.num_examples}.pl')
        return
        
        
    def enumerate_minimal_hitting_sets(
        self,
        evidence_sets: List[frozenset[Constraint]],
        max_size: Optional[int] = 10,
        max_solutions: Optional[int] = 1_000_000,
    ) -> List[Set[Constraint]]:
        """
        Enumerate all minimal hitting sets H such that ∀E in evidence_sets: H ∩ E ≠ ∅.

        Parameters
        ----------
        evidence_sets : list of frozenset[Constraint]
            Non-empty evidence sets (one per tuple).
        max_size : Optional[int]
            Upper bound on the size of hitting sets (branch-and-bound). If None, no size bound.
        max_solutions : Optional[int]
            Stop after enumerating this many minimal solutions (optional).

        Returns
        -------
        List[Set[Constraint]]
            All minimal hitting sets found (as sets of Constraint).
        """
        if not evidence_sets:
            return []

        # Index: for each predicate, which evidence indices contain it?
        idx_by_pred: Dict[Constraint, Set[int]] = defaultdict(set)
        for i, E in enumerate(evidence_sets):
            for p in E:
                idx_by_pred[p].add(i)

        # Precompute each evidence set as a list to iterate deterministically
        E_list: List[List[Constraint]] = [list(E) for E in evidence_sets]

        # Optional predicate ordering: higher coverage first (helps pruning)
        pred_by_cov = sorted(idx_by_pred.items(), key=lambda kv: -len(kv[1]))
        pred_order = [p for p, _ in pred_by_cov]

        solutions: List[frozenset[Constraint]] = []

        # Quick helper: prune if chosen is a superset of an existing minimal solution.
        def dominated_by_existing(chosen: Set[Constraint]) -> bool:
            fc = frozenset(chosen)
            for sol in solutions:
                # if chosen is a superset of a found minimal solution, it can’t be minimal
                if sol.issubset(fc):
                    return True
            return False

        # Check coverage
        FULL = set(range(len(E_list)))

        def covered_indices(chosen: Set[Constraint]) -> Set[int]:
            covered = set()
            for p in chosen:
                covered |= idx_by_pred[p]
            return covered

        # Optimistic bound on remaining picks:
        # If we still need to cover R uncovered evidences, and the best single predicate
        # can cover at most M of them, then we need at least ceil(|R|/M) more predicates.
        # If chosen_size + that lower bound > max_size, prune.
        # (Simple but effective for a BnB cut.)
        def optimistic_lb(uncovered: Set[int]) -> int:
            if not uncovered:
                return 0
            best_gain = 0
            for p in pred_order:
                gain = len(uncovered & idx_by_pred[p])
                if gain > best_gain:
                    best_gain = gain
                    if best_gain == len(uncovered):
                        break
            return 1 if best_gain == 0 else ( (len(uncovered) + best_gain - 1) // best_gain )

        # Choose an uncovered evidence with the fewest candidates (fail-first)
        def pick_uncovered_with_smallest_branch(uncovered: Set[int], chosen: Set[Constraint]) -> int:
            best_i, best_deg = None, 10**9
            for i in uncovered:
                # candidates are predicates in E[i]; we can skip those already in chosen,
                # but keeping them is fine — small effect on degree.
                deg = len(E_list[i])
                if deg < best_deg:
                    best_deg = deg
                    best_i = i
            return best_i  # index in E_list

        # DFS
        def dfs(chosen: Set[Constraint], covered: Set[int]):
            # Early domination prune
            if dominated_by_existing(chosen):
                return

            # Covered all evidences? record a minimal solution
            if covered == FULL:
                fc = frozenset(chosen)
                # Keep only minimal solutions: drop any existing supersets
                keep: List[frozenset[Constraint]] = []
                for s in solutions:
                    if s.issubset(fc):
                        # existing is smaller/equal; then current can't be minimal if strictly superset
                        if s != fc:
                            return
                    elif fc.issubset(s):
                        # new is smaller; drop the old superset
                        continue
                    keep.append(s)
                solutions.clear()
                solutions.extend(keep)
                solutions.append(fc)
                progress.update(1)

                # Optional cap
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return
                return

            # BnB size bound
            if max_size is not None and len(chosen) >= max_size:
                return

            # Remaining uncovered
            uncovered = FULL - covered

            # BnB optimistic bound
            if max_size is not None:
                lb = optimistic_lb(uncovered)
                if len(chosen) + lb > max_size:
                    return

            # Branch on the uncovered evidence with the smallest size (fail-first)
            pivot = pick_uncovered_with_smallest_branch(uncovered, chosen)
            # Order candidate predicates by descending marginal gain
            cand = sorted(
                E_list[pivot],
                key=lambda p: -len((idx_by_pred[p]) & uncovered)
            )

            for p in cand:
                if p in chosen:
                    # Already picked (rare in this branch), but continue recursion to avoid duplicates
                    new_chosen = chosen
                    new_covered = covered
                else:
                    new_chosen = set(chosen)
                    new_chosen.add(p)
                    new_covered = covered | idx_by_pred[p]

                # Another quick domination check before descending
                if dominated_by_existing(new_chosen):
                    continue

                dfs(new_chosen, new_covered)

                # Optional early stop if we reached the cap
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return

        progress = tqdm(desc="... Found minimal hitting sets", unit=" sets", total=max_solutions)
        dfs(set(), set())

        # Convert back to list[set]
        return [set(s) for s in solutions]
    
    def build_evidence_set(self, predicates: Set[Constraint]) -> List[Set[Constraint]]:
        # Pre-pack predicates into a list so all workers use same reference
        predicates_list = list(predicates)

        def process_chunk(worker_id, df_chunk: pd.DataFrame) -> List[Set[Constraint]]:
            chunk_results = []
            for _, row in tqdm(df_chunk.iterrows(), total=df_chunk.shape[0], desc="... Building evidence sets"):
                row_dict = row.to_dict()
                satisfied = set()
                for p in predicates_list:
                    try:
                        if bool(p.expr.subs(row_dict)):
                            satisfied.add(p)
                    except Exception:
                        # Any substitution/eval failure = not satisfied
                        continue
                chunk_results.append(satisfied)
            log.info(f"Worker {worker_id} finished.")
            return chunk_results

        n_jobs = psutil.cpu_count()

        # Split into n_jobs chunks (avoid overhead of too many tasks)
        chunks = np.array_split(self.examples, n_jobs)
        log.info(f"Processing {len(chunks)} chunks with {n_jobs} workers.")

        # Parallel execution — each worker gets a big chunk
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_chunk)(i, chunk)
            for i, chunk in enumerate(chunks)
        )

        # Flatten results (list of lists)
        return [s for chunk in results for s in chunk]


    def generate_predicates(self) -> Set[Constraint]:
        typed_variables, grouped_variables = group_variables_by_type(self.variables)
        
        prior_rules: Set[str] = set()
        #* Collecting categorical variables (with >1 unique value).
        for varname in grouped_variables[DomainType.CATEGORICAL]:
            if self.examples[varname].nunique() > 1:
                #* Only consider categorical variables with more than one unique value.
                self.categoricals.append(varname)
            else:
                #* Neglect variables with only one unique value but add them to prior rules.
                self.variables.remove(varname)
                varval = self.examples[varname].iloc[0].item()
                prior_rules.add(f"Eq({varname}, {varval})")
                
        #* Variables -> their types.
        variable_types = {}
        for vtype, tvars in typed_variables.items():
            for varname in tvars:
                variable_types[varname] = vtype
        
        '''Augment the variables with constants.'''
        avars = set()
        for varname, consts in self.constants.items():
            if varname in self.categoricals:
                #* Only augment numerical variables with constants.
                continue
            vtype = variable_types[varname]
            #& X*c 
            if consts.kind == ConstantType.SCALAR:
                for const in consts.values:
                    if const == 1: continue
                    avar = f"{const}$*${varname}"
                    avars.add(avar)
                    variable_types[avar] = vtype
            #& X+c
            if consts.kind == ConstantType.ADDITION:
                for const in consts.values:
                    avar = f"{const}$+${varname}"
                    avars.add(avar)
                    variable_types[avar] = vtype
                    
                # #* default: X+1
                # avar = f"1+{varname}"
                # variable_types[avar] = vtype
        
        num_avars = len(avars)
        log.info(f"Created {len(avars)} augmented variables.")
        
        '''Compound variables'''
        #& All pairs of X+Y and (some) X*Y
        for vtype in typed_variables:
            domaintype = TYPE_DOMIAN[vtype]
            if domaintype != DomainType.NUMERICAL: 
                #* Only augment numerical vars.
                continue
            
            numericals = typed_variables[vtype]
            for i, var1 in enumerate(numericals):
                for var2 in numericals[i+1:]:
                    if var1 == var2: continue
                    avar = f"{var1}$+${var2}"
                    avars.add(avar)
                    variable_types[avar] = vtype
                    
                    if vtype == VariableType.WINDOW:
                        avar = f"{var1}$*${var2}"
                        avars.add(avar)
                        variable_types[avar] = vtype
        
        log.info(f"Created {len(avars)-num_avars} compund variables.")
        
        '''Create predicates for all variables.'''
        varvalues = self.examples.to_dict(orient='series')
        predicates = set()
        
        #! Restrict LHS to single variables.
        for j, lhs in enumerate(self.variables):
            vtype_lhs = variable_types[lhs]
            domaintype_lhs = TYPE_DOMIAN[vtype_lhs]
            lhs_vars = set([lhs])
            
            #& Predicates with constants.
            if domaintype_lhs == DomainType.CATEGORICAL:
                #! Assuming one variable has only one type of constants.
                if (lhs in self.constants 
                    and self.constants[lhs].kind == ConstantType.ASSIGNMENT):
                    #& X=c
                    for constant in self.constants[lhs].values:
                        predicates.add(f"Eq({lhs}, {constant})")
                        predicates.add(f"Ne({lhs}, {constant})")
            else:
                #& DomainType.NUMERICAL
                if (lhs in self.constants
                    and self.constants[lhs].kind == ConstantType.LIMIT):
                    #& X ≤ c and X > c
                    for constant in self.constants[lhs].values:
                        predicates.add(f"({lhs} >= {constant})")
                        predicates.add(f"({lhs} < {constant})")
                        #? Since we don't care equality (X=c), we omit the following:
                        # predicates.add(f"({lhs} <= {constant})")
                        # predicates.add(f"({lhs} > {constant})")
            
            #* Allow all types of variables in the RHS.
            #! Order matters with `j`
            all_vars = self.variables + list(avars)
            for rhs in all_vars[j+1: ]:
                assert lhs != rhs, "LHS and RHS cannot be the same variable."
                vtype_rhs = variable_types[rhs]
                domaintype_rhs = TYPE_DOMIAN[vtype_rhs]
                rhs_vars = set()
                if vtype_lhs != vtype_rhs:
                    #* Only consider same type variables in the RHS.
                    continue

                if "$" not in rhs:
                    rhs_vars.add(rhs)
                else:
                    v1, op2, v2 = rhs.split('$')
                    rhs = rhs.replace('$', '')
                    rhs_vars.add(v2)
                    
                    const = None
                    if v1 not in self.variables:
                        #* v1 is a constant
                        const = int(v1)
                    else:
                        rhs_vars.add(v1)
                    
                    if rhs not in self.examples.columns:
                        rhs_values: pd.Series = eval(rhs, {}, varvalues)
                        if rhs_values.nunique() <= 1:
                            #* Skip abstract variables with only one unique value.
                            #* Add to prior rules.
                            prior_rules.add(f"Eq({rhs},{rhs_values.iloc[0]})")
                            continue
                        else:
                            self.examples[rhs] = rhs_values
                #^ End if "$" not in rhs
                
                if lhs_vars & rhs_vars:
                    #* Don't generate predicates with overlapping variables.
                    continue
                
                '''Create predicates for the LHS and RHS.'''
                #& Equality predicates: Eq(A,B)
                predicate = f"Eq({lhs},{rhs})"
                predicate_values = (self.examples[lhs]==self.examples[rhs]).astype(int)
                #* Check uniqueness -> add as priors
                if predicate_values.nunique() > 1:
                    self.examples[predicate] = predicate_values
                    predicates.add(predicate)
                else:
                    #* Invert the predicate if it's always false.
                    if predicate_values.iloc[0] == 0:
                        predicate = f"Ne({lhs},{rhs})"
                    prior_rules.add(predicate)
                    #* Skip the rest of the predicates with this LHS.
                    continue
                    
                #& Inequality predicates: Ne(A,B)
                predicate = f"Ne({lhs},{rhs})"
                predicate_values = (self.examples[lhs]!=self.examples[rhs]).astype(int)
                #* Should have >1 unique value to this point.
                assert predicate_values.nunique() > 1
                self.examples[predicate] = predicate_values
                predicates.add(predicate)
                
                if domaintype_lhs == DomainType.NUMERICAL:
                    assert domaintype_rhs == DomainType.NUMERICAL, \
                        "LHS and RHS must have the same domain type."
                    #& Comparison predicates: A≥B
                    predicate = f"({lhs}>={rhs})"
                    predicate_values = (self.examples[lhs]>=self.examples[rhs]).astype(int)
                    if predicate_values.nunique() > 1:
                        self.examples[predicate] = predicate_values
                        predicates.add(predicate)
                    else:
                        if predicate_values.iloc[0] == 0:
                            predicate = f"({lhs}<{rhs})"
                        prior_rules.add(predicate)
                        continue
                    
                    #& Comparison predicates: A<B
                    predicate = f"({lhs}<{rhs})"
                    predicate_values = (self.examples[lhs]<self.examples[rhs]).astype(int)
                    assert predicate_values.nunique() > 1, \
                        "Predicate should have >1 unique value to this point."
                    self.examples[predicate] = predicate_values
                    predicates.add(predicate)
        #^ End for j, lhs in enumerate(self.variables)
        
        '''Add domain constraints to prior rules.'''
        for varname in self.variables:
            domaintype = TYPE_DOMIAN[variable_types[varname]]
            if domaintype == DomainType.NUMERICAL:
                bounds = self.domains[varname].bounds
                prior_rules.add(f"({varname}>={bounds.lb})")
                prior_rules.add(f"({varname}<={bounds.ub})")
                
                #& Add default predicate X>0 if the domain contains 0.
                if bounds.lb <= 0 <= bounds.ub:
                    #* Add X>0 as a default if the domain contains 0.
                    predicate = f"({varname}>0)"
                    predicate_values = (self.examples[varname]>0).astype(int)
                    if predicate_values.nunique() > 1:
                        predicates.add(predicate)
                    else:
                        if predicate_values.iloc[0] == 0:
                            predicate = f"Not({predicate})"
                        prior_rules.add(predicate)
                
            elif domaintype == DomainType.CATEGORICAL:
                prior_rules.add(f"{varname}>=0")
                prior_rules.add(f"{varname}<={max(self.domains[varname].values)}")
                
                #& Add negative prior for missing values.
                unique_values = set(self.examples[varname].unique())
                domain_values = set(self.domains[varname].values)
                missing_values = domain_values - unique_values
                ne_predicates = []
                for value in missing_values:
                    ne_predicates.append(f"Ne({varname},{value})")
                #* Don't add negative assumptions for port variables (too many).
                keywords = ['pt', 'port']
                if ne_predicates and not any(keyword in varname.lower() for keyword in keywords):
                    prior_rules.add(' & '.join(ne_predicates))
                    
        log.info(f"Created {len(predicates)} predicates. First a few:")
        pprint(list(predicates)[:5])
        log.info(f"Created {len(prior_rules)} prior rules. First a few:")
        pprint(list(prior_rules)[:5])
        
        constraint_predicates = set()
        for p in tqdm(predicates, desc="... Converting predicates to constraints"):
            if p not in [sp.true, sp.false]:
                p = Constraint(sp.sympify(p))
                constraint_predicates.add(p)
        
        assert len(constraint_predicates) == len(predicates), \
            f"Duplicate predicates found: {len(predicates) - len(constraint_predicates)}"
        
        self.prior = {Constraint(sp.sympify(r)) for r in prior_rules
                        if r not in [sp.true, sp.false]}
        
        return constraint_predicates