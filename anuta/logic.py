from collections import defaultdict
from itertools import combinations
from pathlib import Path
from multiprocess import Pool
from time import perf_counter, time
from typing import *
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import psutil
import sympy as sp
from dataclasses import dataclass
from enum import Enum, auto
from rich import print as pprint
import pickle
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import sys
sys.setrecursionlimit(100_000)


from anuta.grammar import (
    TYPE_DOMIAN, ConstantType, DomainType, VariableType, 
    group_variables_by_type_and_domain, get_variable_type, tautology, contradition)
from anuta.utils import log
from anuta.theory import Constraint, Theory
from anuta.constructor import Constructor
from anuta.cli import FLAGS

cfg = FLAGS.config

TYPE_PRIORITY = {
    VariableType.IP: 40,
    VariableType.PORT: 40,
    VariableType.SEQUENCING: 30,
    VariableType.FLAG: 40,
    VariableType.PROTO: 30,
    VariableType.POINTER: 2,
    VariableType.TIME: 2,
    VariableType.SIZE: 1,
    VariableType.TTL: 0,
    VariableType.WINDOW: 0,   # lowest priority
    VariableType.UNKNOWN: 0,
}

def is_candidate_trivial(candidate: Constraint) -> bool:
    #* Candidate is trivial if it is a tautology or contradition, or not an implication (assuming not predicates).
    return candidate in [tautology, contradition] or not isinstance(candidate.expr, sp.Implies)

# -------------------------------
# Coverage mask builder
# -------------------------------
def evaluate_predicates_with_masks(
    predicates: Iterable[Constraint],
    df: pd.DataFrame
) -> Tuple[Dict[Constraint, int], Set[Constraint]]:
    """
    Evaluate predicates across df and return:
      - mask_by_pred: {predicate -> int bitmask} (bit i = 1 iff row i satisfies predicate)
      - valid_predicates: {Constraint} (predicates true for ALL rows)

    Notes
    -----
    - This replaces build_evidence_set.
    - Predicates that are valid on all rows are recorded separately.
    - Other predicates have coverage masks stored in mask_by_pred.
    """
    series = df.to_dict(orient="series")
    n = df.shape[0]

    mask_by_pred: Dict[Constraint, int] = {}
    valid_predicates: Set[Constraint] = set()

    for predicate in tqdm(predicates, desc="... Evaluating predicates"):
        expr = predicate.expr
        arg1, arg2 = expr.args

        # LHS values
        assert not isinstance(arg1, sp.Number), f"Predicate {predicate} has a number as lhs"
        values1 = eval(str(arg1), {}, series)

        # RHS values
        if isinstance(arg2, sp.Integer):
            values2 = int(arg2)
        elif isinstance(arg2, sp.Float):
            values2 = float(arg2)
        else:
            values2 = eval(str(arg2), {}, series)

        sat: pd.Series = None
        if isinstance(expr, sp.Eq):
            sat = (values1 == values2)
        elif isinstance(expr, sp.Ne):
            sat = (values1 != values2)
        elif isinstance(expr, sp.StrictGreaterThan):
            sat = (values1 > values2)
        else:
            assert isinstance(expr, sp.LessThan), f"Unsupported predicate type: {expr}"
            sat = (values1 <= values2)

        # Convert to bitmask
        if sat.all():
            valid_predicates.add(predicate)
            continue

        # Convert boolean array to bitmask integer
        bits = np.packbits(sat.astype(np.uint8)[::-1])  # pack, reverse for correct order
        mask = int.from_bytes(bits.tobytes(), "big")

        mask_by_pred[predicate] = mask
    return mask_by_pred, valid_predicates


# -------------------------------
# Coverage-levelwise candidate
# -------------------------------
@dataclass(frozen=True)
class CoverageCandidate:
    preds: frozenset[Constraint]  # set of predicates
    mask: int         # coverage bitmask
    
# -------------------------------
# Frontier helpers
# -------------------------------
def order_frontier(frontier: List[CoverageCandidate]) -> List[CoverageCandidate]:
    """Order candidates by descending coverage size (larger mask population first)."""
    return sorted(frontier, key=lambda c: bin(c.mask).count("1"), reverse=True)


def dedup_frontier(frontier: List[CoverageCandidate]) -> List[CoverageCandidate]:
    """Remove duplicates (by predicate set)."""
    seen = set()
    deduped = []
    for c in frontier:
        if c.preds not in seen:
            deduped.append(c)
            seen.add(c.preds)
    return deduped


def is_super_of_learned(candidate: CoverageCandidate, learned: Set[frozenset]) -> bool:
    """
    True if candidate is a strict superset of any already learned rule.
    Such candidates cannot be minimal and can be pruned.
    """
    for lr in learned:
        if lr.issubset(candidate.preds):
            return True
    return False

def evaluate_predicates(predicates: Iterable[Constraint], df: pd.DataFrame) -> Tuple[Set[Constraint], Set[Constraint]]:
    series = df.to_dict(orient='series')
    valid_predicates = set()
    invalid_predicates = set()
    for predicate in tqdm(predicates, desc="... Testing predicates"):
        if predicate in valid_predicates: 
            continue
        
        expr = predicate.expr
        arg1, arg2 = expr.args
        
        assert not isinstance(arg1, sp.Number), f"Predicate {predicate} has a number as the first argument."
        values1 = eval(str(arg1), {}, series)
        
        if isinstance(arg2, sp.Integer):
            values2 = int(arg2)
        elif isinstance(arg2, sp.Float):
            values2 = float(arg2)
        else:
            values2 = eval(str(arg2), {}, series)
        
        if all(values1 == values2):
            valid_predicates.add(predicate)
        elif all(values1 != values2):
            valid_predicates.add(Constraint(sp.Not(predicate.expr)))
        else:
            invalid_predicates.add(predicate)
        
    return valid_predicates, invalid_predicates
        

def test_candidates(worker_id: int, candidates: Iterable[sp.Expr], df: pd.DataFrame) -> Set[sp.Expr]:
    # True if expr holds for every row (no free symbols after substitution)
    valid_candidates = set()
    for candidate in tqdm(candidates, desc=f"... Testing candidates (worker {worker_id})"):
        is_valid = True
        for _, row in (df.iterrows()):
            assignment = row.to_dict()
            try:
                sat = candidate.subs(assignment)
                if not sat:
                    is_valid = False
                    break
            except Exception as e:
                log.error(f"Error evaluating candidate {candidate} with {assignment=}: {e}")
                exit(1)
        if is_valid:
            valid_candidates.add(candidate)
    return valid_candidates

def generalize_candidates(invalid_candidates: List[sp.Expr]) -> Set[sp.Expr]:
    def process_pair(worker_id: int, pairs: List[Tuple[sp.Expr, sp.Expr]]) -> Set[sp.Expr]:
        new_candidates: Set[sp.Expr] = set()
        for c1, c2 in tqdm(pairs, desc=f"... Generalizing candidates (worker {worker_id})"):
            ant1, con1 = c1.args[0], c1.args[1]
            ant2, con2 = c2.args[0], c2.args[1]

            new_candidate = None
            #& A => B and A => C to A => (B or C)
            #! Best to check `Constraint(ant1) == Constraint(ant2)` but too expensive.
            if ant1 == ant2:
                new_candidate = sp.Implies(ant1, sp.Or(con1, con2))
            #& A => C and B => C to (A and B) => C
            elif con1 == con2:
                new_candidate = sp.Implies(sp.And(ant1, ant2), con1)

            if new_candidate and new_candidate not in [sp.true, sp.false]: #not is_candidate_trivial(new_candidate):
                new_candidates.add(new_candidate)
        return new_candidates

    # Generate all pairs of invalid candidates
    pairs = list(combinations(invalid_candidates, 2))
    if not pairs:
        return set()

    # Split work into chunks
    nworkers = min(psutil.cpu_count(logical=True), len(pairs))
    chunks = np.array_split(pairs, nworkers)

    # Run in parallel
    results = Parallel(n_jobs=nworkers, backend="loky")(
        delayed(process_pair)(i, chunk) for i, chunk in enumerate(chunks) if len(chunk) > 0
    )

    # Merge results
    return set().union(*results)


def build_pairwise_implications(invalid_predicates: Set[Constraint]) -> Set[sp.Expr]:
    invalid_predicates = list(invalid_predicates)  # make sure it's indexable
    nworkers = psutil.cpu_count()

    def worker(worker_id: int, subset: List[Constraint]) -> Set[Constraint]:
        local_new = set()
        for p1 in tqdm(subset, desc=f"... Building pairwise implications (worker {worker_id})"):
            for p2 in invalid_predicates:
                if p1 == p2 and p1 != Constraint(sp.Not(p2.expr)):
                    continue
                #! Conversion to Constraint is too expensive, so we skip it here.
                # candidate = Constraint(sp.Implies(p1.expr, p2.expr))
                # if not is_candidate_trivial(candidate):
                candidate = sp.Implies(p1.expr, p2.expr)
                if candidate not in [sp.true, sp.false] and isinstance(candidate, sp.Implies):
                    local_new.add(candidate)
        return set(local_new)

    # Split invalid_predicates into chunks (for outer loop)
    chunks = np.array_split(invalid_predicates, nworkers)

    results = Parallel(n_jobs=nworkers, backend="loky")(
        delayed(worker)(i, chunk) for i, chunk in enumerate(chunks)
    )

    new_candidates = set().union(*results)
    return new_candidates

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
        #* Variables -> their types.
        self.vtypes: Dict[str, VariableType] = {}
        for var in self.variables:
            self.vtypes[var] = get_variable_type(var)
        self.domains = constructor.anuta.domains
        self.constants = constructor.anuta.constants
        self.multiconstants = constructor.anuta.multiconstants
        
        self.categoricals = [] # constructor.categoricals
        self.prior: Set[Constraint] = set()
    
    def learn_denial(
        self,
        max_predicates: Optional[int] = cfg.MAX_PREDICATES,
        max_learned_rules: Optional[int] = cfg.MAX_RULES,
    ):
        """
        1) Create predicate space (already intra-tuple).
        2) Build intra-tuple evidence sets: for each tuple t, SAT(t) = { P | t |= P }.
        3) Enumerate all minimal hitting sets with bounded DFS/BnB.
        4) Return rules: Or(*[p in cover]) (instead of Denial constraints And(*[¬p in cover])).
        """
        log.info(f"Learning denial constraints: Max {max_predicates} predicates, Max {max_learned_rules} rules.")
        predicates: Set[Constraint] = self.generate_predicates_and_prior()
        
        #* First check if the evidence sets file exists, if so, load it.
        evidence_sets_file = f'data/evidence_sets_{self.dataset}_{self.num_examples}.pkl'
        if Path(evidence_sets_file).exists():
            log.info(f"Loading evidence sets from {evidence_sets_file}...")
            with open(evidence_sets_file, 'rb') as f:
                evidence_sets: List[frozenset[Constraint]] = pickle.load(f)
            log.info(f"Loaded {len(evidence_sets)} evidence sets.")
        else:
            start = perf_counter()
            evidence_sets: List[frozenset[Constraint]] = self.build_evidence_set(predicates)
            end = perf_counter()
            log.info(f"\nBuilt {len(evidence_sets)} evidence sets in {end - start:.2f} seconds.")
        
        assert len(evidence_sets) == self.examples.shape[0], \
            f"Evidence sets size {len(evidence_sets)} != the # of examples {self.examples.shape[0]}."    
        
        if not evidence_sets:
            log.warning("No non-empty evidence sets were produced; returning only prior.")

        start = perf_counter()
        # covers = self.enumerate_minimal_hitting_sets(
        #     evidence_sets=evidence_sets,
        #     max_size=max_predicates,
        #     max_solutions=max_learned_rules,
        # )
        covers = self.enumerate_minimal_hitting_sets_suppression(
            # predicates,
            evidence_sets=evidence_sets,
            max_size=max_predicates,
            max_solutions=max_learned_rules,
        )
        end = perf_counter()
        print()
        log.info(f"Enumerated {len(covers)} minimal hitting sets in {(end - start)/60:.2f} minutes.")
        
        learned_rules: Set[sp.Expr] = set()
        num_trivial = 0
        for cover in tqdm(covers, desc="... Collecting learned rules"):
            rule = sp.Or(*[c.expr for c in cover])
            #* Remove trivial rules (True/False).
            if rule not in [sp.true, sp.false]:
                # learned_rules.add(Constraint(rule))
                learned_rules.add(rule)
            else:
                num_trivial += 1
                
        num_learned = len(learned_rules)
        log.info(f"Learned {num_learned} rules.")
        log.info(f"Removed {num_trivial} trivial rules (True/False).")
        
        # learned_rules |= self.prior
        for constraint in self.prior:
            learned_rules.add(constraint.expr)
        log.info(f"Total {len(learned_rules)} rules (including prior).")
        Theory.save_constraints(learned_rules, f'denial_{self.dataset}_{self.num_examples}_p{max_predicates}.pl')
        return
        
        
    def enumerate_minimal_hitting_sets(
        self,
        evidence_sets: List[frozenset[Constraint]],
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
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
        for i, E in enumerate(tqdm(evidence_sets, desc="... Indexing predicates in evidence sets")):
            for p in E:
                idx_by_pred[p].add(i)

        # #* Filter out predicates with WINDOW or SIZE variables for testing.
        # for predicate in list(idx_by_pred.keys()):
        #     variables = predicate.expr.free_symbols
        #     if any(self.vtypes[str(v)] in [VariableType.WINDOW, VariableType.SIZE] 
        #             for v in variables):
        #         del idx_by_pred[predicate]
                
        # Precompute each evidence set as a list to iterate deterministically
        E_list: List[List[Constraint]] = [list(E) for E in evidence_sets]

        
        # pred_order = [p for p, _ in idx_by_pred.items()]
        
        # #* Predicate ordering: higher type priority first, then higher coverage
        # log.info("Ordering predicates by type...")
        # def _get_pred_score(p: Constraint, covered_examples):
        #     cov = len(covered_examples)
        #     type_score = max(
        #         TYPE_PRIORITY.get(self.vtypes[str(var)], 0)
        #         for var in p.expr.free_symbols
        #     )
        #     return (type_score, cov)
        
        # pred_by_score = sorted(
        #     idx_by_pred.items(),
        #     key=lambda kv: (-_get_pred_score(kv[0], kv[1])[0],   # higher type priority first
        #                     # -_get_pred_score(kv[0], kv[1])[1]    # then higher coverage
        #                 )
        # )
        # pred_order = [p for p, _ in pred_by_score]
        
        #* Predicate ordering: higher coverage first (helps pruning) [Chu et al., VLDB '13]
        log.info("Ordering predicates by coverage...")
        pred_by_cov = sorted(idx_by_pred.items(), key=lambda kv: -len(kv[1]))
        pred_order = [p for p, _ in pred_by_cov]

        solutions: List[frozenset[Constraint]] = []

        # Quick helper: prune if chosen is a superset of an existing minimal solution.
        def _dominated_by_existing(chosen: Set[Constraint]) -> bool:
            fc = frozenset(chosen)
            for sol in solutions:
                # if chosen is a superset of a found minimal solution, it can’t be minimal
                if sol.issubset(fc):
                    return True
            return False

        # Check coverage
        FULL = set(range(len(E_list)))

        #* Optimistic bound on remaining picks:
        # If we still need to cover R uncovered evidences, and the best single predicate
        # can cover at most M of them, then we need at least ceil(|R|/M) more predicates.
        # If len(chosen) + that lower bound > max_size, prune.
        # (Simple but effective for a BnB cut.)
        def _optimistic_lb(uncovered: Set[int]) -> int:
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

        #* Choose an uncovered evidence with the fewest predicates (fail-first)
        def _pick_uncovered_with_smallest_branch(uncovered: Set[int], chosen: Set[Constraint]) -> int:
            best_i, best_deg = None, float('inf')
            for i in uncovered:
                # candidates are predicates in E[i]; we can skip those already in chosen,
                # but keeping them is fine — small effect on degree.
                deg = len(E_list[i])
                if deg < best_deg:
                    best_deg = deg
                    best_i = i
            return best_i  # index in E_list
        
        def _predicate_type_priority(p: Constraint) -> int:
            #TODO: Only need the first var since all vars in a predicate have the same type.
            vars_in_p = [str(v) for v in p.expr.free_symbols]
            if not vars_in_p:
                return 1  # fallback
            return max(
                TYPE_PRIORITY.get(self.vtypes[var], 0)
                for var in vars_in_p
            )

        # DFS
        def dfs(chosen: Set[Constraint], covered: Set[int]):
            # Early domination prune
            if _dominated_by_existing(chosen):
                return

            # Covered all evidences? record a minimal solution
            if covered == FULL:
                fc = frozenset(chosen)

                # --- compute new solution set and net change ---
                old_len = len(solutions)
                keep: List[frozenset[Constraint]] = []
                for s in solutions:
                    if s.issubset(fc):
                        # existing is smaller/equal → new is not minimal
                        if s != fc:
                            return
                    elif fc.issubset(s):
                        # new is smaller → drop the old superset
                        continue
                    keep.append(s)

                #* Replace in place to preserve references
                solutions.clear()
                solutions.extend(keep)
                solutions.append(fc)

                new_len = len(solutions)
                delta = new_len - old_len
                #* delta can be negative when we prune supersets.
                progress.update(delta)

                # Optional cap
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return
                return

            # BnB size bound
            if max_size is not None and len(chosen) >= max_size:
                return

            # Remaining uncovered
            uncovered = FULL - covered

            #* BnB optimistic bound
            if max_size is not None:
                lb = _optimistic_lb(uncovered)
                if len(chosen) + lb > max_size:
                    return

            #* Branch on the uncovered evidence covered by fewest predicates (fail-first)
            pivot = _pick_uncovered_with_smallest_branch(uncovered, chosen)
            
            #* Order candidate predicates by descending marginal gain
            #*  where gain is the number of uncovered evidences they cover.
            predicates = sorted(
                E_list[pivot],
                key=lambda p: -len((idx_by_pred[p]) & uncovered)
            )
            
            # #* Order candidate predicates by descending score
            # predicates = sorted(
            #     E_list[pivot],
            #     key=lambda p: (
            #         -_predicate_type_priority(p),            # prioritize by type
            #         # -len((idx_by_pred[p]) & uncovered)      # then by coverage gain
            #     )
            # )


            for p in predicates:
                if p in chosen:
                    # Already picked (rare in this branch), but continue recursion to avoid duplicates
                    new_chosen = chosen
                    new_covered = covered
                else:
                    new_chosen = set(chosen)
                    new_chosen.add(p)
                    new_covered = covered | idx_by_pred[p]

                # Another quick domination check before descending
                if _dominated_by_existing(new_chosen):
                    continue

                dfs(new_chosen, new_covered)

                #* Early stop if we reached the cap
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return

        progress = tqdm(desc=f"... Enumerating hitting sets ({max_size=})", 
                        unit=" sets", total=max_solutions)
        dfs(set(), set())

        # Convert back to list[set]
        return [set(s) for s in solutions]


    def enumerate_minimal_hitting_sets_suppression(
        self,
        # predicates,
        evidence_sets: List[frozenset[Constraint]],
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
    ) -> List[Set[Constraint]]:
        """
        Enumerate all minimal hitting sets H such that ∀E in evidence_sets: H ∩ E ≠ ∅.
        Iterative stack-based DFS (no recursion).
        Runs multiple searches with different suppression subsets of variable types,
        combining results with global pruning for maximum diversity.
        """
        if not evidence_sets:
            return []

        # All suppression type combinations (power set of surpressed_vtypes)
        vtypes = [vtype for vtype in VariableType if vtype not in [VariableType.UNKNOWN]]
        suppression_combos = []
        if cfg.ENABLE_TYPE_SUPPRESSION:
            for r in range(len(vtypes)):
                for combo in combinations(vtypes, r):
                    suppression_combos.append(set(combo))
        else:
            suppression_combos.append(set())
            
        # allowed = [VariableType.IP, VariableType.PORT, VariableType.SEQUENCING, VariableType.FLAG]
        # suppression_combos = [{vtype for vtype in VariableType if vtype not in allowed}]
        log.info(f"Launching searches with {len(suppression_combos)} suppression sets.")
        
        #* Sort the combos from largest to smallest (suppressing more types first)
        suppression_combos.sort(key=lambda s: -len(s))
        
        # Build idx_by_pred (same for all runs)
        idx_by_pred: Dict[Constraint, Set[int]] = defaultdict(set)
        for i, E in enumerate(tqdm(evidence_sets, desc="... Indexing predicates in evidence sets")):
            for p in E:
                idx_by_pred[p].add(i)
        
        # indexed_preds = list(idx_by_pred.keys())
        # missing_preds = set()
        # for predicate in predicates:        
        #     if predicate not in indexed_preds:
        #         missing_preds.add(predicate)
        # print(f"Predicates not in any evidence set: {len(missing_preds)} / {len(predicates)}")
        # pprint(missing_preds)
        # exit(0)
        
        E_list: List[List[Constraint]] = [list(E) for E in evidence_sets]
        FULL = set(range(len(E_list)))

        # Global solutions across all runs
        global_solutions: List[frozenset[Constraint]] = []

        def _dominated_by_existing(chosen: Set[Constraint]) -> bool:
            fc = frozenset(chosen)
            for sol in global_solutions:
                if sol.issubset(fc):  # superset of existing
                    return True
            return False

        def _optimistic_lb(uncovered: Set[int]) -> int:
            if not uncovered:
                return 0
            best_gain = 0
            for p in idx_by_pred:
                gain = len(uncovered & idx_by_pred[p])
                if gain > best_gain:
                    best_gain = gain
                    if best_gain == len(uncovered):
                        break
            return 1 if best_gain == 0 else ((len(uncovered) + best_gain - 1) // best_gain)

        def _pick_uncovered_with_smallest_branch(uncovered: Set[int]) -> int:
            best_i, best_deg = None, float('inf')
            for i in uncovered:
                deg = len(E_list[i])
                if deg < best_deg:
                    best_deg, best_i = deg, i
            return best_i
        
        def _pick_uncovered_with_smallest_unchosen(uncovered: Set[int], chosen: Set[Constraint]) -> int:
            best_i, best_deg = None, float('inf')
            for i in uncovered:
                deg = len(evidence_sets[i] - frozenset(chosen))
                if deg < best_deg:
                    best_deg, best_i = deg, i
            return best_i
        

        # Run one search with a suppression set
        def run_search(suppressed: Set[VariableType]):
            # log.info(f"Searching with surpressed vtypes:\n{suppressed}...")
            allowed_predicates = set(
                p for p in idx_by_pred
                if all(self.vtypes[str(v)] not in suppressed for v in p.expr.free_symbols)
            )

            solutions_this_run: List[frozenset[Constraint]] = []

            stack = [(set(), set(), FULL, None, 0)]

            start_time = perf_counter()
            last_solution_time = perf_counter()
            while stack:
                if perf_counter() - start_time > cfg.TIMEOUT_SEC:
                    log.warning(f"Learning timed out after {cfg.TIMEOUT_SEC//60}min.")
                    break
                
                chosen, covered, uncovered, candidates, next_idx = stack.pop()
                #& Fully covered
                if not uncovered:
                    fc = frozenset(chosen)
                    if not _dominated_by_existing(fc):
                        # prune global supersets and add
                        keep = []
                        for s in global_solutions:
                            if s.issubset(fc):
                                if s != fc:
                                    break
                            elif fc.issubset(s):
                                continue  # drop superset
                            keep.append(s)
                        else:
                            global_solutions.clear()
                            global_solutions.extend(keep)
                            global_solutions.append(fc)

                            solutions_this_run.append(fc)
                            last_solution_time = perf_counter()
                            progress.update(1)
                            if max_solutions and len(solutions_this_run) >= max_solutions:
                                # progress.close()
                                return
                    continue
                
                #& Timeout: if no new solution in the last Xs, stop
                if perf_counter() - last_solution_time > cfg.STALL_TIMEOUT_SEC:
                    log.warning(
                        f"Search with {len(vtypes)-len(suppressed)} vtypes timed out after {cfg.STALL_TIMEOUT_SEC//60}min of no new solutions.")
                    break
                
                #& Bound by size
                if max_size is not None and len(chosen) >= max_size:
                    continue

                #& BnB optimistic bound
                if max_size is not None:
                    lb = _optimistic_lb(uncovered)
                    if len(chosen) + lb > max_size:
                        continue

                #& Load candidates if first time
                if candidates is None:
                    # pivot = _pick_uncovered_with_smallest_branch(uncovered)
                    pivot = _pick_uncovered_with_smallest_unchosen(uncovered, chosen)
                    candidates = [p for p in E_list[pivot] if p in allowed_predicates]
                    # sort by coverage gain
                    candidates = sorted(candidates, key=lambda p: -len((idx_by_pred[p]) & uncovered))
                    next_idx = 0

                if next_idx >= len(candidates):
                    continue

                #* Push frame back
                stack.append((chosen, covered, uncovered, candidates, next_idx + 1))

                p = candidates[next_idx]
                if p in chosen:
                    new_chosen, new_covered = chosen, covered
                else:
                    new_chosen = set(chosen)
                    new_chosen.add(p)
                    new_covered = covered | idx_by_pred[p]

                if _dominated_by_existing(new_chosen):
                    continue

                new_uncovered = uncovered - idx_by_pred[p]
                stack.append((new_chosen, new_covered, new_uncovered, None, 0))

            return

        progress = tqdm(desc=f"... Enumerating hitting sets (max_size={max_size})",
                        unit=" sets", total=max_solutions*len(suppression_combos))
        # Run searches for all suppression combos
        for suppressed in suppression_combos:
            run_search(suppressed)
        progress.close()

        return [set(s) for s in global_solutions]

    def enumerate_minimal_hitting_sets_stack(
        self,
        evidence_sets: List[frozenset[Constraint]],
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
    ) -> List[Set[Constraint]]:
        """
        Enumerate all minimal hitting sets H such that ∀E in evidence_sets: H ∩ E ≠ ∅.
        Iterative stack-based DFS (no recursion).
        """
        if not evidence_sets:
            return []
        
        surpressed_vtypes = [VariableType.WINDOW, VariableType.SIZE]
        allowed_predicates: Set[Constraint] = set()
        # Index: for each predicate, which evidence indices contain it
        idx_by_pred: Dict[Constraint, Set[int]] = defaultdict(set)
        for i, E in enumerate(tqdm(evidence_sets, desc="... Indexing predicates in evidence sets")):
            for p in E:
                idx_by_pred[p].add(i)
                allowed_predicates.add(p)
                
        allowed_predicates = set(filter(
            lambda p: self.vtypes[str(p.expr.free_symbols.pop())] not in surpressed_vtypes,
            allowed_predicates
        ))

        # #* Filter out predicates with WINDOW or SIZE variables for testing.
        # for predicate in list(idx_by_pred.keys()):
        #     variables = predicate.expr.free_symbols
        #     if any(self.vtypes[str(v)] in [VariableType.WINDOW, VariableType.SIZE] 
        #             for v in variables):
        #         print(f"Removing predicate {predicate} with variables {variables} for testing.")
        #         del idx_by_pred[predicate]

        # Evidence sets as lists
        E_list: List[List[Constraint]] = [list(E) for E in evidence_sets]

        #* Predicate ordering: higher coverage first (Chu et al. 2013)
        log.info("Ordering predicates by coverage...")
        pred_by_cov = sorted(idx_by_pred.items(), key=lambda kv: -len(kv[1]))
        pred_order = [p for p, _ in pred_by_cov]
        
        # #* Predicate ordering: higher type priority first, then higher coverage
        # log.info("Ordering predicates by type...")
        # def _get_pred_score(p: Constraint, covered_examples):
        #     cov = len(covered_examples)
        #     var = p.expr.free_symbols.pop()
        #     type_score = TYPE_PRIORITY.get(self.vtypes[str(var)], 0)
        #     return (type_score, cov)
        
        # pred_by_score = sorted(
        #     idx_by_pred.items(),
        #     key=lambda kv: (-_get_pred_score(kv[0], kv[1])[0],   # higher type priority first
        #                     # -_get_pred_score(kv[0], kv[1])[1]    # then higher coverage
        #                 )
        # )
        # pred_order = [p for p, _ in pred_by_score]

        solutions: List[frozenset[Constraint]] = []
        FULL = set(range(len(E_list)))

        # --- Helpers ---
        #* Order candidate predicates by descending score
        def _predicate_type_priority(p: Constraint) -> int:
            #TODO: Only need the first var since all vars in a predicate have the same type.
            var = p.expr.free_symbols.pop()
            return TYPE_PRIORITY.get(self.vtypes[str(var)], 0)
            
        def _dominated_by_existing(chosen: Set[Constraint]) -> bool:
            fc = frozenset(chosen)
            for sol in solutions:
                if sol.issubset(fc):  # superset of existing
                    return True
            return False

        def _optimistic_lb(uncovered: Set[int]) -> int:
            if not uncovered:
                return 0
            best_gain = 0
            for p in pred_order:
                gain = len(uncovered & idx_by_pred[p])
                if gain > best_gain:
                    best_gain = gain
                    if best_gain == len(uncovered):
                        break
            return 1 if best_gain == 0 else ((len(uncovered) + best_gain - 1) // best_gain)

        def _pick_uncovered_with_smallest_branch(uncovered: Set[int]) -> int:
            best_i, best_deg = None, float('inf')
            for i in uncovered:
                deg = len(E_list[i])
                if deg < best_deg:
                    best_deg, best_i = deg, i
            return best_i

        # --- Iterative DFS with explicit stack ---
        # Each frame: (chosen, covered, uncovered, candidates, next_idx)
        progress = tqdm(desc=f"... Enumerating hitting sets ({max_size=})", 
                        unit=" sets", total=max_solutions)
        stack = [(set(), set(), FULL, None, 0)]

        while stack:
            chosen, covered, uncovered, candidates, next_idx = stack.pop()
            # if len(stack) > 1000:
            #     # print(f"Stack depth: {len(stack)} frames, skipping further checks.")
            #     continue  # too deep, skip this frame

            # Check if fully covered
            if not uncovered:
                fc = frozenset(chosen)
                old_len = len(solutions)
                keep: List[frozenset[Constraint]] = []
                for s in solutions:
                    if s.issubset(fc):
                        if s != fc:
                            break  # new one not minimal
                    elif fc.issubset(s):
                        continue  # drop superset
                    keep.append(s)
                else:
                    # commit only if not broken
                    solutions.clear()
                    solutions.extend(keep)
                    solutions.append(fc)
                    new_len = len(solutions)
                    progress.update(new_len - old_len)
                    if max_solutions and len(solutions) >= max_solutions:
                        progress.close()
                        return [set(s) for s in solutions]
                continue

            # Bound by size
            if max_size is not None and len(chosen) >= max_size:
                continue

            # Bound by optimistic LB
            if max_size is not None:
                lb = _optimistic_lb(uncovered)
                if len(chosen) + lb > max_size:
                    continue

            # If no candidates loaded, compute them
            if candidates is None:
                pivot = _pick_uncovered_with_smallest_branch(uncovered)
                
                predicates = sorted(
                    E_list[pivot],
                    key=lambda p: -len((idx_by_pred[p]) & uncovered)
                )
                # predicates = sorted(
                #     E_list[pivot],
                #     key=lambda p: (
                #         -_predicate_type_priority(p),            # prioritize by type
                #         # -len((idx_by_pred[p]) & uncovered)      # then by coverage gain
                #     )
                # )
                
                candidates, next_idx = predicates, 0
                #* Filter candidates to only those that are allowed
                candidates = [p for p in candidates if p in allowed_predicates]

            # If we exhausted candidates, skip
            if next_idx >= len(candidates):
                continue

            # Push frame back with incremented index (simulate recursion return)
            stack.append((chosen, covered, uncovered, candidates, next_idx + 1))

            # Take candidate p
            p = candidates[next_idx]
            if p in chosen:
                new_chosen, new_covered = chosen, covered
            else:
                new_chosen = set(chosen)
                new_chosen.add(p)
                new_covered = covered | idx_by_pred[p]

            if _dominated_by_existing(new_chosen):
                continue

            new_uncovered = uncovered - idx_by_pred[p]

            # Push recursive call frame
            stack.append((new_chosen, new_covered, new_uncovered, None, 0))

        progress.close()
        return [set(s) for s in solutions]


    def learn_levelwise(
        self,
        max_size: int = 20,
        max_rules: Optional[int] = None,
    ):
        """
        Coverage-levelwise algorithm:
        1. Compute mask per predicate (bitset of satisfied examples).
        2. Singletons that cover ALL are learned.
        3. Iteratively join (Apriori rule: k-1 overlap) to build larger candidates.
        4. Stop when max_size is reached or no new candidates.
        """
        predicates: Set[Constraint] = self.generate_predicates_and_prior()
        mask_by_pred, valid_preds = evaluate_predicates_with_masks(predicates, self.examples)
        log.info(f"Evaluated {len(predicates)} predicates: {len(valid_preds)} valid, {len(mask_by_pred)} invalid.")
        n = self.examples.shape[0]
        FULL_MASK = (1 << n) - 1

        learned_rules: Set[frozenset] = set()
        learned_constraints: Set[sp.Expr] = set()

        # Add always-valid predicates
        for vp in valid_preds:
            learned_rules.add(frozenset({vp}))
            learned_constraints.add(vp.expr)

        # Initialize level-1 candidates (those not valid everywhere)
        current_level: List[CoverageCandidate] = [
            CoverageCandidate(frozenset({p}), mask)
            for p, mask in mask_by_pred.items()
        ]

        k = 1
        start = perf_counter()
        timeout_reached = False
        while current_level and k < max_size:
            log.info(f"Level {k}: {len(current_level)} candidates.")
            # Frontier maintenance
            beam_width = None
            current_level = order_frontier(dedup_frontier(current_level))[:beam_width]
            if len(current_level) == beam_width:
                log.warning(f"Truncated to {beam_width} candidates for level {k}.")

            next_level: List[CoverageCandidate] = []

            prev_learned = len(learned_rules)
            # Apriori joins
            for i, a in enumerate(tqdm(current_level, desc=f"Level {k} joins")):
                if timeout_reached:
                    break
                for j in range(i + 1, len(current_level)):
                    if perf_counter() - start > cfg.TIMEOUT_SEC:
                        log.warning(f"Timeout reached after {cfg.TIMEOUT_SEC//60} minutes.")
                        timeout_reached = True
                        break
                    
                    b = current_level[j]

                    if len(a.preds & b.preds) != (k - 1):
                        continue

                    new_preds = a.preds | b.preds
                    if len(new_preds) != k + 1:
                        continue

                    if is_super_of_learned(CoverageCandidate(new_preds, 0), learned_rules):
                        continue

                    new_mask = a.mask | b.mask

                    if new_mask == FULL_MASK:
                        # Found valid cover
                        if new_preds not in learned_rules:
                            learned_rules.add(new_preds)
                            rule_expr = sp.Or(*[p.expr for p in new_preds])
                            learned_constraints.add(rule_expr)
                            if max_rules and len(learned_constraints) >= max_rules:
                                log.info(f"Reached {max_rules=}, stopping.")
                                Theory.save_constraints(
                                    learned_constraints,
                                    f"coverage_{self.dataset}_{self.num_examples}.pl"
                                )
                                return
                    else:
                        next_level.append(CoverageCandidate(new_preds, new_mask))
            if timeout_reached:
                log.warning(f"Timeout reached, stopping level {k} learning.")
                break
            current_level = next_level
            log.info(f"Level {k}: Learned {len(learned_rules) - prev_learned} new rules, total {len(learned_rules)}.")
            k += 1

        log.info(f"Learned {len(learned_constraints)} rules in total.")
        # learned_constraints |= self.prior
        for constraint in self.prior:
            learned_rules.add(constraint.expr)
        Theory.save_constraints(
            learned_constraints,
            f"levelwise_{self.dataset}_{self.num_examples}.pl"
        )
        return learned_constraints
    
    def learn_levelwise_netnomos(
        self,
        max_epochs: int = FLAGS.config.LEVELWISE_EPOCHS,
        max_rules: Optional[int] = FLAGS.config.MAX_RULES,
    ):
        learned_rules: Set[Constraint] = set()
        new_candidates: Set[sp.Expr] = set()
        
        predicates: Set[Constraint] = self.generate_predicates_and_prior()
        valid_predicates, invalid_predicates = evaluate_predicates(predicates, self.examples)
        log.info(f"Evaluated {len(predicates)} predicates: {len(valid_predicates)} valid, {len(invalid_predicates)} invalid.")
        learned_rules |= valid_predicates
        
        #& Build pairwise implications
        new_candidates = build_pairwise_implications(invalid_predicates)
        log.info(f"Created {len(new_candidates)} pairwise implication candidates.")
        
        epoch = 1
        learned_candidates: Set[sp.Expr] = set()
        while epoch <= max_epochs and new_candidates:
            print(f"\tEpoch {epoch}: {len(new_candidates)} candidates...")
            
            current_candidates = new_candidates
            new_candidates = set()
            prev_learned = len(learned_candidates)

            nworkers = psutil.cpu_count()
            chunks = np.array_split(list(current_candidates), nworkers)
            args = [(i, chunk, self.examples) for i, chunk in enumerate(chunks)]
            pool = Pool(nworkers)
            results = pool.starmap(test_candidates, args)
            # results = Parallel(n_jobs=nworkers, backend="loky")(
            #     delayed(test_candidates)(i, chunk, self.examples)
            #     for i, chunk in enumerate(chunks)
            # )
            valid_candidates = set().union(*results)
            invalid_candidates = list(current_candidates - valid_candidates)
            learned_candidates |= valid_candidates
            total_learned = len(learned_candidates)
            print(f"\tEpoch {epoch}: learned {total_learned-prev_learned} new rules, total {total_learned}.")
            print(f"\tEpoch {epoch}: {len(invalid_candidates)} invalid candidates.")
            
            new_candidates = generalize_candidates(invalid_candidates)
            print(f"\tEpoch {epoch}: {len(new_candidates)} new candidates generated.")
            
            if max_rules and len(total_learned) >= max_rules:
                log.info(f"Reached max rules limit of {max_rules}. Stopping.")
                break
            epoch += 1

        log.info(f"Learned {len(learned_rules)} rules in {epoch} epochs.")
        prev_learned = len(learned_rules)
        for candidate in tqdm(learned_candidates, desc="... Collecting learned candidates"):
            learned_rules.add(Constraint(candidate))
        num_redundant = (prev_learned + len(learned_candidates)) - len(learned_rules)
        log.info(f"Removed {num_redundant} redundant rules.")
        
        learned_rules |= self.prior
        log.info(f"Total {len(learned_rules)} rules (including prior).")
        Theory.save_constraints(learned_rules, f'levelwise_{self.dataset}_{self.num_examples}_e{epoch}.pl')
        return
    
    def build_evidence_set(self, predicates: Set[Constraint], save: bool = True) -> List[Set[Constraint]]:
        # Pre-pack predicates into a list so all workers use same reference
        predicates_list = list(predicates)

        def _process_examples(worker_id, df_chunk: pd.DataFrame) -> List[Set[Constraint]]:
            chunk_results = []
            for _, row in tqdm(df_chunk.iterrows(), 
                               total=df_chunk.shape[0], 
                               desc=f"... Building evidence sets (worker {worker_id})"):
                row_dict = row.to_dict()
                satisfied = set()
                for p in predicates_list:
                    try:
                        if p.expr.subs(row_dict):
                            satisfied.add(p)
                    except Exception:
                        # Any substitution/eval failure = not satisfied
                        continue
                chunk_results.append(satisfied)
            return chunk_results

        n_jobs = psutil.cpu_count()

        # Split into n_jobs chunks (avoid overhead of too many tasks)
        chunks = np.array_split(self.examples, n_jobs)
        log.info(f"Processing {len(chunks)} batches with {n_jobs} workers.")

        # Parallel execution — each worker gets a big chunk
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_examples)(i, chunk)
            for i, chunk in enumerate(chunks)
        )
        
        results = [s for chunk in results for s in chunk]  # Flatten list of lists
        if save:
            with open(f"evidence_sets_{self.dataset}_{self.num_examples}.pkl", 'wb') as f:
                pickle.dump(results, f)
                log.info(f"Saved evidence sets to 'evidence_sets_{self.dataset}_{self.num_examples}.pkl'.")

        return results


    def generate_predicates_and_prior(self) -> Set[Constraint]:
        vtype2vars, domaintype2vars = group_variables_by_type_and_domain(self.variables)
        variable_types = self.vtypes
        # pprint(vtype2vars)
        
        prior_rules: Set[str] = set()
        #* Collecting categorical variables (with >1 unique value).
        for varname in domaintype2vars[DomainType.CATEGORICAL]:
            if self.examples[varname].nunique() > 1:
                #* Only consider categorical variables with more than one unique value.
                self.categoricals.append(varname)
            else:
                #* Neglect variables with only one unique value but add them to prior rules.
                self.variables.remove(varname)
                varval = self.examples[varname].iloc[0].item()
                prior_rules.add(f"Eq({varname}, {varval})")
                
        '''Augment the variables with constants.'''
        avars = set()
        # for varname, consts in self.constants.items():
        #     if varname in self.categoricals:
        #         #* Only augment numerical variables with constants.
        #         continue
        #     vtype = variable_types[varname]
        #     #& X*c 
        #     if consts.kind == ConstantType.SCALAR:
        #         for const in consts.values:
        #             if const == 1: continue
        #             avar = f"{const}$*${varname}"
        #             avars.add(avar)
        #             variable_types[avar] = vtype
        #     #& X+c
        #     if consts.kind == ConstantType.ADDITION:
        #         for const in consts.values:
        #             avar = f"{const}$+${varname}"
        #             avars.add(avar)
        #             variable_types[avar] = vtype
            
        for varname, constants in self.multiconstants:
            if varname in self.categoricals:
                #* Only augment numerical variables with constants.
                continue
            vtype = variable_types[varname]
            #& X*c 
            if constants.kind == ConstantType.SCALAR:
                for value in constants.values:
                    if value == 1: continue
                    avar = f"{value}$*${varname}"
                    avars.add(avar)
                    variable_types[avar] = vtype
            #& X+c 
            if constants.kind == ConstantType.ADDITION:
                for value in constants.values:
                    avar = f"{value}$+${varname}"
                    avars.add(avar)
                    variable_types[avar] = vtype

        
        num_avars = len(avars)
        log.info(f"Created {len(avars)} augmented variables.")
        
        '''Compound variables'''
        #& X+Y and (some) X*Y of the same type.
        for vtype in vtype2vars:
            if vtype == VariableType.MEASUREMENT:
                #* Skip fine-grained ingress for now (too many combinations).
                continue
            domaintype = TYPE_DOMIAN[vtype]
            if domaintype != DomainType.NUMERICAL: 
                #* Only augment numerical vars.
                continue
            
            numericals = vtype2vars[vtype]
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
        predicates: Set[str] = set()
        
        #! Restrict LHS to single variables.
        for j, lhs in enumerate(self.variables):
            vtype_lhs = variable_types[lhs]
            domaintype_lhs = TYPE_DOMIAN[vtype_lhs]
            lhs_vars = set([lhs])
            
            # if vtype_lhs == VariableType.WINDOW:
            #     continue
            
            #& Predicates with constants.
            for varname, constants in self.multiconstants:
                if varname != lhs:
                    #* Find the constants for this variable.
                    continue
                #& X=c
                if constants.kind == ConstantType.ASSIGNMENT:
                    for constant in constants.values:
                        predicates.add(f"Eq({lhs}, {constant})")
                        predicates.add(f"Ne({lhs}, {constant})")
                elif (domaintype_lhs == DomainType.CATEGORICAL
                    and vtype_lhs not in [VariableType.IP, VariableType.PORT]):
                    #* Don't create Eq/Ne predicates for identifiers (IP/PORT).
                    #* Only consider domain values if no constants are defined.
                    #& X=x
                    for value in self.domains[lhs].values:
                        predicates.add(f"Eq({lhs}, {value})")
                        predicates.add(f"Ne({lhs}, {value})")

                #& X > c and X ≤ c
                if constants.kind == ConstantType.LIMIT:
                    for constant in constants.values:
                        predicates.add(f"({lhs} > {constant})")
                        if constant != 0:
                            #! Assume no negative values.
                            predicates.add(f"({lhs} <= {constant})")
            #TODO: Move this to constructor.
            if vtype_lhs == VariableType.SEQUENCING:
                predicates.add(f"({lhs} > 1)")
                predicates.add(f"({lhs} <= 1)")
                
            # if domaintype_lhs == DomainType.CATEGORICAL:
                # #! Assuming one variable has only one type of constants.
                # if (lhs in self.constants 
                #     and self.constants[lhs].kind == ConstantType.ASSIGNMENT):
                #     #& X=c
                #     for constant in self.constants[lhs].values:
                #         predicates.add(f"Eq({lhs}, {constant})")
                #         predicates.add(f"Ne({lhs}, {constant})")
                # elif vtype_lhs not in [VariableType.IP, VariableType.PORT]:
                #     #* Don't create Eq/Ne predicates for identifiers (IP/PORT).
                #     #* Only consider domain values if no constants are defined.
                #     #& X=x
                #     for value in self.domains[lhs].values:
                #         predicates.add(f"Eq({lhs}, {value})")
                #         predicates.add(f"Ne({lhs}, {value})")
            # else:
                #& DomainType.NUMERICAL
                # if (lhs in self.constants
                #     and self.constants[lhs].kind == ConstantType.LIMIT):
                #     #& X > c and X ≤ c
                #     for constants in self.constants[lhs].values:
                #         predicates.add(f"({lhs} > {constants})")
                #         if constants != 0:
                #             #! Assume no negative values.
                #             predicates.add(f"({lhs} <= {constants})")
                #         #? Since we don't care equality (X=c), we omit the following:
                #         # predicates.add(f"({lhs} <= {constant})")
                #         # predicates.add(f"({lhs} > {constant})")
                # #TODO: Enable one var with multiple types of constants.
                # #*  Here, ACK and SEQ numbers require ADDITION (+1) and LIMIT (<1).
                # if vtype_lhs == VariableType.SEQUENCING:
                #     predicates.add(f"({lhs} > 1)")
                #     predicates.add(f"({lhs} <= 1)")
            
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
                if vtype_rhs != VariableType.MEASUREMENT:
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
                
                #* Doesn't make sense to compare sequencing variables other than equality.
                if (vtype_lhs != VariableType.SEQUENCING
                    and domaintype_lhs == DomainType.NUMERICAL ):
                    assert domaintype_rhs == DomainType.NUMERICAL, \
                        "LHS and RHS must have the same domain type."
                    #& Comparison predicates: A>B
                    predicate = f"({lhs}>{rhs})"
                    predicate_values = (self.examples[lhs]>self.examples[rhs]).astype(int)
                    if predicate_values.nunique() > 1:
                        self.examples[predicate] = predicate_values
                        predicates.add(predicate)
                    else:
                        if predicate_values.iloc[0] == 0:
                            predicate = f"({lhs}<={rhs})"
                        prior_rules.add(predicate)
                        continue
                    
                    #& Comparison predicates: A<=B
                    predicate = f"({lhs}<={rhs})"
                    predicate_values = (self.examples[lhs]<=self.examples[rhs]).astype(int)
                    assert predicate_values.nunique() > 1, \
                        "Predicate should have >1 unique value to this point."
                    self.examples[predicate] = predicate_values
                    predicates.add(predicate)
        #^ End for j, lhs in enumerate(self.variables)
        
        '''Add domain constraints to prior rules.'''
        self.examples = self.examples[self.variables]
        #* First drop identifiers
        #*  (if they have associated conts, e.g., Pts in CIDDS, they've been generated above).
        identifiers = vtype2vars[VariableType.IP]+vtype2vars[VariableType.PORT]
        for var in identifiers:
            if var in self.variables:
                self.variables.remove(var)
            if var in self.categoricals:
                self.categoricals.remove(var)
                
        for varname in self.variables:
            vtype = variable_types[varname]
            domaintype = TYPE_DOMIAN[vtype]
            if domaintype == DomainType.NUMERICAL:
                bounds = self.domains[varname].bounds
                prior_rules.add(f"({varname}>={bounds.lb})")
                prior_rules.add(f"({varname}<={bounds.ub})")
                
                #& Add default predicate X>0 if the domain contains 0 (except for ack and seq which already have >1).
                if (bounds.lb <= 0 <= bounds.ub 
                    and bounds.lb != bounds.ub 
                    and vtype != VariableType.SEQUENCING
                ):
                    #* Add X=0 as a default if the domain contains 0.
                    predicate = f"Eq({varname},0)"
                    predicate_values = (self.examples[varname]==0).astype(int)
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
                    
        log.info(f"Created {len(predicates)} predicates.")
        # pprint(list(predicates)[:5])
        log.info(f"Created {len(prior_rules)} prior rules.")
        # pprint(list(prior_rules)[:5])
        
        constraint_predicates = set()
        for p in tqdm(predicates, desc="... Converting predicates to constraints"):
            if p not in [sp.true, sp.false]:
                p = Constraint(sp.sympify(p))
                constraint_predicates.add(p)
        
        # #* Save predicates to file for inspection.
        # # pprint(prior_rules)
        # Theory.save_constraints(constraint_predicates, f'predicates_{self.dataset}_new.pl')
        # exit(0)
        
        log.info(f"Duplicate predicates found: {len(predicates) - len(constraint_predicates)}")
        
        self.prior = {Constraint(sp.sympify(r)) for r in prior_rules
                        if r not in [sp.true, sp.false]}
        
        return constraint_predicates

    
    def enumerate_minimal_hitting_sets_parallel(
        self,
        evidence_sets: List[frozenset[Constraint]],
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
    ) -> List[Set[Constraint]]:
        """
        Parallelized enumeration of all minimal hitting sets.
        Each top-level branch (first predicate choices) is explored in parallel.
        """
        if not evidence_sets:
            return []

        # Index: predicate -> evidence indices
        log.info("Indexing predicates in evidence sets...")
        idx_by_pred: Dict[Constraint, Set[int]] = defaultdict(set)
        for i, E in enumerate(evidence_sets):
            for p in E:
                idx_by_pred[p].add(i)

        E_list: List[List[Constraint]] = [list(E) for E in evidence_sets]

        log.info("Ordering predicates by coverage...")
        pred_by_cov = sorted(idx_by_pred.items(), key=lambda kv: -len(kv[1]))
        pred_order = [p for p, _ in pred_by_cov]

        FULL = set(range(len(E_list)))

        def _optimistic_lb(uncovered: Set[int]) -> int:
            if not uncovered:
                return 0
            best_gain = 0
            for p in pred_order:
                gain = len(uncovered & idx_by_pred[p])
                if gain > best_gain:
                    best_gain = gain
                    if best_gain == len(uncovered):
                        break
            return 1 if best_gain == 0 else (len(uncovered) + best_gain - 1) // best_gain

        def _pick_uncovered_with_smallest_branch(uncovered: Set[int]) -> int:
            best_i, best_deg = None, 10**9
            for i in uncovered:
                deg = len(E_list[i])
                if deg < best_deg:
                    best_deg = deg
                    best_i = i
            return best_i

        def dfs(chosen: Set[Constraint], covered: Set[int], local_solutions: List[frozenset[Constraint]]):
            # Covered all?
            if covered == FULL:
                fc = frozenset(chosen)
                # enforce minimality locally
                keep = []
                for s in local_solutions:
                    if s.issubset(fc):
                        if s != fc:
                            return
                    elif fc.issubset(s):
                        continue
                    keep.append(s)
                local_solutions.clear()
                local_solutions.extend(keep)
                local_solutions.append(fc)
                return

            if max_size is not None and len(chosen) >= max_size:
                return

            uncovered = FULL - covered
            if max_size is not None:
                lb = _optimistic_lb(uncovered)
                if len(chosen) + lb > max_size:
                    return

            pivot = _pick_uncovered_with_smallest_branch(uncovered)
            cand = sorted(E_list[pivot], key=lambda p: -len((idx_by_pred[p]) & uncovered))

            for p in cand:
                if p in chosen:
                    continue
                new_chosen = set(chosen)
                new_chosen.add(p)
                new_covered = covered | idx_by_pred[p]
                dfs(new_chosen, new_covered, local_solutions)

        # --- Parallel frontier expansion ---
        log.info("Launching parallel hitting set enumeration...")
        first_pivot = _pick_uncovered_with_smallest_branch(FULL)
        first_cands = sorted(E_list[first_pivot], key=lambda p: -len(idx_by_pred[p]))

        n_jobs = psutil.cpu_count()

        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda p: (
                p,
                dfs({p}, idx_by_pred[p], [])
            ))(p) for p in first_cands
        )

        # Merge results
        all_solutions: List[frozenset[Constraint]] = []
        for _, sols in results:
            if sols is not None:
                for s in sols:
                    # enforce global minimality
                    dominated = False
                    to_keep = []
                    for t in all_solutions:
                        if t.issubset(s):
                            dominated = True
                            break
                        elif s.issubset(t):
                            continue
                        to_keep.append(t)
                    if not dominated:
                        all_solutions = to_keep + [s]

        return [set(s) for s in all_solutions]
