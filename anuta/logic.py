from collections import defaultdict
import heapq
from itertools import combinations
import math
from pathlib import Path
import random
from multiprocess import Pool
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
from functools import lru_cache
import pickle
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import sys
sys.setrecursionlimit(100_000)


from anuta.grammar import (
    TYPE_DOMIAN, ConstantType, DomainType, VariableType, 
    group_variables_by_type_and_domain, get_variable_type, tautology, contradition)
from anuta.utils import *
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
        elif isinstance(expr, sp.GreaterThan):
            sat = (values1 >= values2)
        elif isinstance(expr, sp.StrictLessThan):
            sat = (values1 < values2)
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
    
    def __init__(self, constructor: Constructor, negative_constructor: Constructor = None, limit: int = None):
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
        
        self.negative_examples: Optional[pd.DataFrame] = None
        if negative_constructor:
            if limit and limit < negative_constructor.df.shape[0]:
                log.info(f"Limiting negative dataset to {limit} examples.")
                negative_constructor.df = negative_constructor.df.sample(n=limit, random_state=42)
            #* Ensure the negative dataset has the same schema.
            missing_cols = set(self.examples.columns) - set(negative_constructor.df.columns)
            assert not missing_cols, f"Negative dataset missing columns: {missing_cols}"
            self.negative_examples = negative_constructor.df[self.examples.columns].copy()
    
    def generate_analytical_predicates(self) -> Set[Constraint]:
        assert self.dataset == 'ana', f"Analytical predicates are only for 'ana' dataset, not {self.dataset}."
        assert not self.categoricals, "No categorical variables expected in 'ana' dataset."
        
        predicate_strs = set()
        prior_rules = set()
        #TODO: Add compound variables X-Y, X+Y, ... as lhs.
        for i, lhs in enumerate(self.variables):
            #& Compare with 0 as a special case.
            predicate = f"{lhs} > 0"
            predicate_values = (self.examples[lhs] > 0)
            if predicate_values.nunique() > 1:
                predicate_strs.add(predicate)
            else:
                if predicate_values.iloc[0] == True:
                    prior_rules.add(predicate)
                else:
                    prior_rules.add(f"Eq({lhs}, 0)")
            
            for rhs in self.variables[i+1:]:
                
                if not is_meaningful_pair(lhs, rhs):
                    continue
                
                # #& Equality and Inequality
                # predicate = f"Eq({lhs}, {rhs})"
                # predicate_values = (self.examples[lhs] == self.examples[rhs])
                # if predicate_values.nunique() > 1:
                #     predicate_strs.add(predicate)
                # else:
                #     if predicate_values.iloc[0] == True:
                #         prior_rules.add(predicate)
                #     else:
                #         prior_rules.add(f"Ne({lhs}, {rhs})")
                
                #& StrictGreaterThan
                predicate = f"{lhs} > {rhs}"
                predicate_values = (self.examples[lhs] > self.examples[rhs])
                if predicate_values.nunique() > 1:
                    predicate_strs.add(predicate)
                else:
                    if predicate_values.iloc[0] == True:
                        prior_rules.add(predicate)
                    else:
                        prior_rules.add(f"({lhs} <= {rhs})")
                
                #& StrictLessThan
                predicate = f"{lhs} < {rhs}"
                predicate_values = (self.examples[lhs] < self.examples[rhs])
                if predicate_values.nunique() > 1:
                    predicate_strs.add(predicate)
                else:
                    if predicate_values.iloc[0] == True:
                        prior_rules.add(predicate)
                    else:
                        prior_rules.add(f"({lhs} >= {rhs})")
        
        #& Add variable domain bounds as prior rules.
        for varname in self.variables:
            bounds = self.domains[varname].bounds
            prior_rules.add(f"({varname}>={bounds.lb})")
            prior_rules.add(f"({varname}<={bounds.ub})")
        
        log.info(f"Generated {len(predicate_strs)} analytical predicates.")
        log.info(f"Generated {len(prior_rules)} prior rules from constant predicates.")
        
        constraint_predicates = set()
        for p in tqdm(predicate_strs, desc="... Converting predicates to constraints"):
            if p not in [sp.true, sp.false]:
                p = Constraint(sp.sympify(p))
                constraint_predicates.add(p)
        
        for p in tqdm(prior_rules, desc="... Converting prior rules to constraints"):
            p = Constraint(sp.sympify(p))
            self.prior.add(p)
            
        #* Save predicates to file for inspection.
        # pprint(prior_rules)
        # Theory.save_constraints(constraint_predicates, f'ana_preds_{self.dataset}.pl')
        # Theory.save_constraints(self.prior, f'ana_priors_{self.dataset}.pl')
        # exit(0)
        return constraint_predicates
                
    
    def _filter_predicates_against_negatives(
        self,
        predicates: Set[Constraint],
    ) -> Set[Constraint]:
        """
        Remove predicates (and priors) that ever evaluate to True on negative examples.
        Any disjunction built from the remaining predicates will be False on all negatives.
        """
        if self.negative_examples is None or self.negative_examples.empty:
            return predicates
        
        start = perf_counter()
        combined = predicates | self.prior
        mask_by_pred, valid_preds = evaluate_predicates_with_masks(combined, self.negative_examples)

        def _is_safe(p: Constraint) -> bool:
            if p in valid_preds:
                return False  # always true on negatives
            return mask_by_pred.get(p, 0) == 0  # never satisfied on negatives

        filtered_predicates = {p for p in predicates if _is_safe(p)}
        filtered_prior = {p for p in self.prior if _is_safe(p)}

        dropped_predicates = len(predicates) - len(filtered_predicates)
        dropped_prior = len(self.prior) - len(filtered_prior)
        elapsed = perf_counter() - start
        log.info(
            f"Negative filtering: kept {len(filtered_predicates)}/{len(predicates)} predicates "
            f"(dropped {dropped_predicates}), dropped {dropped_prior} priors in {elapsed:.2f}s."
        )

        self.prior = filtered_prior
        return filtered_predicates
    
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
        if self.negative_examples is not None:
            predicates = self._filter_predicates_against_negatives(predicates)
            if not predicates:
                log.warning("All predicates were eliminated by negative examples; saving priors only.")
                Theory.save_constraints({pr.expr for pr in self.prior}, "denial_priors_only.pl")
                return
        # predicates: Set[Constraint] = self.generate_analytical_predicates()

        start = perf_counter()
        evidence_sets: List[frozenset[Constraint]] = self.build_evidence_set(predicates, df=self.examples)
        end = perf_counter()
        log.info(f"Obtained {len(evidence_sets)} evidence sets in {end - start:.2f} seconds.\n")
        
        assert len(evidence_sets) == self.examples.shape[0], \
            f"Evidence sets size {len(evidence_sets)} != the # of examples {self.examples.shape[0]}."    
        
        if not evidence_sets:
            log.warning("No evidence sets were produced; returning only prior.")

        start = perf_counter()
        covers, stop_reason = self.enumerate_minimal_hitting_sets(
            evidence_sets=evidence_sets,
            max_size=max_predicates,
            max_solutions=max_learned_rules,
        )
        end = perf_counter()
        print()
        log.info(f"Enumerated {len(covers)} minimal hitting sets in {(end - start)/60:.2f} minutes.")
        if stop_reason not in ("exhausted", "unknown"):
            log.info(f"Hitting-set search stopped due to: {stop_reason}")
        
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
        outputf = f"denial_{self.dataset}_{self.num_examples}_p{max_predicates}"
        if FLAGS.label:
            outputf += f"_{FLAGS.label}.pl"
        else:            
            outputf += ".pl"
        Theory.save_constraints(learned_rules, outputf)
        return
    
    def enumerate_minimal_hitting_sets(
        self,
        evidence_sets: List[FrozenSet[Constraint]],
        max_size: Optional[int] = None,
        max_solutions: Optional[int] = None,
    ) -> Tuple[List[Set[Constraint]], str]:
        """
        Enumerate all minimal hitting sets H such that ∀E in evidence_sets: H ∩ E ≠ ∅.
        
        Iterative stack-based DFS (no recursion).
        Prunes branches whose disjunction is tautological (e.g., conflicting Ne or complementary bounds).

        Size scheduling
        ---------------
        If `max_size` is provided and >2, the search is run with an increasing size bound:
          - start with `current_max_size = 2` (or `max_size` if `max_size < 2`)
          - whenever the search under the current bound is exhausted, or the stall timeout triggers,
            bump `current_max_size += 1` (up to `max_size`) and restart the DFS.

        Note: restarting avoids retaining a potentially enormous "frontier" of pruned states in memory
        when increasing the size bound.
        """
        if not evidence_sets:
            return [], "exhausted"

        # Build idx_by_pred: Map<Predicate, Set<EvidenceSetIndices>>
        idx_by_pred: Dict[Constraint, Set[int]] = defaultdict(set)
        empty_esets: Set[int] = set()
        evidence_sizes: List[int] = []
        for i, E in enumerate(tqdm(evidence_sets, desc="... Indexing predicates")):
            evidence_sizes.append(len(E))
            if not E:
                empty_esets.add(i)
            for p in E:
                idx_by_pred[p].add(i)
        
        # If any evidence set is empty, no hitting set is possible.
        if empty_esets:
            log.warning(f"Found {len(empty_esets)} empty evidence sets out of {len(evidence_sets)}.")
            log.warning("Problem is unsatisfiable due to one or more empty evidence sets.")
            return [], "unsatisfiable"

        # All evidence sets that must be covered (all of them, since empty_esets is 0)
        FULL = set(range(len(evidence_sets)))
        signature_by_pred: Dict[Constraint, Optional[Tuple[str, str, Any, bool]]] = {}
        for pred in idx_by_pred:
            signature_by_pred[pred] = None

        #* --- PERFORMANCE FIX (bitmask-based minimality tracking) ---
        pred_list: List[Constraint] = list(idx_by_pred.keys())
        pred_to_bit: Dict[Constraint, int] = {p: i for i, p in enumerate(pred_list)}
        global_solution_masks: Set[int] = set()
        masks_by_bit: Dict[int, Set[int]] = defaultdict(set)
        # Index solutions by their minimum bit to support fast "subset-of" pruning in the DFS:
        # if a partial chosen mask already contains any known minimal solution, no extension can be minimal.
        solutions_by_minbit: Dict[int, Set[int]] = defaultdict(set)
        solution_minbits_mask: int = 0

        def _bits_from_mask(mask: int) -> List[int]:
            bits: List[int] = []
            bit = 0
            m = mask
            while m:
                if m & 1:
                    bits.append(bit)
                bit += 1
                m >>= 1
            return bits

        def _mask_for(preds: Set[Constraint]) -> int:
            mask = 0
            for p in preds:
                mask |= 1 << pred_to_bit[p]
            return mask

        def _minbit_index(mask: int) -> int:
            """Index of the least-significant set bit. Undefined for mask=0."""
            return (mask & -mask).bit_length() - 1

        def _has_subset(mask: int) -> bool:
            """
            Returns True if any known minimal solution mask is a subset of `mask`.
            Used for:
              - minimality filtering when adding a new solution, and
              - pruning DFS states whose `chosen` already contains a known solution.
            """
            if not global_solution_masks:
                return False
            relevant_minbits = mask & solution_minbits_mask
            while relevant_minbits:
                lsb = relevant_minbits & -relevant_minbits
                b = lsb.bit_length() - 1
                for sol in solutions_by_minbit.get(b, ()):
                    if (sol & mask) == sol:
                        return True
                relevant_minbits ^= lsb
            return False

        def _add_solution(mask: int) -> bool:
            """
            Adds mask as a new minimal solution if not dominated.
            Removes supersets using per-bit indexes to avoid scanning all solutions.
            Returns True if added.
            """
            nonlocal solution_minbits_mask

            if _has_subset(mask):
                #* Check if any subsets of the selected predicates (stronger rule)
                #*  already exist as solutions.
                return False

            bits = _bits_from_mask(mask)
            #* Check if any supersets (weaker rule) of the newly found rule exist.
            superset_candidates: Optional[Set[int]] = None
            for b in bits:
                superset_masks = masks_by_bit.get(b, set())
                if superset_candidates is None:
                    superset_candidates = set(superset_masks)
                else:
                    superset_candidates &= superset_masks
                if superset_candidates is not None and not superset_candidates:
                    break

            if superset_candidates:
                for sup in superset_candidates:
                    global_solution_masks.discard(sup)
                    # Remove from minbit index
                    sup_minb = _minbit_index(sup)
                    solutions_by_minbit[sup_minb].discard(sup)
                    if not solutions_by_minbit[sup_minb]:
                        del solutions_by_minbit[sup_minb]
                        solution_minbits_mask &= ~(1 << sup_minb)
                    for b in _bits_from_mask(sup):
                        masks_by_bit[b].discard(sup)

            global_solution_masks.add(mask)
            for b in bits:
                masks_by_bit[b].add(mask)
            minb = _minbit_index(mask)
            solutions_by_minbit[minb].add(mask)
            solution_minbits_mask |= 1 << minb
            return True

        def _optimistic_lb(uncovered: Set[int]) -> int:
            """Calculates a simple greedy lower bound on the number of predicates needed."""
            if not uncovered:
                return 0
            best_gain = 0
            for p in idx_by_pred:
                gain = len(uncovered & idx_by_pred[p])
                best_gain = max(best_gain, gain)
                if best_gain == len(uncovered): # Perfect predicate found
                    return 1
            
            # Ceiling division to get lower bound
            return 1 if best_gain == 0 else ((len(uncovered) + best_gain - 1) // best_gain)

        #* --- PERFORMANCE FIX (efficient pivot selection) ---
        def _pick_pivot(uncovered: Set[int]) -> int:
            """Picks the uncovered evidence set with the smallest size (branching factor)."""
            best_i, best_deg = None, float('inf')
            for i in uncovered:
                deg = evidence_sizes[i]
                if deg < best_deg:
                    best_deg, best_i = deg, i
            return best_i

        #* --- Tautology-aware candidate filtering ---
        @lru_cache(maxsize=None)
        def _extract_signature(p: Constraint) -> Optional[Tuple[str, str, Any, bool]]:
            """
            Returns (var, kind, rhs, inclusive) where kind in {'eq','ne','lower','upper'}.
            Only defined for single-variable predicates; otherwise returns None.
            """
            cached = signature_by_pred.get(p)
            if cached is not None:
                return cached
            expr = p.expr
            if len(expr.free_symbols) != 1:
                signature_by_pred[p] = None
                return None
            var = str(next(iter(expr.free_symbols)))
            rhs = expr.args[1]
            if isinstance(expr, sp.Eq):
                sig = (var, "eq", rhs, True)
            elif isinstance(expr, sp.Ne):
                sig = (var, "ne", rhs, False)
            elif isinstance(expr, sp.StrictGreaterThan):
                sig = (var, "lower", rhs, False)
            elif isinstance(expr, sp.GreaterThan):
                sig = (var, "lower", rhs, True)
            elif isinstance(expr, sp.StrictLessThan):
                sig = (var, "upper", rhs, False)
            elif isinstance(expr, sp.LessThan):
                sig = (var, "upper", rhs, True)
            else:
                sig = None
            signature_by_pred[p] = sig
            return sig

        # Precompute signatures once to avoid repeated Sympy walks during search.
        for pred in signature_by_pred:
            signature_by_pred[pred] = _extract_signature(pred)

        def _interval_covers_all(lower_val, lower_inc: bool, upper_val, upper_inc: bool) -> bool:
            """
            True if lower/upper bounds together cover the full line (their disjunction is tautology).
            """
            try:
                if upper_val > lower_val:
                    return True
                if upper_val == lower_val:
                    return lower_inc or upper_inc
            except Exception:
                return False
            return False

        def _is_tautological_with(chosen_info: Dict[str, Dict[str, Any]], p: Constraint) -> bool:
            sig = signature_by_pred.get(p)
            if sig is None:
                sig = _extract_signature(p)
            if not sig:
                return False
            var, kind, rhs, inclusive = sig
            info = chosen_info.get(var)
            if not info:
                return False

            if kind == "eq":
                return rhs in info["ne"]

            if kind == "ne":
                if rhs in info["eq"]:
                    return True
                #* Any additional Ne on the same variable makes OR tautological.
                #*  E.g., Ne(X, x1) ∨ Ne(X, x2) is always true for any X.
                return bool(info["ne"])

            if kind == "lower":
                for ub, uinc in info["upper"]:
                    if _interval_covers_all(rhs, inclusive, ub, uinc):
                        return True
                return False

            if kind == "upper":
                for lb, linc in info["lower"]:
                    if _interval_covers_all(lb, linc, rhs, inclusive):
                        return True
                return False

            return False

        def _extend_chosen_info(chosen_info: Dict[str, Dict[str, Any]], p: Constraint) -> Dict[str, Dict[str, Any]]:
            sig = signature_by_pred.get(p)
            if sig is None:
                sig = _extract_signature(p)
            if not sig:
                return chosen_info
            var, kind, rhs, inclusive = sig
            var_info_existing = chosen_info.get(var)
            # Copy only the touched variable info to reduce per-branch cloning.
            if var_info_existing is None:
                var_info = {"eq": set(), "ne": set(), "lower": [], "upper": []}
            else:
                var_info = {
                    "eq": set(var_info_existing["eq"]),
                    "ne": set(var_info_existing["ne"]),
                    "lower": list(var_info_existing["lower"]),
                    "upper": list(var_info_existing["upper"]),
                }

            if kind == "eq":
                var_info["eq"].add(rhs)
            elif kind == "ne":
                var_info["ne"].add(rhs)
            elif kind == "lower":
                var_info["lower"].append((rhs, inclusive))
            elif kind == "upper":
                var_info["upper"].append((rhs, inclusive))

            if var_info_existing is None and not chosen_info:
                # Fast path: no existing info, avoid full dict copy.
                return {var: var_info}
            new_info = dict(chosen_info)
            new_info[var] = var_info
            return new_info

        # Size scheduling
        target_max_size = max_size
        current_max_size: Optional[int]
        if target_max_size is None:
            current_max_size = None
        else:
            current_max_size = 2 if target_max_size > 2 else target_max_size

        progress = tqdm(
            desc=f"... Enumerating hitting sets (max_size={current_max_size})",
            unit=" sets",
            total=max_solutions,
        )

        rng = random.Random(None)

        overall_start_time = perf_counter()
        stop_reason = "unknown"

        try:
            while True:
                # frame: (chosen, chosen_mask, covered, uncovered, chosen_info, candidates, next_idx)
                stack: List[
                    Tuple[
                        Set[Constraint],
                        int,
                        Set[int],
                        Set[int],
                        Dict[str, Dict[str, Any]],
                        Optional[List[Constraint]],
                        int,
                    ]
                ] = [(set(), 0, set(), FULL, {}, None, 0)]

                stage_last_solution_time = perf_counter()
                stage_stop_reason = "exhausted"

                while stack:
                    now = perf_counter()
                    if now - overall_start_time > cfg.TIMEOUT_SEC:
                        log.warning(f"Learning timed out after {cfg.TIMEOUT_SEC//60}min.")
                        stage_stop_reason = "timeout"
                        break

                    if now - stage_last_solution_time > cfg.STALL_TIMEOUT_SEC:
                        log.warning(
                            f"Search stalled for {cfg.STALL_TIMEOUT_SEC//60}min at max_size={current_max_size}."
                        )
                        stage_stop_reason = "stall_timeout"
                        break

                    chosen, chosen_mask, covered, uncovered, chosen_info, candidates, next_idx = stack.pop()

                    # If we already contain a known minimal solution, any extension is a non-minimal superset.
                    if chosen_mask and _has_subset(chosen_mask):
                        continue

                    #* --- SOLUTION FOUND (Optimized) ---
                    if not uncovered:
                        if chosen_mask and _add_solution(chosen_mask):
                            stage_last_solution_time = perf_counter()
                            # Track the number of *current* minimal solutions (not total additions),
                            # since `_add_solution` may remove dominated supersets.
                            progress.n = len(global_solution_masks)
                            progress.refresh()

                            if max_solutions and len(global_solution_masks) >= max_solutions:
                                stage_stop_reason = "max_solutions"
                                break
                        continue

                    #* --- PRUNING ---
                    if current_max_size is not None and len(chosen) >= current_max_size:
                        continue

                    if current_max_size is not None:
                        lb = _optimistic_lb(uncovered)
                        if len(chosen) + lb > current_max_size:
                            continue

                    #* --- CANDIDATE GENERATION (Optimized) ---
                    if candidates is None:
                        pivot = _pick_pivot(uncovered)

                        scored_candidates: List[Tuple[int, Constraint]] = []
                        for p in evidence_sets[pivot]:
                            if _is_tautological_with(chosen_info, p):
                                continue
                            gain = len(idx_by_pred[p] & uncovered)
                            if gain == 0:
                                continue
                            scored_candidates.append((gain, p))
                        if not scored_candidates:
                            continue
                        # Sort candidates to explore most promising branches first; tie-break with RNG for diversity.
                        scored_candidates.sort(key=lambda gp: (-gp[0], rng.random()))
                        candidates = [p for _, p in scored_candidates]
                        next_idx = 0

                    # Backtracking: exhausted all candidates for this pivot
                    if next_idx >= len(candidates):
                        continue

                    #* --- STACK PUSH (Branch 1: "Don't choose p") ---
                    # This state explores solutions *without* candidates[next_idx]
                    stack.append((chosen, chosen_mask, covered, uncovered, chosen_info, candidates, next_idx + 1))

                    #* --- STACK PUSH (Branch 2: "Choose p") ---
                    p = candidates[next_idx]

                    new_chosen = set(chosen)
                    new_chosen.add(p)
                    new_chosen_mask = chosen_mask | (1 << pred_to_bit[p])

                    new_covered = covered | idx_by_pred[p]
                    new_info = _extend_chosen_info(chosen_info, p)
                    new_uncovered = uncovered - idx_by_pred[p]

                    stack.append((new_chosen, new_chosen_mask, new_covered, new_uncovered, new_info, None, 0))

                stop_reason = stage_stop_reason

                if stop_reason in ("max_solutions", "timeout"):
                    break

                if stop_reason not in ("exhausted", "stall_timeout"):
                    break

                if current_max_size is None or target_max_size is None:
                    break

                if current_max_size >= target_max_size:
                    break

                prev = current_max_size
                current_max_size += 1
                if stop_reason == "exhausted":
                    log.info(f"Search exhausted at max_size={prev}; increasing to {current_max_size}.")
                else:
                    log.warning(
                        f"Search stalled at max_size={prev}; increasing to {current_max_size}."
                    )
                progress.set_description(f"... Enumerating hitting sets (max_size={current_max_size})")
                # Continue to the next stage (restart DFS with the new bound).
                continue

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            log.warning("\nSearch interrupted by user (Ctrl+C). Returning partial results.")
        finally:
            progress.close()  # Ensure progress bar is always closed

        merged_masks = set(global_solution_masks)

        # Convert final solution masks to List[Set[Constraint]]
        solutions: List[Set[Constraint]] = []
        for mask in merged_masks:
            pred_set = {pred_list[i] for i in _bits_from_mask(mask)}
            solutions.append(pred_set)
        return solutions, stop_reason

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
        outputf = f"levelwise_{self.dataset}_{self.num_examples}_e{epoch}"
        if FLAGS.label:
            outputf += f"_{FLAGS.label}.pl"
        else:            
            outputf += ".pl"
        Theory.save_constraints(learned_rules, outputf)
        return
    
    def build_evidence_set(self, predicates: Set[Constraint], df: Optional[pd.DataFrame] = None, save: bool = True) -> List[Set[Constraint]]:
        # Pre-pack predicates into a list so all workers use same reference
        examples_df = df if df is not None else self.examples
        evidence_sets: List[Set[Constraint]] = []
        eq_rtol = getattr(cfg, "NUMERIC_EQ_RTOL", None)
        eq_atol = getattr(cfg, "NUMERIC_EQ_ATOL", None)
        if eq_rtol is None and eq_atol is None:
            # Backward compatibility (deprecated): treat RESOLUTION as both relative/absolute tolerance.
            eq_resolution = getattr(cfg, "NUMERIC_EQ_RESOLUTION", 0.0)
            eq_rtol = eq_resolution
            eq_atol = eq_resolution
        try:
            eq_rtol = float(eq_rtol or 0.0)
        except Exception:
            eq_rtol = 0.0
        try:
            eq_atol = float(eq_atol or 0.0)
        except Exception:
            eq_atol = 0.0
        if eq_rtol < 0:
            eq_rtol = 0.0
        if eq_atol < 0:
            eq_atol = 0.0
        use_isclose = (eq_rtol > 0.0) or (eq_atol > 0.0)

        def _is_intlike(value: Any, value_f: float) -> bool:
            if isinstance(value, (bool, np.bool_)):
                return True
            if isinstance(value, (int, np.integer)):
                return True
            is_integer = getattr(value, "is_integer", None)
            if is_integer is True:
                return True
            if isinstance(value, (float, np.floating)):
                return float(value).is_integer()
            try:
                return float(value_f).is_integer()
            except Exception:
                return False

        def _satisfies(expr: sp.Expr, row_dict: Dict[str, Any]) -> bool:
            """
            Evaluate predicate with numeric-tolerant Eq/Ne.

            If `cfg.NUMERIC_EQ_RTOL/ATOL` are set, Eq/Ne over numeric expressions use
            math.isclose-style comparison to avoid false negatives due to float precision.
            """
            if use_isclose and isinstance(expr, (sp.Eq, sp.Ne)):
                lhs, rhs = expr.args
                lhs_val = lhs.subs(row_dict)
                rhs_val = rhs.subs(row_dict)
                try:
                    lhs_f = float(lhs_val)
                    rhs_f = float(rhs_val)
                except Exception:
                    return bool(expr.subs(row_dict))

                if not (math.isfinite(lhs_f) and math.isfinite(rhs_f)):
                    return False

                if _is_intlike(lhs_val, lhs_f) and _is_intlike(rhs_val, rhs_f):
                    eq = int(lhs_f) == int(rhs_f)
                else:
                    eq = math.isclose(lhs_f, rhs_f, rel_tol=eq_rtol, abs_tol=eq_atol)

                if isinstance(expr, sp.Eq):
                    return eq
                return not eq

            return bool(expr.subs(row_dict))

        #* First check if the evidence sets file exists, if so, load it.
        data_dir = FLAGS.config.DATA_DIR
        evidence_sets_file = f'{data_dir}/evidence_sets_{self.dataset}_{self.num_examples}_{FLAGS.label}.pkl' \
            if FLAGS.label else  f'{data_dir}/evidence_sets_{self.dataset}_{self.num_examples}.pkl'
        if Path(evidence_sets_file).exists():
            log.info(f"Loading evidence sets from {evidence_sets_file}...")
            with open(evidence_sets_file, 'rb') as f:
                evidence_sets = pickle.load(f)
            log.info(f"Loaded {len(evidence_sets)} evidence sets.")
        
            missing_predicates = set()
            # ! Below check doesn't consider contradictory predicates that no example can satisfy.
            existing_predicates = [_predicates for _predicates in 
                                tqdm(evidence_sets,
                                        desc="... Collecting existing predicates from evidence sets")]
            missing_predicates = predicates - set().union(*existing_predicates)
            if missing_predicates:
                log.warning(f"Missing {len(missing_predicates)} predicates in the evidence set (usually invalid ones).")
            else:
                log.info("All predicates already covered in the evidence sets.")
            return evidence_sets
            
        predicates_list = list(predicates) # list() if not missing_predicates else list(missing_predicates)
        log.info(f"Evaluating {len(predicates_list)} predicates over {examples_df.shape[0]} examples...")
        # pprint(missing_predicates)
        # exit(0)

        def _process_examples(worker_id, df_chunk: pd.DataFrame) -> List[Set[Constraint]]:
            chunk_results = []
            for _, row in tqdm(df_chunk.iterrows(), 
                               total=df_chunk.shape[0], 
                               desc=f"... Building evidence sets (worker {worker_id})"):
                row_dict = row.to_dict()
                satisfied = set()
                for p in predicates_list:
                    # try:
                    if _satisfies(p.expr, row_dict):
                        satisfied.add(p)
                    # except Exception:
                    #     # Any substitution/eval failure = not satisfied
                    #     continue
                # if not satisfied:
                #     log.warning(f"Example {row.name} satisfied no predicates.")
                assert satisfied, f"Example {row.name} satisfied no predicates. Improve predicate space!!!"
                chunk_results.append(satisfied)
            return chunk_results

        n_jobs = psutil.cpu_count()

        # Split into n_jobs chunks (avoid overhead of too many tasks)
        chunks = np.array_split(examples_df, n_jobs)
        log.info(f"Processing {len(chunks)} batches with {n_jobs} workers.")

        # Parallel execution — each worker gets a big chunk
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_examples)(i, chunk)
            for i, chunk in enumerate(chunks)
        )
        
        results: List[Set[Constraint]] = [s for chunk in results for s in chunk]  # Flatten list of lists
        #* Merge with existing evidence sets if any
        if evidence_sets:
            if len(evidence_sets) != len(results):
                log.error(f"Existing evidence sets size {len(evidence_sets)} != new results size {len(results)}.")
            else:
                for i in tqdm(range(len(results)), desc="... Merging evidence sets"):
                    results[i] = results[i].union(evidence_sets[i])
        
        evidence_sets_file = f'./data/evidence_sets_{self.dataset}_{self.num_examples}_{FLAGS.label}.pkl' \
            if FLAGS.label else  f'./data/evidence_sets_{self.dataset}_{self.num_examples}.pkl'
        if save:
            with open(evidence_sets_file, 'wb') as f:
                pickle.dump(results, f)
                log.info(f"Saved evidence sets to {evidence_sets_file}.")

        return results

    def _generate_abr_template_predicates(self) -> Tuple[Set[str], Set[str]]:
        predicates: Set[str] = set()
        priors: Set[str] = set()

        required = {"ThroughputMbps", "DelayMs", "ChosenChunkBytes"}
        missing = required - set(self.examples.columns)
        if missing:
            log.warning(f"[ABR] Skipping ABR templates; missing columns: {sorted(missing)}")
            return predicates, priors

        df = self.examples
        thr = pd.to_numeric(df["ThroughputMbps"], errors="coerce")
        delay = pd.to_numeric(df["DelayMs"], errors="coerce")
        chunk = pd.to_numeric(df["ChosenChunkBytes"], errors="coerce")

        eq_rtol = getattr(cfg, "NUMERIC_EQ_RTOL", None)
        eq_atol = getattr(cfg, "NUMERIC_EQ_ATOL", None)
        if eq_rtol is None and eq_atol is None:
            # Backward compatibility (deprecated): treat RESOLUTION as both relative/absolute tolerance.
            eq_resolution = getattr(cfg, "NUMERIC_EQ_RESOLUTION", 0.0)
            eq_rtol = eq_resolution
            eq_atol = eq_resolution
        try:
            eq_rtol = float(eq_rtol or 0.0)
        except Exception:
            eq_rtol = 0.0
        try:
            eq_atol = float(eq_atol or 0.0)
        except Exception:
            eq_atol = 0.0
        if eq_rtol < 0:
            eq_rtol = 0.0
        if eq_atol < 0:
            eq_atol = 0.0
        use_isclose = (eq_rtol > 0.0) or (eq_atol > 0.0)

        lhs_bits = thr.astype(float) * 1000.0 * delay.astype(float)
        rhs_bits = chunk.astype(float) * 8.0
        abs_err = (lhs_bits - rhs_bits).abs()

        def _keep(expr: str, mask: pd.Series) -> None:
            mask = mask.fillna(False).astype(bool)
            if mask.nunique() > 1:
                predicates.add(expr)
                return
            if len(mask) and bool(mask.iloc[0]):
                priors.add(expr)

        # Exact cross-multiplied equality:
        #   ThroughputMbps*1000 (bits/ms) * DelayMs (ms) == ChosenChunkBytes*8 (bits)
        if use_isclose:
            eq_mask = pd.Series(
                np.isclose(lhs_bits, rhs_bits, rtol=eq_rtol, atol=eq_atol),
                index=df.index,
            )
        else:
            eq_mask = lhs_bits == rhs_bits
        _keep(
            "Eq(ThroughputMbps*1000*DelayMs, ChosenChunkBytes*8)",
            eq_mask,
        )

        # Relative-error templates (scale with chunk size).
        for eps in (0.01, 0.05, 0.1):
            _keep(
                f"(Abs(ThroughputMbps*1000*DelayMs - ChosenChunkBytes*8) <= {eps}*ChosenChunkBytes*8)",
                abs_err <= (eps * rhs_bits),
            )

        # Absolute-error templates (data-driven cutoffs).
        try:
            abs_err_thresholds = get_quantiles(abs_err)
        except Exception:
            abs_err_thresholds = []
        for t in abs_err_thresholds:
            try:
                t_val = float(t)
            except Exception:
                continue
            if t_val <= 0:
                continue
            _keep(
                f"(Abs(ThroughputMbps*1000*DelayMs - ChosenChunkBytes*8) <= {t_val})",
                abs_err <= t_val,
            )

        # Bitrate should generally be <= throughput (in compatible units).
        if "BitrateKbps" in df.columns:
            bitrate = pd.to_numeric(df["BitrateKbps"], errors="coerce").astype(float)
            throughput_kbps = thr.astype(float) * 1000.0
            _keep(
                "(BitrateKbps<1000*ThroughputMbps)",
                bitrate < throughput_kbps,
            )
            _keep(
                "(BitrateKbps>1000*ThroughputMbps)",
                bitrate > throughput_kbps,
            )
            _keep(
                "Eq(BitrateKbps,1000*ThroughputMbps)",
                pd.Series(
                    np.isclose(bitrate, throughput_kbps, rtol=eq_rtol, atol=eq_atol),
                    index=df.index,
                )
                if use_isclose
                else (bitrate == throughput_kbps),
            )

            bitrate_x_time = bitrate * delay.astype(float)
            _keep(
                "(BitrateKbps*DelayMs<8*ChosenChunkBytes)",
                bitrate_x_time < rhs_bits,
            )
            _keep(
                "(BitrateKbps*DelayMs>8*ChosenChunkBytes)",
                bitrate_x_time > rhs_bits,
            )
            _keep(
                "Eq(BitrateKbps*DelayMs,8*ChosenChunkBytes)",
                pd.Series(
                    np.isclose(bitrate_x_time, rhs_bits, rtol=eq_rtol, atol=eq_atol),
                    index=df.index,
                )
                if use_isclose
                else (bitrate_x_time == rhs_bits),
            )

        return predicates, priors


    def generate_predicates_and_prior(self) -> Set[Constraint]:
        vtype2vars, domaintype2vars = group_variables_by_type_and_domain(self.variables)
        variable_types = self.vtypes
        # pprint(vtype2vars)
        # exit(0)
        
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
            if vtype in [VariableType.MEASUREMENT, VariableType.AGGREGATE, VariableType.CONNECTION]:
                #* Skip fine-grained ingress for now (too many combinations).
                continue
            domaintype = TYPE_DOMIAN[vtype]
            if domaintype == DomainType.CATEGORICAL:
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
            lhs_unique_values = None
            lhs_num_unique = None
            for varname, constants in self.multiconstants:
                if varname != lhs:
                    #* Find the constants for this variable.
                    continue
                #& X=c
                if constants.kind == ConstantType.ASSIGNMENT:
                    if lhs_unique_values is None:
                        lhs_unique_values = set(self.examples[lhs].unique())
                        lhs_num_unique = len(lhs_unique_values)
                    for constant in constants.values:
                        eq_predicate = f"Eq({lhs}, {constant})"
                        ne_predicate = f"Ne({lhs}, {constant})"
                        #* Add only predicates that vary across the dataset; otherwise record as priors.
                        if lhs_num_unique and lhs_num_unique > 1 and constant in lhs_unique_values:
                            predicates.add(eq_predicate)
                            predicates.add(ne_predicate)
                        else:
                            #* Invert if equality is always false.
                            prior_rules.add(ne_predicate if constant not in lhs_unique_values else eq_predicate)

                #* Use strict inequality, the negations will cover the other side.
                if constants.kind == ConstantType.LIMIT:
                    if lhs_unique_values is None:
                        lhs_unique_values = set(self.examples[lhs].unique())
                        lhs_num_unique = len(lhs_unique_values)
                    for constant in constants.values:
                        #& X > c 
                        predicate = f"({lhs} > {constant})"
                        predicate_values = (self.examples[lhs] > constant).astype(int)
                        if predicate_values.nunique() > 1:
                            predicates.add(predicate)
                        else:
                            if predicate_values.iloc[0] == 0:
                                predicate = f"({lhs} <= {constant})"
                            prior_rules.add(predicate)
                        
                        #& X < c
                        if constant != 0:
                            #! Assume no negative values.
                            predicate = f"({lhs} < {constant})"
                            predicate_values = (self.examples[lhs] < constant).astype(int)
                            if predicate_values.nunique() > 1:
                                predicates.add(predicate)
                            else:
                                if predicate_values.iloc[0] == 0:
                                    predicate = f"({lhs} >= {constant})"
                                prior_rules.add(predicate)

                        #& Equality predicate at the boundary (to cover >= / <= via disjunction).
                        #* Only add if equality can occur in the data; otherwise it can't help cover anything.
                        if constant in lhs_unique_values:
                            eq_predicate = f"Eq({lhs}, {constant})"
                            if eq_predicate not in predicates and eq_predicate not in prior_rules:
                                if lhs_num_unique and lhs_num_unique > 1:
                                    predicates.add(eq_predicate)
                                else:
                                    prior_rules.add(eq_predicate)
                        
            if (domaintype_lhs == DomainType.CATEGORICAL
                and vtype_lhs not in [VariableType.IP, VariableType.PORT]):
                #* Don't create Eq/Ne predicates for identifiers (IP/PORT).
                #* Only consider domain values if no constants are defined.
                #& X=x
                if lhs_unique_values is None:
                    lhs_unique_values = set(self.examples[lhs].unique())
                    lhs_num_unique = len(lhs_unique_values)
                for value in self.domains[lhs].values:
                    eq_predicate = f"Eq({lhs}, {value})"
                    ne_predicate = f"Ne({lhs}, {value})"
                    if lhs_num_unique and lhs_num_unique > 1 and value in lhs_unique_values:
                        predicates.add(eq_predicate)
                        predicates.add(ne_predicate)
                    else:
                        #* Invert if equality is always false.
                        prior_rules.add(ne_predicate if value not in lhs_unique_values else eq_predicate)
            #TODO: Move this to constructor.
            if vtype_lhs == VariableType.SEQUENCING:
                predicates.add(f"({lhs} > 1)")
                predicates.add(f"({lhs} <= 1)")
            
            #* Allow all types of variables in the RHS.
            #! Order matters with `j`
            all_vars = self.variables + list(avars)
            for rhs in all_vars[j+1: ]:
                assert lhs != rhs, "LHS and RHS cannot be the same variable."
                vtype_rhs = variable_types[rhs]
                domaintype_rhs = TYPE_DOMIAN[vtype_rhs]
                rhs_vars = set()
                if vtype_lhs != vtype_rhs or vtype_lhs == VariableType.UNKNOWN:
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
                        
                        #& Inequality predicates: Ne(A,B)
                        predicate = f"Ne({lhs},{rhs})"
                        predicate_values = (self.examples[lhs]!=self.examples[rhs]).astype(int)
                        #* Should have >1 unique value to this point.
                        assert predicate_values.nunique() > 1, \
                            "Ne predicate should have more than one unique value at this point."
                        self.examples[predicate] = predicate_values
                        predicates.add(predicate)
                    else:
                        #* Record the always-true relation as a prior (Eq if always true, else Ne).
                        if predicate_values.iloc[0] == 0:
                            predicate = f"Ne({lhs},{rhs})"
                        prior_rules.add(predicate)
                        #* Do not `continue`: (lhs > rhs) / (lhs < rhs) may still vary even if Eq is constant.
                
                #* Doesn't make sense to compare sequencing variables other than equality.
                if (vtype_lhs != VariableType.SEQUENCING
                    and domaintype_lhs != DomainType.CATEGORICAL ):
                        assert domaintype_rhs != DomainType.CATEGORICAL, \
                            "LHS and RHS must both be non-categorical for comparison predicates."
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
                            # continue
                        
                        #& Comparison predicates: A<B
                        predicate = f"({lhs}<{rhs})"
                        predicate_values = (self.examples[lhs]<self.examples[rhs]).astype(int)
                        if predicate_values.nunique() > 1:
                            self.examples[predicate] = predicate_values
                            predicates.add(predicate)
                        else:
                            if predicate_values.iloc[0] == 0:
                                predicate = f"({lhs}>={rhs})"
                            prior_rules.add(predicate)
                            # continue
        #^ End for j, lhs in enumerate(self.variables)

        if self.dataset == "abr":
            abr_preds, abr_priors = self._generate_abr_template_predicates()
            predicates |= abr_preds
            prior_rules |= abr_priors

        # prior_rules = set()
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
            if domaintype in (DomainType.INTEGER, DomainType.REAL):
                bounds = self.domains[varname].bounds
                assert bounds is not None, f"Variable {varname} has no bounds defined."
                prior_rules.add(f"({varname}>={bounds.lb})")
                prior_rules.add(f"({varname}<={bounds.ub})")
                
                # #& Add default predicate X>0 if the domain contains 0 (except for ack and seq which already have >1).
                # if (bounds.lb <= 0 <= bounds.ub 
                #     and bounds.lb != bounds.ub 
                #     and vtype != VariableType.SEQUENCING
                # ):
                #     #* Add X=0 as a default if the domain contains 0.
                #     predicate = f"Eq({varname},0)"
                #     predicate_values = (self.examples[varname]==0).astype(int)
                #     if predicate_values.nunique() > 1:
                #         predicates.add(predicate)
                #     else:
                #         if predicate_values.iloc[0] == 0:
                #             predicate = f"Not({predicate})"
                #         prior_rules.add(predicate)
                
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
        
        log.info(f"Duplicate predicates found: {len(predicates) - len(constraint_predicates)}")
        
        self.prior = {Constraint(sp.sympify(r)) for r in prior_rules
                        if r not in [sp.true, sp.false]}
        # #* Save predicates to file for inspection.
        # # pprint(prior_rules)
        Theory.save_constraints(constraint_predicates, f'predicates_{self.dataset}_{FLAGS.label}.pl')
        Theory.save_constraints(self.prior, f'prior_{self.dataset}_{FLAGS.label}.pl')
        # exit(0)
        
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
