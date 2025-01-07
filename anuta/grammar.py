from itertools import combinations
from typing import *
import sympy as sp
from dataclasses import dataclass
from enum import Enum, auto
from rich import print as pprint
import warnings
warnings.filterwarnings("ignore")

from anuta.utils import log, consecutive_combinations, clausify, true, false
from anuta.theory import Constraint


@dataclass
class Bounds:
    lb: float
    ub: float

class ConstantType(Enum):
    ASSIGNMENT = auto()
    SCALAR = auto()

@dataclass
class Constants(object):
    kind: ConstantType
    values: List[int]
    
    def __repr__(self) -> str:
        return f"{self.kind}, {len(self.values)} constants"

class DomainType(Enum):
    NUMERICAL = auto()
    CATEGORICAL = auto()

@dataclass
class Domain:
    kind: DomainType
    bounds: Bounds
    values: List[Enum]

class Operator(Enum):
    NOP = 0
    PLUS = auto()
    MAX = auto()
    
    
class Anuta(object):
    def __init__(self, variables: List[str], domains: Dict[str, Domain], 
                 constants: Dict[str, Constants]=None, 
                 prior_kb: List[Constraint]=[]):
        variables = sp.symbols(' '.join(variables))
        self.variables: Dict[str, sp.Symbol] = {v.name: v for v in variables}
        self.domains: Dict[str, Domain] = domains
        self.constants: Dict[str, Constants] = constants
        
        self.prior_kb = prior_kb
        self.initial_kb: List[Constraint] = []
        self.learned_kb: List[Constraint] = []
        
        self.prior: Set[Constraint] = set() #* Untested facts.
        self.kb: Set[Constraint] = set()
        
        self.candidates: List[Constraint] = [] #* List for ordered elimination.
        self.backlog: Set[Constraint] = set() #* Backlog candidates.
        self.num_candidates_proposed = 0
        self.num_candidates_rejected = 0
        self.search_arity = 1
        
        #* Caching intermediate results.
        self._arity2_cache: Set[Constraint] = set()
        self._literal_cache: Set[Constraint] = set()
    
    def rank_literal(self, literal: sp.Mul | sp.Symbol) -> Tuple[int, int]:
        if type(literal) != sp.Mul:
            assert type(literal) == sp.Symbol, f"{literal=} is not a monomial."
            return -1, -1
        
        scalar, var = literal.args
        assert type(var) == sp.Symbol, f"Unexpected {var=} in {literal}"
        assert type(scalar) == sp.Integer, f"Unexpected {scalar=} in {literal}"
        #TODO: Support floats.
        scalar = int(scalar)
        #! Assume all scalars in front of a var are specified constants.
        constants = self.constants.get(str(var))
        assert constants, f"Missing constants for {literal=}."
        assert constants.kind == ConstantType.SCALAR, (
            f"Non-scalar constant {constants} for {var}.")
        
        rank = constants.values.index(scalar)
        maxrank = len(constants.values) - 1
        return rank, maxrank
    
    def generalize_clause(self, clause: sp.Expr, init=False) -> Tuple[bool, sp.Expr]:
        """Try generalizing a numerical clause.

        :param clause: `sp.Expr` A numerical clause.
        :param init: `bool` If True, initialize the clause to the most general boundary.
        :return: `Tuple[bool, sp.Expr]` True if the clause is generalizable, 
                the generalized clause, otherwise False and the original clause.
        """
        if type(clause) not in [sp.GreaterThan, sp.StrictGreaterThan]:
            #* Not a bound (not generalizable).
            return False, clause

        assert len(clause.args) == 2, f"Unexpected Ge clause: {clause}"
        #* (S x) lhs ≥ (S x) lhs
        lhs, rhs = clause.args
        generalized = False
        new_clause = clause
        
        if type(lhs) == sp.Mul:
            _, var = lhs.args
            rank, maxrank = self.rank_literal(lhs)
            constants = self.constants.get(str(var))
            assert rank >= 0 and maxrank >= 0 and constants
            
            #* Try increasing the scalar of the LHS.
            if init:
                #* Initialize it to the most general.
                lhs = sp.Mul(sp.S(constants.values[-1]), var)
                generalized = True
                #? Should here be ≥ or strict >??? (the inversion should be ≤)
                new_clause = sp.Ge(lhs, rhs)
            elif rank+1 <= maxrank:
                lhs = sp.Mul(sp.S(constants.values[rank+1]), var)
                generalized = True
                new_clause = sp.Ge(lhs, rhs)
            else:
                log.info(f"Reached the maximum scalar for {lhs=}.")
                
        if not generalized and type(rhs) == sp.Mul:
            #& Only generalize RHS if the LHS isn't generalizable.
            _, var = rhs.args
            rank, _ = self.rank_literal(rhs)
            constants = self.constants.get(str(var))
            assert rank >= 0 and constants
            
            #* Try decreasing the scalar of the RHS.
            if init:
                #* Initialize it to the most general.
                lhs = sp.Mul(sp.S(constants.values[0]), var)
                generalized = True
                #? >= or >
                new_clause = sp.Ge(lhs, rhs)
            elif rank-1 >= 0:
                rhs = sp.Mul(sp.S(constants.values[rank-1]), var)
                generalized = True
                new_clause = sp.Ge(lhs, rhs)
            else:
                log.info(f"Already the most generalized: {clause=}")

        if generalized:
            if self.interval_filter(new_clause):
                log.info(f"Interval filtered {new_clause=}")
                self.num_candidates_rejected += 1
                generalized = False
                #* Don't add the interval-filtered clause.
                new_clause = None
            else:
                log.info(f"Generalized ({clause}) to ({new_clause}).")
                
        #* If both sides are NOT generalizable, return the original clause.
        return generalized, new_clause
    
    def specialize_clause(self, clause: sp.Expr, init: bool=False) -> Tuple[bool, sp.Expr]:
        """Try specializing a numerical clause.

        :param clause: `sp.Expr` A numerical clause.
        :param init: `bool` If True, initialize the clause to the most specific boundary.
        :return: `Tuple[bool, sp.Expr]` True if the clause is specializable, 
                the generalized clause, otherwise False and the original clause.
        """
        if type(clause) not in [sp.GreaterThan, sp.StrictGreaterThan]:
            #* Not a bound (not generalizable).
            return False, clause

        assert len(clause.args) == 2, f"Unexpected Ge clause: {clause}"
        #* (S x) lhs ≥ (S x) lhs
        lhs, rhs = clause.args
        specialized = False
        new_clause = clause
        
        if type(lhs) == sp.Mul:
            _, var = lhs.args
            rank, _ = self.rank_literal(lhs)
            constants = self.constants.get(str(var))
            assert rank >= 0 and constants
            
            #* Try decreasing the scalar of the LHS.
            if init:
                #* Start with the most specific boundary.
                lhs = sp.Mul(sp.S(constants.values[0]), var)
                specialized = True
                #? >= or >
                new_clause = sp.Ge(lhs, rhs)
            elif rank-1 >= 0:
                lhs = sp.Mul(sp.S(constants.values[rank-1]), var)
                specialized = True
                new_clause = sp.Ge(lhs, rhs)
            else:
                log.info(f"Reached the minimum scalar for {lhs=}.")
        
        if not specialized and type(rhs) == sp.Mul:
            _, var = rhs.args
            rank, maxrank = self.rank_literal(rhs)
            constants = self.constants.get(str(var))
            assert rank >= 0 and maxrank >= 0 and constants
            
            #* Try increasing the scalar of the RHS.
            if init:
                #* Start with the most specific boundary.
                rhs = sp.Mul(sp.S(constants.values[-1]), var)
                specialized = True
                new_clause = sp.Ge(lhs, rhs)
            elif rank+1 <= maxrank:
                rhs = sp.Mul(sp.S(constants.values[rank+1]), var)
                specialized = True
                new_clause = sp.Ge(lhs, rhs)
            else:
                log.info(f"Already the most specific: {clause=}.")
        
        if specialized:
            if self.interval_filter(new_clause):
                log.info(f"Interval filtered {new_clause=} ({init=})")
                self.num_candidates_rejected += 1
                specialized = False
                #* Don't add the interval-filtered clause.
                new_clause = None
            else:
                log.info(f"Specialized {clause} to ({new_clause}).")
                
        #* If both sides are NOT specializable, return the original clause.
        return specialized, new_clause
     
    def interative_refinement(self) -> Set[Constraint]:
        assert self.search_arity <= 3, f"{self.search_arity=} not supported."
        log.info(f"Interative refinement for {len(self.candidates)} arity-{self.search_arity} constraints.")
        
        #* Proposes new candidates by generalizing inconsistant (rejected) ones.
        refined_candidates: Set[Constraint] = set()
        
        if self.search_arity == 2:
            for candidate in self.candidates:
                #* Try generalizing the clause (A ≥ B but not A => B).
                generalized, new_expr = self.generalize_clause(candidate.expr)
                if generalized:
                    refined_candidates.add(Constraint(new_expr))
                elif new_expr is not None:
                    self.backlog.add(candidate)
            return refined_candidates
    
        for candidate in self.candidates:
            assert type(candidate.expr) == sp.Implies
            premise, conclusion = candidate.expr.args
            
            # if candidate == Constraint(sp.Implies(sp.Eq(self.variables['Proto'], 2), self.variables['Bytes']>=sp.Mul(self.variables['Packets'], 65535))):
            #     print(f"Refining {candidate=}")
            
            if type(premise) in [sp.And, sp.Or]:
                #^ (A & B & ...) => ...
                assert type(premise) == sp.And
                clauses = premise.args
                new_clauses = []
                specialized = False
                for clause in clauses:
                    #* Tweak one clause at a time.
                    if not specialized:
                        #* Specializing premise is generalizing the implication.
                        specialized, clause = self.specialize_clause(clause)
                    new_clauses.append(clause)
                
                if specialized:
                    new_candidate = Constraint(sp.And(*new_clauses) >> conclusion)
                    refined_candidates.add(new_candidate)
                else:
                    self.backlog.add(candidate)
            else:
                #^ A => ...
                specialized, new_premise = self.specialize_clause(premise)
                if specialized:
                    new_candidate = Constraint(new_premise >> conclusion)
                    refined_candidates.add(new_candidate)
                elif new_premise is not None:
                    self.backlog.add(candidate)
            
            if type(conclusion) in [sp.And, sp.Or]:
                #^ ... => (A | B | ...)
                assert type(conclusion) == sp.Or
                clauses = conclusion.args
                new_clauses = []
                generalized = False
                for clause in clauses:
                    if not generalized:
                        #* Generalizing conclusion is generalizing the implication.
                        generalized, clause = self.generalize_clause(clause)
                    new_clauses.append(clause)
                
                if generalized:
                    new_candidate = Constraint(premise >> sp.Or(*new_clauses))
                    refined_candidates.add(new_candidate)
                else:
                    self.backlog.add(candidate)
            else:
                #^ ... => A
                generalized, new_conclusion = self.generalize_clause(conclusion)
                if generalized:
                    new_candidate = Constraint(premise >> new_conclusion)
                    refined_candidates.add(new_candidate)
                elif new_conclusion is not None:
                    self.backlog.add(candidate)

        log.info(f"Created {len(refined_candidates)} refined constraints.")
        return refined_candidates
            
    def propose_new_candidates(self) -> None:
        # pprint("\tEntered propose_new_candidates()")
        #* Set for dedupe.
        new_candidates: Set[Constraint] = set()
        
        if self.search_arity == 1:
            new_candidates = self.propose_arity2_candidates()
            self.search_arity = 2
                    
        elif self.search_arity == 2:
            
            new_candidates = self.interative_refinement()
            if new_candidates:
                #NOTE: A bit ugly but don't want to nest if-else again.
                self.num_candidates_proposed += len(new_candidates)
                self.candidates = list(new_candidates)
                log.info(f"Refined {len(self.candidates)} arity-{self.search_arity} constraints. First a few:")
                pprint(self.candidates[:5])
                return
            else:
                #* If no candidates can be generalized, propose higher-arity constraints from the backlog.
                self.candidates = list(self.backlog)
                self.backlog = set()
            
            #& Only propose new candidates of higher arity if no existing candidates can be generalized.
            log.info(f"Proposing arity-3 constraints from {len(self.candidates)} rejected arity-2 constraints.")
            self.search_arity = 3
            new_candidates: Set[Constraint] = set()
            
            #* Combining two rejected arity-2 implications to form a more general arity-3 constraint.
            rejected_bounds: List[sp.Expr] = []
            for candidate1 in self.candidates:
                if type(candidate1.expr) != sp.Implies:
                    assert type(candidate1.expr) in [sp.GreaterThan, sp.StrictGreaterThan]
                    rejected_bounds.append(candidate1.expr)
                    continue
                premise1, conclusion1 = candidate1.expr.args
                premise1, conclusion1 = Constraint(premise1), Constraint(conclusion1)
                
                for candidate2 in self.candidates:
                    if type(candidate2.expr) != sp.Implies: continue
                    if candidate1 == candidate2: continue
                    
                    composite = Constraint(true)
                    
                    premise2, conclusion2 = candidate2.expr.args
                    premise2 = Constraint(premise2)
                    conclusion2 = Constraint(conclusion2)
                    # if sp.Equivalent(desugar(premise1), desugar(premise2)):
                    #^ Equivalence check is expensive (e.g., (A=>B)+(~A=>B) == (F=>B) == True).
                    if premise1 == premise2:
                        #* Specifics: (A=>B)+(A=>C) -> More general: (A => (B | C))
                        composite = Constraint(premise1.expr >> (conclusion1.expr | conclusion2.expr))
                    # elif sp.Equivalent(desugar(conclusion1), desugar(conclusion2)):
                    elif conclusion1 == conclusion2:
                        #* Specifics: (A=>B)+(C=>B) -> More general: ((A & C) => B)
                        composite = Constraint((premise1.expr & premise2.expr) >> conclusion1.expr)
                        
                    if composite.expr in [true, false]:
                        #^ Filter trivial constraints.
                        self.num_candidates_rejected += 1
                        self.num_candidates_proposed += 1
                    else:
                        new_candidates.add(composite)
                    
                    print(f"Proposed # of arity-{self.search_arity} (in)equality implications:\t{len(new_candidates)}", end='\r')
            log.info(f"Proposed {len(new_candidates)} arity-{self.search_arity} (in)equality implications.")
            cur_ncandidates = len(new_candidates)
            
            #* Try generalizing the rejected bounds to implication forms.   
            log.info(f"Proposing implications with bounds ...")
            #! Should not limit to only the rejected bounds:
            #!  Generalized (Bytes >= 64*Packets) to (Bytes >= 42*Packets), which is accepted, 
            #!  but (Bytes >= 64*Packets) should still be considered, e.g., (Proto=ICMP) => (Bytes >= 64*Packets).
            # specialized_rej_bounds = [self.specialize_clause(bound, init=True)[1] for bound in rejected_bounds]
            
            #NOTE: Arity 2 bounds are the most specific.
            log.info(f"{rejected_bounds=}")
            arity2_bounds = [bound.expr for bound in self._arity2_cache if type(bound.expr) in [sp.GreaterThan, sp.StrictGreaterThan]]
            equality_literals = [literal.expr for literal in self._literal_cache if type(literal.expr) in [sp.Equality, sp.Unequality]]  
            # print(f"{len(arity2_bounds)=}")
            # print(f"{len(equality_literals)=}")
            for i, rejected_bound in enumerate(rejected_bounds):
                for literal in equality_literals:
                    #* The rejected bounds are the most general after interative refinement.
                    #* (rejeceted general bound: sX≥sY) => (A≠a)
                    composite = Constraint( rejected_bound >> literal )
                    new_candidates.add(composite)
                
                #TODO: Add the following to arity-4 constraints.
                # for j, specialized_bound in enumerate(specialized_rej_bounds):
                #     if i == j: 
                #         #^ Prevent redundancy: general(A ≥ 5B) => specialized(A ≥ 3B)
                #         self.num_candidates_rejected += 1
                #         self.num_candidates_proposed += 1
                #         continue
                #     #* (rejeceted general bound: sX≥sY) => (specialized rejeceted bound: sX≥sY)
                #     composite = Constraint( rejected_bound >> specialized_bound )
                #     new_candidates.add(composite)
            for literal in equality_literals:
                for specialized_bound in arity2_bounds:
                    if Constraint(specialized_bound) in self.kb: 
                        #^ Prevent redundancy: If `Bytes >= 42*Packets` is learned, 
                        #^  don't learn any `... => (Bytes >= 42*Packets)`.
                        continue
                    #* (A≠a) => (specialized rejeceted bound: sX≥sY)
                    composite = Constraint( literal >> specialized_bound )
                    new_candidates.add(composite)
                print(f"Proposed # of arity-{self.search_arity} implications w/ bounds:\t{len(new_candidates)}", end='\r')
            log.info(f"Proposed {len(new_candidates)-cur_ncandidates} arity-{self.search_arity} implications w/ bounds")
            
        elif self.search_arity == 3:            
            #* Propose new candidates by generalizing inconsistant (rejected) ones.
            #* (currently not proposing candidates of arity>3).
            new_candidates = self.interative_refinement()
            if not new_candidates:
                #* Increase the arity to end the search.
                self.search_arity += 1
                #TODO: Continue searching for higher arity constraints.
                # self.candidates = list(self.backlog)

        #> End of candidate generation
        if new_candidates:
            self.num_candidates_proposed += len(new_candidates)
            self.candidates = list(new_candidates)
            log.info(f"Proposed {len(self.candidates)} arity-{self.search_arity} constraints. First a few:")
            pprint(self.candidates[:5])
        else:
            self.candidates = []
        return

    def propose_arity2_candidates(self) -> Set[Constraint]:
        if self._arity2_cache:
            return self._arity2_cache
        #^ Generate the shortest (most specific, arity-1) constraints.
        literals: List[Constraint] = list(self.generate_literals())
        self._literal_cache = set(literals)
        new_candidates: Set[Constraint] = set()
        
        for lhs in literals:
            #& Connectives: = and ≠
            if type(lhs.expr) in [sp.Equality, sp.Unequality]:
                for conclusion in literals:
                    if type(conclusion.expr) not in [sp.Equality, sp.Unequality]: continue
                    #* X=>X is trivial.
                    if lhs == conclusion: continue
                    #* (X=10) => (X=20) is trivial.
                    if lhs.expr.args[0] == conclusion.expr.args[0]: continue
                    
                    candidate = Constraint(sp.Implies(lhs.expr, conclusion.expr))
                    # if candidate == Constraint(sp.Implies(sp.Eq(self.variables['DstIpAddr'], 3), sp.Ne(self.variables['SrcPt'], 8080))):
                    #     print(f"Generated {candidate=}")
                    #^ A => B
                    assert type(candidate.expr) == sp.Implies and not candidate.expr in [true, false], (
                        f"{candidate} is not an implication or is trivial.")
                    new_candidates.add(candidate)
            #& Connectives: ×, identity
            elif type(lhs.expr) in [sp.Mul, sp.Symbol]:
                for rhs in literals:
                    if type(rhs.expr) not in [sp.Mul, sp.Symbol]: continue
                    if lhs == rhs: continue
                    if type(lhs.expr) != sp.Symbol and type(rhs.expr) != sp.Symbol:
                        #* (10X) => (20X) is trivial.
                        if lhs.expr.args[1] == rhs.expr.args[1]: 
                            self.num_candidates_rejected += 1
                            self.num_candidates_proposed += 1
                            continue
                    
                    #& Start with the most specific boudary constraints.
                    #* I.e., lhs_max >= rhs_max
                    #* allowing both sides to have identity vars (rank -1).
                    lhs_rank, lhs_maxrank = self.rank_literal(lhs.expr)
                    rhs_rank, rhs_maxrank = self.rank_literal(rhs.expr)
                    if lhs_rank in [0, -1] and rhs_rank in [rhs_maxrank, -1]:
                        candidate = Constraint(lhs.expr >= rhs.expr)
                        #^ sA ≥ sB
                        #& Specific
                        assert not candidate.expr in [true, false], (f"{candidate} is trivial.")
                        if self.interval_filter(candidate.expr):
                            self.num_candidates_rejected += 1
                            self.num_candidates_proposed += 1
                        else:
                            new_candidates.add(candidate)
            else:
                raise NotImplementedError(f"Unsupported literal: {lhs}")
        
        self._arity2_cache = new_candidates.copy()
        return new_candidates
        
    
    def generate_literals(self) -> Generator[Constraint, None, None]:
        for name, var in self.variables.items():
            if name in self.constants:
                #& If the var has associated constants, don't enumerate its domain (often too large).
                match self.constants[name].kind:
                    case ConstantType.ASSIGNMENT:
                        eq_priors: List[Constraint] = []
                        ne_priors: List[Constraint] = []
                        for const in self.constants[name].values:
                            self.num_candidates_proposed += 2
                            #* Var == const
                            constraint = Constraint(sp.Eq(var, sp.S(const)))
                            assert not constraint.expr in [true, false], ("{constraint} is trivial.")
                            #* Var != const
                            neg_constraint = Constraint(sp.Ne(var, sp.S(const)))
                            assert not neg_constraint.expr in [true, false], (f"{neg_constraint} is trivial.")
                            
                            #& First-level elimination through domain check.
                            if const in self.domains[name].values:
                                #! Do NOT learn possible assignments as facts (too strong).
                                eq_priors.append(constraint.expr)
                                yield constraint
                                #* Still need the negation literals.
                                yield neg_constraint
                            else:
                                log.warning(f"Constant {const} not in the domain of {name}.")
                                #* Learn negation assignment as a fact, i.e., this constant is not in the domain.
                                ne_priors.append(neg_constraint.expr)
                                self.num_candidates_rejected += 1
                        # if eq_priors:
                        #     #* Add the prior knowledge (X=1 | X=2 | ...)
                        #     self.prior.add(Constraint(sp.Or(*eq_priors)))
                        if ne_priors:
                            #* Add the prior knowledge (X!=1 & X!=2 & ...)
                            #! AND for values not in the domain.
                            self.prior.add(Constraint(sp.And(*ne_priors)))
                    case ConstantType.SCALAR:
                        assert self.domains[name].kind == DomainType.NUMERICAL, (
                            f"Scalar constant {self.constants[name]} must be associated with a numerical var.")
                        
                        self.constants[name].values.sort() #* Just in case.
                        #& Version-spacing: Start with the most specific (i.e., extreme) constraints 
                        #* then generalize upon violation.
                        lb_constraint = Constraint(sp.Mul(var, sp.S(self.constants[name].values[0])))
                        yield lb_constraint
                        ub_constraint = Constraint(sp.Mul(var, sp.S(self.constants[name].values[-1])))
                        yield ub_constraint
                    case _:
                        raise NotImplementedError(f"Unsupported constant: {self.constants[name]}")
            else:
                #& Variables w/o associated constants.
                domain = self.domains[name]
                if domain.kind == DomainType.CATEGORICAL:
                    eq_priors: List[Constraint] = []
                    #* Enumerate the domain of a categorical var.
                    for value in domain.values:
                        self.num_candidates_proposed += 2
                        #* Var == value
                        constraint = Constraint(sp.Eq(var, sp.S(value)))
                        assert not constraint.expr in [true, false], (f"{constraint} is trivial.")
                        #* Var != value
                        neg_constraint = Constraint(sp.Ne(var, sp.S(value)))
                        assert not neg_constraint.expr in [true, false], (f"{neg_constraint} is trivial.")
                        
                        eq_priors.append(constraint.expr)
                        yield constraint
                        yield neg_constraint
                    # if eq_priors:
                    #     #* Add the prior knowledge (X=1 | X=2 | ...)
                    #     self.prior.add(Constraint(sp.Or(*eq_priors)))
                elif domain.kind == DomainType.NUMERICAL: 
                    #* For numerical vars w/o associated constants, use the unary identity (NOP).
                    identity = Constraint(var)
                    # identity.rank = -1
                    # identity.maxrank = -1
                    yield identity
                    
        log.info(f"Prior size: {len(self.prior)}")
        return
    
    def generate_expressions(self) -> Generator[sp.Expr, None, None]:
        """Generate expressions of a single atom (arity-1).
        """
        for name, var in self.variables.items():
            if name in self.constants:
                #* If the var has associated constants, don't enumerate its domain (often too large).
                match self.constants[name].kind:
                    case ConstantType.ASSIGNMENT:
                        for const in self.constants[name].values:
                            #* Var == const
                            yield sp.Eq(var, sp.S(const))
                            #* Var != const
                            yield sp.Ne(var, sp.S(const))
                    case ConstantType.SCALAR:
                        for const in self.constants[name].values:
                            #* Var x const
                            yield sp.Mul(var, sp.S(const))
                    case _:
                        raise NotImplementedError(f"Unsupported constant: {self.constants[name]}")
            else:
                domain = self.domains[name]
                if domain.kind == DomainType.CATEGORICAL:
                    #* Enumerate the domain of a categorical var.
                    for value in domain.values:
                        #* Var == value
                        yield sp.Eq(var, sp.S(value))
                        #* Var != value
                        yield sp.Ne(var, sp.S(value))
                if domain.kind == DomainType.NUMERICAL: 
                    #* Omit numerical vars w/o associated constants for now.
                    continue
        
    def generate_arity2_constraints(self) -> Generator[sp.Expr, None, None]:
        #* Arity-1 constraints: self.prior_kb
        
        #* Arity-2 constraints:
        expressions = list(self.generate_expressions())
        for i, expr_lhs in enumerate(expressions):
            if type(expr_lhs) in [sp.Equality, sp.Unequality]:
                #* Filter self-comparison and ordering.
                for expr_rhs in expressions[i+1:]:
                    if type(expr_rhs) in [sp.Equality, sp.Unequality]:
                        
                        #& Avoid (Var==const1) AND (Var==const2)
                        if expr_lhs.args[0] != expr_rhs.args[0]:
                            #* (Var==const1) AND (Var==const2)
                            #* ~(Var==const1) AND (Var==const2)
                            #* (Var==const1) AND ~(Var==const2)
                            #* ~(Var==const1) AND ~(Var==const2)
                            yield sp.And(expr_lhs, expr_rhs)
                            #^ AND constraints are too restrictive and will be all eliminated.
                            #! But, we need them for arity-3 constraints.
                        
                        #! Do NOT avoid (Var==const1) OR (Var==const2)
                        #* (Var==const1) OR (Var==const2)
                        #* ~(Var==const1) OR (Var==const2)
                        #* (Var==const1) OR ~(Var==const2)
                        #* ~(Var==const1) OR ~(Var==const2)
                        yield sp.Or(expr_lhs, expr_rhs)
                
            elif type(expr_lhs) == sp.Mul:
                for name, var in self.variables.items():
                    if self.domains[name].kind == DomainType.NUMERICAL:
                        #* (Var x const1) >= Var
                        yield expr_lhs >= var
                        #* (Var x const1) <= Var
                        yield expr_lhs <= var
    
    def interval_filter(self, clause: sp.Expr) -> bool:
        """Filter constraints using interval arithmetic.

        :param clause: Constraint to filter.
        :return: False if the constraint is valid, True if the constraint is invalid.
        """
        if type(clause) not in [sp.GreaterThan, sp.StrictGreaterThan]:
            #* Not a bound.
            return False
        
        lhs, rhs = clause.args
        if type(lhs) == sp.Mul:
            scalar1, var1 = lhs.args
            scalar1 = int(scalar1)
        else:
            scalar1 = 1
            var1 = lhs
            
        if type(rhs) == sp.Mul:
            scalar2, var2 = rhs.args
            scalar2 = int(scalar2)
        else:
            scalar2 = 1
            var2 = rhs
        
        match type(clause):
            case sp.GreaterThan:
                #* Amax < Bmin
                return scalar1*self.domains[str(var1)].bounds.ub < scalar2*self.domains[str(var2)].bounds.lb
            case sp.StrictGreaterThan:
                return scalar1*self.domains[str(var1)].bounds.ub <= scalar2*self.domains[str(var2)].bounds.lb
            # case sp.LessThan:
            #     #* s x Amin >= Bmax
            #     return scalar*self.domains[str(var)].bounds.lb > self.domains[str(rhs)].bounds.ub
            case _:
                raise NotImplementedError(f"Unsupported constraint type: {type(clause)}") 
    
    def generate_arity3_constraints(self, arity2_constraints) -> Generator[sp.Expr, None, None]:
        for expression in self.generate_expressions():
            #* Multiplication expressions can't be premises.
            if type(expression) == sp.Mul: continue
            
            for constraint in arity2_constraints:
                #* Dedupe.
                lhs, rhs = constraint.args
                if (expression == lhs or sp.Not(expression) == lhs or 
                    expression == rhs or sp.Not(expression) == rhs): continue
                #! Omit AND constraints for now.
                # yield sp.And(expression, constraint)
                yield sp.Or(expression, constraint)
                # yield sp.Or(sp.Not(expression), constraint)
                yield sp.Or(expression, sp.Not(constraint))
                # yield sp.Or(sp.Not(expression), sp.Not(constraint))
        #^ No ≤ or ≥ constraints for now.
    
    def populate_kb(self) -> None:
        added = set()
        num_duplicates = 0
        num_constraints = 0
        num_trivial = 0
        interval_filtered = 0
        arity2_constraints = []
        for constraint in self.generate_arity2_constraints():
            if isinstance(constraint, sp.logic.boolalg.BooleanTrue):
                num_trivial += 1
            else:
                if self.interval_filter(constraint):
                    log.info(f"Interval-filtered: {constraint}")
                    interval_filtered += 1
                    continue
                
                #* Dedupe: sp.Or(sp.Eq(x, 10), sp.Eq(y, 303)) == sp.Or(sp.Eq(y, 303), sp.Eq(x, 10))
                if constraint in added:
                    num_duplicates += 1
                    continue
                added.add(constraint)
                arity2_constraints.append(constraint)
                #^ Collect arity-2 constraints for creating arity-3 constraints.
                #& Avoid arity-2 AND constraints for now.
                if type(constraint) != sp.And:
                    self.candidates.append(constraint)
                
                num_constraints += 1
                if num_constraints % 10_000 == 0:
                    log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Trivial constraints: {num_trivial}")
        log.info(f"Arity-2 constraints: {num_constraints}")
        log.info(f"Interval-filtered constraints: {interval_filtered}")
        log.info(f"Duplicate arity-2 constraints: {num_duplicates}")
        
        old_num_constraints = num_constraints
        old_num_duplicates = num_duplicates
        for constraint in self.generate_arity3_constraints(arity2_constraints):
            if constraint in added:
                #* Dedupe for cases like (A | B | C == C | B | A)
                num_duplicates += 1
                continue
            added.add(constraint)
            self.candidates.append(constraint)
            num_constraints += 1
            if num_constraints % 10_000 == 0:
                log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Arity-3 constraints: {num_constraints-old_num_constraints}")
        log.info(f"Duplicate arity-3 constraints: {num_duplicates-old_num_duplicates}")
        
        log.info(f"Populated KB with {len(self.candidates)} constraints.\n")
        
        #! Too expansive, and the reduction is little (26/149720)
        # log.info(f"Filtering redundant constraints ...")
        # # Nonnegative integers
        # assumptions = [v >= 0 for v in self.variables.values()]
        # cnf = sp.And(*(self.initial_kb + assumptions))
        # simplified_logic = sp.simplify_logic(cnf)
        # #! Too expansive
        # # simplified = simplified_logic.simplify()
        # reduced_constraints = list(simplified_logic.args) \
        #     if isinstance(simplified_logic, sp.And) else [simplified_logic]
        # num_redundant = len(self.initial_kb) - len(reduced_constraints)
        # #* Update KB
        # self.initial_kb = reduced_constraints
        # log.info(f"Removed {num_redundant} redundant constraints.\nFinal KB size: {len(self.initial_kb)}")
        return


class AnutaMilli(object):
    def __init__(self, variables: List[str], bounds: Dict[str, Bounds], constants: Dict[str, int]=None, operators: List[int]=None):
        variables = sp.symbols(' '.join(variables), integer=True, nonnegative=True)
        self.variables = {v.name: v for v in variables}
        self.constants = constants
        self.bounds = bounds
        self.operators = list(Operator) if not operators else \
            [Operator(o) for o in operators]
        
        
        self.varlabel = 'aggregate'
        self.static_vars = [v for v in self.variables.values() if self.varlabel in v.name and 
                            'host' not in v.name and 'stride' not in v.name and 'window' not in v.name and 'canary' not in v.name]
        self.dynamic_vars = [v for v in self.variables.values() if self.varlabel not in v.name and 'canary' not in v.name]
        self.canary_vars = [v for v in self.variables.values() if 'canary' in v.name]
        self.kb = []
    
    def generate_expressions(self) -> Generator[Tuple[Operator, sp.Expr], None, None]:
        #& Limit to homogeneous expressions only (i.e., no mixed operators).
        if Operator.NOP in self.operators:
            for v in self.dynamic_vars:
                yield (Operator.NOP, v)
                
        if Operator.PLUS in self.operators:
            #! Too expansive:
            # for k in range(1, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sum(combo)
            #! Too expansive:
            # for k in range(2, len(measurement_vars)+1):
            #     for combo in combinations(measurement_vars.values(), k):
            #         yield sum(combo)
            
            #* Isolate the interactions to within static vars and dynamic vars.
            #! Omit interactions within static vars.
            # for k in range(2, len(self.static_vars)+1):
            #     #* All combinations of sampled vars
            #     for combo in combinations(self.static_vars, k):
            #         yield sum(combo)
            #& Limit to consecutive combinations only.
            for combo in consecutive_combinations(self.dynamic_vars):
                yield (Operator.PLUS, sum(combo))
                    
        if Operator.MAX in self.operators:
            #! Too expansive:
            # #* Max over ≥2 args.
            # for k in range(2, len(self.variables)+1):
            #     for combo in combinations(self.variables.values(), k):
            #         yield sp.Max(*combo)
            #& Limit to consecutive combinations only.
            yield (Operator.MAX, sp.Max(*self.dynamic_vars))
    
    def generate_constraints(self):
        for op, expr in self.generate_expressions():
            #& Restrict the bounds to constants and static vars only.
            bounds = self.static_vars + [sp.symbols(name) for name in self.constants]
            for b in bounds:
                match op:
                    case _:
                        yield expr >= b
                        yield expr <= b
            #* Add canary max constraint
            if op == Operator.MAX:
                canary_max = [v for v in self.canary_vars if 'max' in v.name][0]
                print(type(canary_max))
                yield expr >= canary_max
                yield expr <= canary_max
            
            #* Implications
            triggers = self.static_vars + self.dynamic_vars
            for trigger in triggers:
                for name in self.constants:
                    const = sp.symbols(name)
                    #* Only consider premises of the form: (var > 0)
                    #* Only consider conclusions of the form: (expr {} const)
                    yield (trigger > 0) >> (expr >= const)
                    yield (trigger > 0) >> (expr <= const)
        #* Add canary implications
        canary_premise = [v for v in self.canary_vars if 'premise' in v.name][0]
        canary_conclusion = [v for v in self.canary_vars if 'conclusion' in v.name][0]
        yield (canary_premise > 0) >> (canary_conclusion >= sp.symbols('burst_threshold'))
            

    def interval_filter(self, constraint: sp.Expr) -> bool:
        #* Interval arithmetic
        args = constraint.args
        match type(constraint):
            case sp.GreaterThan:
                operation = args[0]
                operands = operation.args
                comparator = args[1]
                assert not comparator.args, f"Comparator is not a singleton: {comparator}"
                
                rhs_bounds = self.bounds[comparator.name]
                if isinstance(operation, sp.Add):
                    lhs_ub = 0
                    for operand in operands:
                        lhs_ub += self.bounds[operand.name].ub
                    return lhs_ub < rhs_bounds.lb
                elif isinstance(operation, sp.Max):
                    lhs_ubs = [self.bounds[operand.name].ub for operand in operands]
                    return max(lhs_ubs) < rhs_bounds.lb
            case sp.LessThan:
                operation = args[0]
                operands = operation.args
                comparator = args[1]
                assert not comparator.args, f"Comparator is not a singleton: {comparator}"
                
                rhs_bounds = self.bounds[comparator.name]
                if isinstance(operation, sp.Add):
                    lhs_lb = 0
                    for operand in operands:
                        lhs_lb += self.bounds[operand.name].lb
                    return lhs_lb > rhs_bounds.ub
                elif isinstance(operation, sp.Max):
                    # #! Need to double-check this
                    # lhs_ubs = [self.bounds[operand.name].ub for operand in operands]
                    # return min(lhs_ubs) > rhs_bounds.ub
                    lhs_lbs = [self.bounds[operand.name].lb for operand in operands]
                    return max(lhs_lbs) > rhs_bounds.ub
            case sp.Implies:
                #* Assume the premise is always true, and check if the conclusion is valid.
                conclusion = args[1]
                return self.interval_filter(conclusion)
            case _:
                raise NotImplementedError(f"Unsupported constraint type: {type(constraint)}")
    
    def populate_kb(self):
        num_trivial = 0
        num_constraints = 0
        interval_filtered = 0
        for constraint in self.generate_constraints():
            # log.info(type(constraint))
            if isinstance(constraint, sp.logic.boolalg.BooleanTrue):
                num_trivial += 1
            else:
                if self.interval_filter(constraint):
                    interval_filtered += 1
                    continue
                self.kb.append(constraint)
                num_constraints += 1
                if num_constraints % 10_000 == 0:
                    log.info(f"Generated {num_constraints} constraints.")
        log.info(f"Skipped {num_trivial} trivial constraints.")
        log.info(f"Interval-filtered {interval_filtered} constraints.")
        log.info(f"Populated KB with {len(self.kb)} constraints.\n")
        
        #! Too expansive, and the reduction is little (26/149720)
        # log.info(f"Filtering redundant constraints ...")
        # # Nonnegative integers
        # assumptions = [v >= 0 for v in self.variables.values()]
        # cnf = sp.And(*(self.kb + assumptions))
        # simplified_logic = sp.simplify_logic(cnf)
        # #! Too expansive
        # # simplified = simplified_logic.simplify()
        # reduced_constraints = list(simplified_logic.args) \
        #     if isinstance(simplified_logic, sp.And) else [simplified_logic]
        # num_redundant = len(self.kb) - len(reduced_constraints)
        # #* Update KB
        # self.kb = reduced_constraints
        # log.info(f"Removed {num_redundant} redundant constraints.\nFinal KB size: {len(self.kb)}")