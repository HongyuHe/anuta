import logging
from typing import *
import sympy as sp
import pandas as pd
from collections import defaultdict
import z3
from functools import reduce

from anuta.known import *

true = sp.logic.true
false = sp.logic.false

log = logging.getLogger("anuta")
handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(name)s @ %(asctime)s] %(levelname)-8s | %(message)s", 
    datefmt="%H:%M:%S"  # Format to show only hour, minute, and second
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)


#* Mapping Sympy to Z3 operators:
def z3eq(a, b): return a == b
def z3ne(a, b): return a != b
def z3and(*args): return z3.And(*args)
def z3or(*args): return z3.Or(*args)
def z3implies(a, b): return z3.Implies(a, b)
def z3equiv(a, b): return a == b
def z3max(*args):
    return reduce(lambda a, b: z3.If(a >= b, a, b), args)
def z3mod(a, b):
    # assert z3.is_int(a) and z3.is_int(b), \
    #     f"Z3 mode operator supports only integer types, got {type(a)} and {type(b)}"
    return a % b
z3evalmap = {'Eq': z3eq, 'Ne': z3ne, 'And': z3and, 'Or': z3or, 
             'Implies': z3implies, 'Equivalent': z3equiv, 'Max': z3max, 'Mod': z3mod}

for varname in cidds_categoricals + cidds_ints:
    z3evalmap[varname] = z3.Int(varname)
for varname in cidds_floats:
    z3evalmap[varname] = z3.Real(varname)
for varname in metadc_ints:
    z3evalmap[varname] = z3.Int(varname)
#TODO: Improve this ugly piece
for varname in ['FrameLen_1','IpLen_1','IpHdrLen_1','IpTtl_1','IpProto_1','IpSrc_1','IpDst_1','TcpSrcport_1','TcpDstport_1','TcpHdrLen_1','TcpLen_1','TcpFlags_1','TcpSeq_1','TcpAck_1','TcpUrgentPointer_1','TcpWindowSizeValue_1','TcpWindowSizeScalefactor_1','TcpWindowSize_1','Tsval_1','Tsecr_1','FrameLen_2','IpLen_2','IpHdrLen_2','IpTtl_2','IpProto_2','IpSrc_2','IpDst_2','TcpSrcport_2','TcpDstport_2','TcpHdrLen_2','TcpLen_2','TcpFlags_2','TcpSeq_2','TcpAck_2','TcpUrgentPointer_2','TcpWindowSizeValue_2','TcpWindowSizeScalefactor_2','TcpWindowSize_2','Tsval_2','Tsecr_2','FrameLen_3','IpLen_3','IpHdrLen_3','IpTtl_3','IpProto_3','IpSrc_3','IpDst_3','TcpSrcport_3','TcpDstport_3','TcpHdrLen_3','TcpLen_3','TcpFlags_3','TcpSeq_3','TcpAck_3','TcpUrgentPointer_3','TcpWindowSizeValue_3','TcpWindowSizeScalefactor_3','TcpWindowSize_3','Tsval_3','Tsecr_3']:
    z3evalmap[varname] = z3.Int(varname)  #* For Netflix dataset, all variables are integers
#TODO: Add vars from other datasets


def normalize_pcap_5tuple(row):
    # Sort IP addresses and port numbers to normalize direction
    ip_pair = sorted([row["ip_src"], row["ip_dst"]])
    port_pair = sorted([row["tcp_srcport"], row["tcp_dstport"]])
    return pd.Series({
        "flow_ip_1": ip_pair[0],
        "flow_ip_2": ip_pair[1],
        "flow_port_1": port_pair[0],
        "flow_port_2": port_pair[1],
        "flow_proto": row["ip_proto"]
    })

def transform_consequent(expression):
    """
    Transform an implication so that terms in the consequent (right-hand side)
    are grouped by variable:
        - Same variable -> OR relationship
        - Different variables -> AND relationship
    """
    if not isinstance(expression, sp.Implies):
        raise ValueError("The input must be an Implies expression.")
    
    antecedent = expression.args[0]  # Left-hand side
    consequent = expression.args[1]  # Right-hand side

    # Step 1: Collect terms in the consequent
    terms_by_variable = defaultdict(list)

    def collect_terms(expr):
        """Recursively collect terms and group by variable."""
        if isinstance(expr, (sp.And, sp.Or)):  # If it's a logical combination
            for arg in expr.args:
                collect_terms(arg)
        else:  # If it's an individual equality or inequality
            # assert len(expr.args)==2, f"Unexpected form: {expr}, {expression=}"
            if not isinstance(expr, sp.Symbol):
                variable = expr.lhs  # Extract variable (e.g., DstIpAddr in Eq(DstIpAddr, 1))
                terms_by_variable[variable].append(expr)

    collect_terms(consequent)

    # Step 2: Reconstruct the consequent with OR-grouped terms
    new_consequent_terms = []
    for var, terms in terms_by_variable.items():
        if len(terms) > 1:
            new_consequent_terms.append(sp.Or(*terms))  # OR-grouped if same variable
        else:
            new_consequent_terms.append(terms[0])  # Single term remains unchanged

    new_consequent = sp.And(*new_consequent_terms) if len(new_consequent_terms) > 1 else new_consequent_terms[0]

    # Step 3: Construct the new implication
    return sp.Implies(antecedent, new_consequent)

def generate_sliding_windows(df: pd.DataFrame, stride: int, window: int) -> pd.DataFrame:
    #* Collect rows for the transformed dataframe
    rows = []
    for i in range(0, len(df) - window + 1, stride):
        #* Concatenate the window of rows into a single flattened list
        flattened_row = df.iloc[i:i + window].values.flatten().tolist()
        rows.append(flattened_row)
    
    #* Generate new column names
    columns = [
        f"{col}_{j+1}" for j in range(window) for col in df.columns
    ]
    
    #* Create the new dataframe
    return pd.DataFrame(rows, columns=columns)

def rename_pcap(columns):
    columns = list(columns)
    names = {}
    for col in columns:
        fields = col.split('.')
        if len(fields) < 3:
            names[col] = '_'.join(fields)
        else:
            names[col] = fields[-1]
    return names

def parse_tcp_flags(bitmask) -> str:
    """
    Convert a numeric TCP flags bitmask to a list of corresponding flag names.

    :param bitmask: Numeric bitmask of TCP flags (integer).
    :return: List of flag names joined by hyphens.
    """
    if isinstance(bitmask, str):
        bitmask = int(bitmask, base=16)
    if bitmask < 0: return ""
    
    #* Define TCP flags and their corresponding bit positions
    flags = [
        (0x01, "FIN"),  # 0b00000001
        (0x02, "SYN"),  # 0b00000010
        (0x04, "RST"),  # 0b00000100
        (0x08, "PSH"),  # 0b00001000
        (0x10, "ACK"),  # 0b00010000
        (0x20, "URG"),  # 0b00100000
        (0x40, "ECE"),  # 0b01000000
        (0x80, "CWR"),  # 0b10000000
    ]
    
    #* Extract flags from the bitmask
    result = sorted([name for bit, name in flags if bitmask & bit])
    return '-'.join(result)

def is_purely_or(expr):
    from sympy.logic.boolalg import Or
    """
    Check if a SymPy formula is purely made of `Or` logic (disjunctions of comparison operations).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is purely made of `Or` logic with `Eq` or `Ne` comparisons, False otherwise.
    """
    # Check if the expression is a comparison operation (Eq or Ne)
    def is_comparison(sub):
        return isinstance(sub, (sp.Eq, sp.Ne))

    # Main logic: Traverse the tree to ensure it's purely Or logic
    if isinstance(expr, Or):  # Top-level Or
        return all(is_comparison(arg) or is_purely_or(arg) for arg in expr.args)
    return is_comparison(expr)  # Single comparison is valid

def is_pure_dnf(expr):
    from sympy.logic.boolalg import Or, And, Not
    """
    Check if a SymPy formula is purely made of sp.Or logic (DNF).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is in DNF, False otherwise.
    """

    # Check if a sub-expression is a valid DNF clause
    def is_dnf_clause(sub):
        # A DNF clause must be a single variable, its negation, or an And operation
        if isinstance(sub, sp.Symbol):
            return True
        elif isinstance(sub, Not) and isinstance(sub.args[0], sp.Symbol):
            return True
        elif isinstance(sub, And):
            # All terms in the And must be symbols or negated symbols
            return all(
                isinstance(arg, sp.Symbol) or 
                (isinstance(arg, Not) and isinstance(arg.args[0], sp.Symbol))
                for arg in sub.args
            )
        return False

    # Main logic: Check if the expression is an Or of DNF clauses
    if isinstance(expr, Or):
        # All arguments of the Or must be valid DNF clauses
        return all(is_dnf_clause(arg) for arg in expr.args)
    elif is_dnf_clause(expr):  # A single clause can itself be valid DNF
        return True
    return False

def consecutive_combinations(lst):
    ccombo = []
    n = len(lst)
    
    #* Start with combinations of size 2 up to size n
    for size in range(2, n + 1):  
        #* Ensure that the combination is consecutive
        for start in range(n - size + 1):  
            ccombo.append(lst[start: start+size])
    
    return ccombo

def to_big_camelcase(string: str, sep=' ') -> str:
    words = string.split(sep)
    return ''.join(word.capitalize() for word in words) \
        if len(words) > 1 else string.capitalize()

def save_constraints(constraints: List[sp.Expr], fname: str='constraints'):
    # Convert expressions to strings
    # expressions_str = [str(expr) for expr in constraints]

    with open(f"{fname}.pl", 'w') as f:
        for constraint in constraints:
            f.write(sp.srepr(constraint) + '\n')
        # json.dump(expressions_str, f, indent=4, sort_keys=True)
        # #* Save the variable bounds
    # print(f"Saved to {fname}.json")
    # with open(f"{fname}_bounds.json", 'w') as f:
    #     json.dump({k: (v.lb, v.ub) for k, v in bounds.items()}, f)
    #     print(f"Saved to {fname}_bounds.json")

def load_constraints(fname: str='constraints') -> List[sp.Expr]:
    constraints = []
    with open(f"{fname}.pl", 'r') as f:
        for line in f:
            constraints.append(sp.sympify(line.strip()))
    return constraints

def clausify(expr: sp.Expr) -> sp.Expr:
    return sp.simplify_logic(expr, form='dnf', deep=True)