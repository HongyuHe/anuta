import logging
import sys
from typing import *
import sympy as sp
import pandas as pd
from collections import defaultdict
import z3
from functools import reduce

from anuta.known import *

true = sp.logic.true
false = sp.logic.false

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

log = logging.getLogger("anuta")
log.setLevel(logging.INFO)
log.propagate = False  # Don't send to root

handler = FlushStreamHandler(stream=sys.stdout)
formatter = logging.Formatter(
    "[%(name)s @ %(asctime)s] %(levelname)-7s | %(message)s", 
    datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
log.addHandler(handler)


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
for varname in ['FrameLen_1', 'IpLen_1', 'IpVersion_1', 'IpHdrLen_1', 'IpTtl_1', 'IpProto_1', 'IpSrc_1', 'IpDst_1', 'TcpSrcport_1', 'TcpDstport_1', 'TcpHdrLen_1', 'TcpLen_1', 'TcpFlags_1', 'TcpSeq_1', 'TcpAck_1', 'TcpUrgentPointer_1', 'TcpWindowSizeValue_1', 'TcpWindowSizeScalefactor_1', 'TcpWindowSize_1', 'Tsval_1', 'Tsecr_1', 'Protocol_1', 'FrameLen_2', 'IpLen_2', 'IpVersion_2', 'IpHdrLen_2', 'IpTtl_2', 'IpProto_2', 'IpSrc_2', 'IpDst_2', 'TcpSrcport_2', 'TcpDstport_2', 'TcpHdrLen_2', 'TcpLen_2', 'TcpFlags_2', 'TcpSeq_2', 'TcpAck_2', 'TcpUrgentPointer_2', 'TcpWindowSizeValue_2', 'TcpWindowSizeScalefactor_2', 'TcpWindowSize_2', 'Tsval_2', 'Tsecr_2', 'Protocol_2', 'FrameLen_3', 'IpLen_3', 'IpVersion_3', 'IpHdrLen_3', 'IpTtl_3', 'IpProto_3', 'IpSrc_3', 'IpDst_3', 'TcpSrcport_3', 'TcpDstport_3', 'TcpHdrLen_3', 'TcpLen_3', 'TcpFlags_3', 'TcpSeq_3', 'TcpAck_3', 'TcpUrgentPointer_3', 'TcpWindowSizeValue_3', 'TcpWindowSizeScalefactor_3', 'TcpWindowSize_3', 'Tsval_3', 'Tsecr_3', 'Protocol_3', 'InterArrivalMicro_1', 'InterArrivalMicro_2']:
    z3evalmap[varname] = z3.Int(varname)  #* For Netflix dataset, all variables are integers
#TODO: Add vars from other datasets


def get_quartiles(series: pd.Series):
    #* Check int or float
    isint = pd.api.types.is_integer_dtype(series)
    isfloat = pd.api.types.is_float_dtype(series)
    assert isint or isfloat, \
        f"Expected int or float series, got {series.dtype}"
    if isint:
        return int(series.max()), int(series.quantile(0.75)), int(series.quantile(0.5)), int(series.quantile(0.25)), int(series.min())
    else:
        return series.max(), series.quantile(0.75), series.quantile(0.5), series.quantile(0.25), series.min()

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

def get_tcp_flag_mask(flag_string: str) -> int:
    """
    Convert a hyphen-joined TCP flag string (e.g., 'SYN-ACK') to its numeric bitmask.
    
    :param flag_string: String of flag names joined by hyphens.
    :return: Integer bitmask.
    """
    flag_to_bit = {
        "FIN": 0x01,
        "SYN": 0x02,
        "RST": 0x04,
        "PSH": 0x08,
        "ACK": 0x10,
        "URG": 0x20,
        "ECE": 0x40,
        "CWR": 0x80
    }
    
    if not flag_string or not isinstance(flag_string, str):
        return 0
    
    return sum(flag_to_bit.get(flag.strip(), 0) for flag in flag_string.upper().split('-'))


from multiprocessing import Pool, cpu_count
from itertools import combinations
import pandas as pd

def _generate_predicate(pair, avars, df, variables, existing_columns):
    var1, var2 = pair
    results = []
    new_cols = {}
    new_categoricals = []

    def expand(var, variables, df, existing_columns):
        if "$" in var:
            v1, op, v2 = var.split('$')
            name = f"{v1}{op}{v2}"
            colname = f"Mul({v1},{v2})" if op == '×' else name
            const = int(v1) if v1 not in variables else None
            if colname not in existing_columns:
                if op == '+':
                    new_cols[colname] = const + df[v2] if const else df[v1] + df[v2]
                elif op == '×':
                    new_cols[colname] = const * df[v2] if const else df[v1] * df[v2]
            return colname, {v1, v2} if const is None else {v2}
        return var, {var}

    lhs, lhs_vars = expand(var1, variables, df, existing_columns)
    rhs, rhs_vars = expand(var2, variables, df, existing_columns)

    if lhs_vars & rhs_vars:
        return results, new_cols, new_categoricals

    # Generate predicates
    for op, fmt in [("==", "@Eq({},{})@"), (">=", "@({}>={})@")]:
        predicate = fmt.format(lhs, rhs)
        new_categoricals.append(predicate)
        if predicate not in df.columns:
            new_cols[predicate] = (df[lhs] >= df[rhs]).astype(int) if ">=" in predicate else (df[lhs] == df[rhs]).astype(int)
        results.append(predicate)

    return results, new_cols, new_categoricals


def parallel_predicate_generation(avars, df, variables, num_workers=None):
    pairs = list(combinations(avars, 2))
    existing_columns = set(df.columns)
    num_workers = num_workers or cpu_count()

    with Pool(num_workers) as pool:
        results = pool.starmap(
            _generate_predicate,
            [(pair, avars, df, variables, existing_columns) for pair in pairs]
        )

    all_preds, all_cols, all_cats = [], {}, []
    for preds, cols, cats in results:
        all_preds.extend(preds)
        all_cols.update(cols)
        all_cats.extend(cats)

    return all_preds, all_cats, all_cols

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
    return sp.to_dnf(expr)