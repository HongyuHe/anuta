from dataclasses import dataclass
from time import perf_counter
from rich import print as pprint
from collections import defaultdict
import pandas as pd
import numpy as np
import sympy as sp
import json
import sys

from anuta.grammar import (
    Bounds, Anuta, Domain, DomainType, ConstantType, Constants, VariableType, 
    TYPE_DOMIAN, group_variables_by_type)
from anuta.known import *
from anuta.utils import *
from anuta.cli import FLAGS


# #* Load configurations.
# cfg = FLAGS.config

@dataclass
class DomainCounter:
    count: int
    frequency: int
    #* Lexicographic ordering
    def __lt__(self, other):  # Less than
        if self.count == other.count:
            return self.frequency < other.frequency
        else:
            return self.count < other.count

    def __le__(self, other):  # Less than or equal to
        if self.count == other.count:
            return self.frequency <= other.frequency
        else:
            return self.count <= other.count

    def __eq__(self, other):  # Equal to
        if self.count == other.count:
            return self.frequency == other.frequency
        else:
            return self.count == other.count
        
    def __gt__(self, other):  # Greater than
        if self.count == other.count:
            return self.frequency > other.frequency
        else:
            return self.count > other.count

    def __ge__(self, other):  # Greater than or equal to
        if self.count == other.count:
            return self.frequency >= other.frequency
        else:
            return self.count >= other.count
    
    def __repr__(self) -> str:
        return f"(count:{self.count} freq:{self.frequency})"
    
    def __str__(self) -> str:
        return self.__repr__()

class Constructor(object):
    def __init__(self) -> None:
        self.label: str = None
        self.df: pd.DataFrame = None
        self.anuta: Anuta = None
        self.categoricals: list[str] = []
        self.feature_marker = ''
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        pass

    def build_abstract_domain(
        self, 
        variables: List[str],
        constants: Dict[str, Constants], 
        df: pd.DataFrame,
        drop_identifiers: bool = True
    ) -> tuple[List[str], List[str], Set[str], pd.DataFrame]:
        adf = df.copy()
        avars: Set[str] = set()
        variable_types = {}
        typed_variables, grouped_variables = group_variables_by_type(self.df.columns)
        # pprint(typed_variables)
        # pprint(grouped_variables)
        categoricals = []
        for domaintype, varnames in grouped_variables.items():
            if domaintype == DomainType.CATEGORICAL:
                categoricals.extend(varnames)
        
        #* Variables -> their types.
        for vtype, tvars in typed_variables.items():
            for var in tvars:
                # avars.add(var)
                variable_types[var] = vtype
        
        start = perf_counter()
        for varname, consts in constants.items():
            if varname in categoricals:
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
                    
                #* default: X+1
                avar = f"1$+${varname}"
                avars.add(avar)
                variable_types[avar] = vtype
            #& X>c    
            if consts.kind == ConstantType.LIMIT:
                #* Trees are good at this, so we don't augment these variables.
                pass
            
        #& All pairs of X*Y and X+Y
        for vtype in typed_variables:
            domaintype = TYPE_DOMIAN[vtype]
            if domaintype != DomainType.NUMERICAL: 
                #* Only augment numerical vars.
                continue
            
            numericals = typed_variables[vtype]
            for i, var1 in enumerate(numericals):
                for var2 in numericals[i+1:]:
                    if var1 == var2: continue
                    avar = f"{var1}$*${var2}"
                    avars.add(avar)
                    variable_types[avar] = vtype
                    avar = f"{var1}$+${var2}"
                    avars.add(avar)
                    variable_types[avar] = vtype
        
        # pprint(avars)
        log.info(f"Created {len(avars)} abstract vars.")
        
        #& Generate augmented predicates
        #& All pairs of A==B for A, B in avars
        # new_vars.add('0')
        #! The order is important: raw variables first, then augmented variables.
        avars: List[str] = variables + list(avars)
        abstract_predicates = set()
        prior_rules: Set[str] = set()
        # for i, var1 in enumerate(avars):
        #* Assumes LHS contains no operators (raw variables).
        for j, var1 in enumerate(variables):
            vtype1 = variable_types[var1]
            for var2 in avars[j+1:]:
                if var1 == var2: continue
                vtype2 = variable_types[var2]
                if vtype1 != vtype2:
                    #* Don't generate predicates with different variable types.
                    continue
                
                lhs_vars = set()
                rhs_vars = set()
                
                assert '$' not in var1, f"LHS var {var1} shouldn't be augmented."
                lhs = var1
                lhs_vars.add(lhs)
                # if "$" not in var1:
                #     lhs = var1
                #     lhs_vars.add(lhs)
                # else:
                #     #* Recover values and operators
                #     v1, op1, v2 = var1.split('$')
                #     lhs = f"({v1}{op1}{v2})"
                #     lhs_vars.add(v2)
                        
                #     const = None
                #     if v1 not in variables:
                #         #* v1 is a constant
                #         const = int(v1)
                #     else:
                #         lhs_vars.add(v1)
                    
                #     if lhs not in adf.columns:
                #         if op1 == '+':
                #             adf[lhs] = const+adf[v2] if const else adf[v1]+adf[v2]
                #         elif op1 == '*':
                #             adf[lhs] = const*adf[v2] if const else adf[v1]*adf[v2]
                    # if adf[lhs].nunique() <= 1:
                    #     #* Skip abstract variables with only one unique value.
                    #     skipped.add(lhs)
                    #     #! This's a very strong assumption.
                    #     # log.info(f"Skipping abstract {lhs=} with only one unique value.")
                    #     continue                    
                    
                if "$" not in var2:
                    rhs = var2
                    rhs_vars.add(rhs)
                else:
                    v1, op2, v2 = var2.split('$')
                    rhs = f"({v1}{op2}{v2})"
                    rhs_vars.add(v2)
                    # if op2 == '×':
                    #     rhs = f"({v1}*{v2})"
                    
                    const = None
                    if v1 not in variables:
                        #* v1 is a constant
                        const = int(v1)
                    else:
                        rhs_vars.add(v1)
                    
                    if rhs not in adf.columns:
                        if op2 == '+':
                            adf[rhs] = const+adf[v2] if const else adf[v1]+adf[v2]
                        elif op2 == '*':
                            adf[rhs] = const*adf[v2] if const else adf[v1]*adf[v2]
                    # if adf[rhs].nunique() <= 1:
                    #     #* Skip abstract variables with only one unique value.
                    #     skipped.add(rhs)
                    #     # log.info(f"Skipping abstract {rhs=} with only one unique value.")
                    #     continue
                
                if lhs_vars & rhs_vars:
                    #* Don't generate predicates with overlapping variables.
                    continue
                #& Equality predicates: Eq(A,B)
                predicate = f"@Eq({lhs},{rhs})@"
                if predicate not in adf.columns:
                    predicate_values = (adf[lhs] == adf[rhs]).astype(int)
                else:
                    log.warning(f"Duplicated {predicate=}.")
                #* Check uniqueness -> add as priors
                if predicate_values.nunique() > 1:
                    adf[predicate] = predicate_values
                    abstract_predicates.add(predicate)
                    categoricals.append(predicate)
                else:
                    predicate = predicate.replace('@', '')
                    #* Invert the predicate if it's always false.
                    if predicate_values.iloc[0] == 0:
                        predicate = f"Ne({lhs},{rhs})"
                    prior_rules.add(predicate)
                
                domaintype1 = TYPE_DOMIAN[vtype1]
                if domaintype1 == DomainType.NUMERICAL:
                    #& Comparison predicates: A>B
                    predicate = f"@({lhs}>{rhs})@"
                    if predicate not in adf.columns:
                        predicate_values = (adf[lhs] > adf[rhs]).astype(int)
                    else:
                        log.warning(f"Duplicated {predicate=}.")
                        
                    if predicate_values.nunique() > 1:
                        adf[predicate] = predicate_values
                        abstract_predicates.add(predicate)
                        categoricals.append(predicate)
                    else:
                        predicate = predicate.replace('@', '')
                        if predicate_values.iloc[0] == 0:
                            predicate = f"Not({predicate})"
                        prior_rules.add(predicate)
                    # if '+' not in lhs:
                    #     new_vars.add(f"@({lhs} > 0)")
                    # if '+' not in rhs:
                    #     new_vars.add(f"@({rhs} > 0)")
            print(f"... {i+1}/{len(avars)} avars: {len(abstract_predicates)=}, {len(prior_rules)=}", end='\r')
        end = perf_counter()
        log.info(f"Generated {len(abstract_predicates)} abstract predicates in {end-start:.2f} seconds.")
        log.info(f"Learned {len(prior_rules)} prior rules.")
        # pprint(abstract_predicates)
        # pprint(prior_rules)
        # print(f"{new_vars=}")
        assert not (abstract_predicates & set(variables)), f"Abstract variables overlap with existing variables."
        new_variables = list(abstract_predicates) + variables
        # pprint(adf[augmented_vars].head(5))
        # print(f"{len(abstract_predicates)=}")
        # print(f"{len(new_variables)=}")
        
        adf = adf[new_variables]
        if drop_identifiers:
            identifiers = typed_variables[VariableType.IP]+typed_variables[VariableType.PORT]
            adf.drop(columns=identifiers, inplace=True)
            for var in identifiers:
                if var in new_variables:
                    new_variables.remove(var)
                if var in categoricals:
                    categoricals.remove(var)
        
        return new_variables, categoricals, prior_rules, adf

class Yatesbury(Constructor):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.label = 'yatesbury'
        log.info(f"Loading data from {filepath}")
        self.categoricals = yatesbury_categoricals
        self.categoricals.remove('Decision')
        self.categoricals.remove('Label')
        self.df = pd.read_csv(filepath)
        allowed_cols = yatesbury_categoricals + yatesbury_numericals
        
        for col in self.df.columns:
            #* Drop labels for now, since we aren't aiming to predict them 
            #*  but rather help the models.
            if col not in allowed_cols:
                self.df.drop(columns=[col], inplace=True)
        
        self.df['SrcIp'] = self.df['SrcIp'].apply(yatesbury_ip_map)
        self.df['DstIp'] = self.df['DstIp'].apply(yatesbury_ip_map)
        self.df['FlowDir'] = self.df['FlowDir'].apply(yatesbury_direction_map)
        self.df['Proto'] = self.df['Proto'].apply(yatesbury_proto_map)
        # self.df['Decision'] = self.df['Decision'].apply(yatesbury_decision_map)
        self.df['FlowState'] = self.df['FlowState'].apply(yatesbury_flowstate_map)
        self.df = self.df.astype(int)
        
        domains = {}
        for name in self.df.columns:
            if name not in self.categoricals:
                domains[name] = Domain(DomainType.NUMERICAL, 
                                       Bounds(self.df[name].min().item(), 
                                              self.df[name].max().item()), 
                                       None)
            else:
                domains[name] = Domain(DomainType.CATEGORICAL, 
                                       None, 
                                       self.df[name].unique().tolist())
        self.anuta = Anuta(list(self.df.columns), domains, constants={})
        
        return

class Cicids2017(Constructor):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.label = 'cicids'
        log.info(f"Loading data from {filepath}")
        #! This dataset has to be preprocessed (removed nan, inf, spaces in cols, etc.)
        self.df = pd.read_csv(filepath)
        todrop = ['Flow_Duration', 'Packet_Length_Mean', 'Fwd_Header_Length','Bwd_Header_Length',
                  'Packet_Length_Std', 'Packet_Length_Variance', 'Fwd_Packets_s', 'Bwd_Packets_s', 
                  'Total_Fwd_Packets', 'Total_Bwd_Packets', 'Label',
                #   'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags'
                  ]
        # for col in self.df.columns:
        #     if 'std' in col.lower() or 'mean' in col.lower():
        #         todrop.append(col)
        todrop = set(todrop) & set(self.df.columns)
        self.df = self.df.drop(columns=todrop)
        
        col_to_var = {col: to_big_camelcase(col, sep='_') for col in self.df.columns}
        self.df.rename(columns=col_to_var, inplace=True)
        variables = list(self.df.columns)
        self.categoricals = ['Protocol']
        
        domains = {}
        for name in self.df.columns:
            if name not in self.categoricals:
                domains[name] = Domain(DomainType.NUMERICAL, 
                                       Bounds(self.df[name].min().item(), 
                                              self.df[name].max().item()), 
                                       None)
            else:
                domains[name] = Domain(DomainType.CATEGORICAL, 
                                       None, 
                                       self.df[name].unique().tolist())
        
        self.constants: dict[str, Constants] = {}
        for name in self.df.columns:
            if any(keyword in name.lower() for keyword in ('min', 'mean', 'max', 'std')):
                self.constants[name] = Constants(
                    kind=ConstantType.SCALAR,
                    values=[1] #* Issue identity (global) constraints for these variables.
                )
            if any(keyword in name.lower() for keyword in ('packets', 'flag')):
                self.constants[name] = Constants(
                    kind=ConstantType.LIMIT,
                    values=[0] #* Compare these variables to zero (>0)
                )
        self.anuta = Anuta(variables, domains, self.constants)
        pprint(self.anuta.variables)
        pprint(self.anuta.domains)
        pprint(self.anuta.constants)
        return
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        indexset = {}
        #^ dict[var: dict[val: array(indices)]] // Var -> {Distinct value -> indices}
        for cat in self.categoricals:
            # print(f"Processing {cat}")
            indices = df.groupby(by=cat).indices
            indexset[cat] = indices
        for name in self.constants:
            if name in indexset:
                continue
            constants = self.constants[name]
            #* Create indexses for numerical variables with associated limits.
            #! Not ideal since it's only considering ==limit (not >limit or <limit).
            if constants.kind == ConstantType.LIMIT:
                for const in constants.values:
                    indices = df[df[name] == const].index.to_numpy()
                    if indices.size > 0:
                        indexset[name] = {const: indices}
                    indices = df[df[name] != const].index.to_numpy()
                    if indices.size > 0:
                        indexset[name] |= {'neq': indices}
                    
        #TODO: Create index set also for numerical variables.
        fcount = defaultdict(dict)
        #^ dict[var: dict[val: count]] // Var -> {Value of interest -> (count,freq)}
        for cat in indexset:
            if cat in self.constants and self.constants[cat].kind != ConstantType.LIMIT:
                #* Don't enumerate the domain if it has associated constants.
                values = self.constants[cat].values
            else:
                values = indexset[cat].keys()
            
            for key in values:
                #* Initialize the counters to the frequency of the value in the data.
                #& Prioritize rare values (inductive bias).
                #& Using the frequency only as a tie-breaker [count, freq].
                freq = len(indexset[cat][key])
                dc = DomainCounter(count=0, frequency=freq)
                fcount[cat] |= {key: dc} if type(key) in [int, str] else {key.item(): dc}
        return indexset, fcount

class Mawi(Constructor):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.label = 'mawi'
        log.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        self.df["frame.time_epoch"] = pd.to_datetime(self.df["frame.time_epoch"], unit="s")
        self.df['tcp.flags'] = self.df['tcp.flags'].fillna(value='0x0').apply(int, base=16)
        self.df['ip.version'] = self.df['ip.version'].apply(lambda v: int(v) if ',' not in str(v) else int(v.split(',')[0]))
        self.df = self.df.rename(columns=rename_pcap(self.df.columns))[used_pcap_cols]

        self.df = self.df[self.df.protocol=='TCP'].reset_index(drop=True)
        self.df['tcp_urgent_pointer'] = self.df['tcp_urgent_pointer'].fillna(value=0).astype(int)
        self.df['tcp_window_size_scalefactor'] = self.df['tcp_window_size_scalefactor'].fillna(value=1).astype(int)
        self.df['tcp_window_size_scalefactor'] = self.df['tcp_window_size_scalefactor'].apply(lambda s: s if s > 0 else 1)
        self.df.dropna(subset=[
            'tcp_hdr_len',
            'tcp_len',
            'tcp_seq',
            'tcp_ack',
            'tcp_window_size_value',
            'tcp_window_size',
        ], inplace=True)
        columns_with_nan_mask = self.df.isna().any()
        log.warning(f"Columns with missing values: {self.df.columns[columns_with_nan_mask].tolist()}")
        df = self.df
        # Apply the normalization
        flow_keys = df.apply(normalize_pcap_5tuple, axis=1)
        df = pd.concat([df, flow_keys], axis=1)
        df_sorted = df.sort_values(
            by=["flow_ip_1", "flow_ip_2", "flow_port_1", "flow_port_2", "flow_proto", "frame_time_epoch", "frame_number"]
        )
        df = df_sorted.reset_index(drop=True).drop(columns=flow_keys)
        df = df.drop(columns=['frame_number', 'frame_time_epoch', 'protocol', 'tsval', 'tsecr'])

        col_to_var = {col: to_big_camelcase(col, sep='_') for col in df.columns}
        df.rename(columns=col_to_var, inplace=True)
        
        base_idx = 70_000 #* Prevent collisions with other fields.
        ip_map = {ip: i+base_idx for i, ip in enumerate(set(df.IpSrc.unique()) | set(df.IpDst.unique()))}
        df['IpSrc'] = df['IpSrc'].apply(lambda x: ip_map[x])
        df['IpDst'] = df['IpDst'].apply(lambda x: ip_map[x])
        df = df.astype(int)
        self.df = generate_sliding_windows(df, stride=1, window=3)
        
        variables = list(self.df.columns)
        # typed_variables, grouped_variables = group_variables_by_type(self.df.columns)
        # self.categoricals = [varname for varname, domaintype in grouped_variables.items()
        #                      if domaintype == DomainType.CATEGORICAL]
        
        # for name in self.df.columns:
        #     if any(keyword in name.lower() for keyword in ('src', 'dst', 'proto', 'flags')):
        #         self.categoricals.append(name)
        # print(f"Categorical variables: {self.categoricals}")
        
        # #TODO: Move to known.py
        # typed_variables: Dict[VariableType, List[str]] = defaultdict(list)
        # for name in self.df.columns:
        #     if any(keyword in name.lower() for keyword in ('src', 'dst', 'port')):
        #         typed_variables[VariableType.ID].append(name)
        #     elif any(keyword in name.lower() for keyword in ('len', 'size', 'window')):
        #         typed_variables[VariableType.SIZE].append(name)
        #     elif any(keyword in name.lower() for keyword in ('time', 'epoch', 'duration', 'ts')):
        #         typed_variables[VariableType.TIME].append(name)
        #     elif any(keyword in name.lower() for keyword in ('seq', 'ack', 'ttl')):
        #         typed_variables[VariableType.COUNT].append(name)
        #     elif any(keyword in name.lower() for keyword in ('proto', 'flags', 'pointer', 'version')):
        #         typed_variables[VariableType.CLASS].append(name)
        #     else:
        #         raise ValueError(f"Unknown variable type for {name}.")
        # pprint(typed_variables)
            
        #TODO: Improve
        self.constants: dict[str, Constants] = {}
        for name in self.df.columns:
            if 'seq' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.ADDITION,
                    values=netflix_seqnum_increaments
                )
            if 'len' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.LIMIT,
                    values=netflix_tcplen_limits
                )

        variables, self.categoricals, prior_rules, self.df = self.build_abstract_domain(
            variables, self.constants, self.df)
        # #* Remove identifiers after adding the predicates.
        # self.df.drop(columns=typed_variables[VariableType.IP] + 
        #              typed_variables[VariableType.PORT], inplace=True)
        
        domains = {}
        for name in self.df.columns:
            if name not in self.categoricals:
                domains[name] = Domain(DomainType.NUMERICAL, 
                                       Bounds(self.df[name].min().item(), 
                                              self.df[name].max().item()), 
                                       None)
            else:
                domains[name] = Domain(DomainType.CATEGORICAL, 
                                       None, 
                                       self.df[name].unique().tolist())
        #TODO: Add prior rules
        self.anuta = Anuta(variables, domains, self.constants, prior_kb=prior_rules)
        # pprint(self.anuta.variables)
        # pprint(self.anuta.domains)
        # pprint(self.anuta.constants)
        return

class Netflix(Constructor):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.label = 'netflix'
        STRIDE = 1
        WINDOW = 3
        
        log.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        self.df["frame.time_epoch"] = pd.to_datetime(self.df["frame.time_epoch"], unit="s")
        self.df['tcp.flags'] = self.df['tcp.flags'].apply(int, base=16)
        self.df = self.df.rename(columns=rename_pcap(self.df.columns))[used_pcap_cols]
        self.df['tcp_window_size_scalefactor'] = self.df['tcp_window_size_scalefactor'].fillna(value=1).astype(int)
        
        df = self.df
        # Step 1:Apply the normalization
        flow_keys = df.apply(normalize_pcap_5tuple, axis=1)
        df = pd.concat([df, flow_keys], axis=1)

        # Step 2: Sort by normalized 5-tuple and frame_time_epoch
        df_sorted = df.sort_values(
            by=["flow_ip_1", "flow_ip_2", "flow_port_1", "flow_port_2", "flow_proto", 
                "frame_time_epoch", "frame_number"]
        )

        # Step 3: Group by normalized 5-tuple
        grouped = df_sorted.groupby(["flow_ip_1", "flow_ip_2", "flow_port_1", "flow_port_2", "flow_proto"])

        for flow_id, flow_df in grouped:
            print(f"Flow {flow_id}: {len(flow_df)} packets")
        #! Assuming a single (bidirectional) flow for now.
        self.df = flow_df.drop(columns=flow_keys)
        todrop = ['frame_number', 'frame_time_epoch', 'protocol', 'frame_len', 'ip_version']
        for col in todrop:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        col_to_var = {col: to_big_camelcase(col, sep='_') for col in self.df.columns}
        self.df.rename(columns=col_to_var, inplace=True)
        assert self.df.IpSrc.nunique() == self.df.IpDst.nunique() == 2, "Expected 2 IPs in a flow."
        assert self.df.TcpSrcport.nunique() == self.df.TcpDstport.nunique() == 2, "Expected 2 ports in a flow."
        #* Encode the IPs to 0 and 1.
        ip_map = {ip: i for i, ip in enumerate(self.df.IpSrc.unique())}
        self.df['IpSrc'] = self.df['IpSrc'].apply(lambda x: ip_map[x])
        self.df['IpDst'] = self.df['IpDst'].apply(lambda x: ip_map[x])
        #* Encode the ports to 0 and 1.
        port_map = {port: i for i, port in enumerate(self.df.TcpSrcport.unique())}
        self.df['TcpSrcport'] = self.df['TcpSrcport'].apply(lambda x: port_map[x])
        self.df['TcpDstport'] = self.df['TcpDstport'].apply(lambda x: port_map[x])
        self.df = self.df.astype(int)
        
        self.df = generate_sliding_windows(self.df, stride=STRIDE, window=WINDOW)
        
        variables = list(self.df.columns)
        self.categoricals = []
        for name in self.df.columns:
            if any(keyword in name.lower() for keyword in ('src', 'dst', 'proto', 'flags')):
                self.categoricals.append(name)
        # print(f"Categorical variables: {self.categoricals}")
        
        self.constants: dict[str, Constants] = {}
        for name in self.df.columns:
            if 'seq' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.ADDITION,
                    values=netflix_seqnum_increaments
                )
            if 'len' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.LIMIT,
                    values=netflix_tcplen_limits
                )
        # variables, self.categoricals, self.df = self.build_abstract_domain(
        #     variables, self.constants, self.categoricals, self.df)
        # self.df.to_csv('netflix_abstracted.csv', index=False)
        variables, self.categoricals, prior_rules, self.df = self.build_abstract_domain(
            variables, self.constants, self.df)
        
        domains = {}
        for name in self.df.columns:
            if name not in self.categoricals:
                domains[name] = Domain(DomainType.NUMERICAL, 
                                       Bounds(self.df[name].min().item(), 
                                              self.df[name].max().item()), 
                                       None)
            else:
                domains[name] = Domain(DomainType.CATEGORICAL, 
                                       None, 
                                       self.df[name].unique().tolist())
        
        self.anuta = Anuta(variables, domains, self.constants, prior_kb=prior_rules)
        # pprint(self.anuta.variables)
        # pprint(self.anuta.domains)
        # pprint(self.anuta.constants)
        return
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        indexset = {}
        #^ dict[var: dict[val: array(indices)]] // Var -> {Distinct value -> indices}
        for cat in self.categoricals:
            # print(f"Processing {cat}")
            indices = df.groupby(by=cat).indices
            indexset[cat] = indices
        #TODO: Create index set also for numerical variables.
        fcount = defaultdict(dict)
        #^ dict[var: dict[val: count]] // Var -> {Value of interest -> (count,freq)}
        for cat in indexset:
            if cat in self.constants:
                #* Don't enumerate the domain if it has associated constants.
                values = self.constants[cat].values
            else:
                values = indexset[cat].keys()
            
            for key in values:
                #! Some values could be in the constants but not in the data (partition).
                if key in domains[cat].values and key in indexset[cat]:
                    #* Initialize the counters to the frequency of the value in the data.
                    #& Prioritize rare values (inductive bias).
                    #& Using the frequency only as a tie-breaker [count, freq].
                    freq = len(indexset[cat][key])
                    dc = DomainCounter(count=0, frequency=freq)
                    fcount[cat] |= {key: dc} if type(key) == int else {key.item(): dc}
        return indexset, fcount
        
        
class Cidds001(Constructor):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.label = 'cidds'
        log.info(f"Loading data from {filepath}")
        self.df: pd.DataFrame = pd.read_csv(filepath).iloc[:, :11]
        #* Discard the timestamps for now, and Flows is always 1.
        for col in ['Date first seen', 'Flows']:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        col_to_var = {col: to_big_camelcase(col) for col in self.df.columns}
        self.df.rename(columns=col_to_var, inplace=True)
        variables = list(self.df.columns)
        self.feature_marker = ''
        
        #* Convert the Flags and Proto columns to integers        
        self.df['Flags'] = self.df['Flags'].apply(cidds_flag_map)
        self.df['Proto'] = self.df['Proto'].apply(cidds_proto_map)
        self.df['SrcIpAddr'] = self.df['SrcIpAddr'].apply(cidds_ip_map)
        self.df['DstIpAddr'] = self.df['DstIpAddr'].apply(cidds_ip_map)
        self.df['SrcPt'] = self.df['SrcPt'].apply(cidds_port_map)
        self.df['DstPt'] = self.df['DstPt'].apply(cidds_port_map)
        self.categoricals = cidds_categoricals
        
        #* Add the constants associated with the vars.
        self.constants: dict[str, Constants] = {}
        for name in variables:
            if 'ip' in name.lower():
                #& Don't need to add the IP constants here, as the domain is small and can be enumerated.
                self.constants[name] = Constants(
                    kind=ConstantType.ASSIGNMENT,
                    values=cidds_constants['ip']
                )
            elif 'pt' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.ASSIGNMENT,
                    values=cidds_constants['port']
                )
            elif 'packet' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.SCALAR,
                    #* Sort the values in ascending order.
                    values=sorted(cidds_constants['packet'])
                )
            elif 'bytes' in name.lower():
                self.constants[name] = Constants(
                    kind=ConstantType.SCALAR,
                    #* Sort the values in ascending order.
                    values=sorted(cidds_constants['bytes'])
                )
        
        prior_rules: Set[str] = set()
        variables, self.categoricals, prior_rules, self.df = self.build_abstract_domain(
            variables, self.constants, self.df, drop_identifiers=False)
                
        domains = {}
        for name in self.df.columns:
            if name not in self.categoricals:
                domains[name] = Domain(DomainType.NUMERICAL, 
                                      Bounds(self.df[name].min().item(), 
                                             self.df[name].max().item()), 
                                      None)
            else:
                domains[name] = Domain(DomainType.CATEGORICAL, 
                                      None, 
                                      self.df[name].unique())

        self.anuta = Anuta(variables, domains, self.constants, prior_kb=prior_rules)
        # pprint(self.anuta.variables)
        # pprint(self.anuta.domains)
        # pprint(self.anuta.constants)
        # print(f"Prior KB size: {len(self.anuta.prior_kb)}:")
        # print(f"\t{self.anuta.prior_kb}\n")
        
        # # save_constraints(self.anuta.initial_kb + self.anuta.prior_kb, 'initial_constraints_arity3_negexpr')
        # print(f"Initial KB size: {len(self.anuta.initial_kb)}")
        return
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        indexset = {}
        #^ dict[var: dict[val: array(indices)]] // Var -> {Distinct value -> indices}
        for cat in self.categoricals:
            indices = df.groupby(by=cat).indices
            filtered_indices = {}
            if cat in self.constants:
                for val in indices:
                    if val in self.constants[cat].values:
                        filtered_indices[val] = indices[val]
            indexset[cat] = filtered_indices
        #TODO: Create index set also for numerical variables.
        
        fcount = defaultdict(dict)
        #^ dict[var: dict[val: count]] // Var -> {Value of interest -> (count,freq)}
        for cat in indexset:
            if cat in self.constants:
                #* Don't enumerate the domain if it has associated constants.
                values = self.constants[cat].values
            else:
                values = indexset[cat].keys()
            
            for val in values:
                #! Some values could be in the constants but not in the data (partition).
                if val in domains[cat].values and val in indexset[cat]:
                    #* Initialize the counters to the frequency of the value in the data.
                    #& Prioritize rare values (inductive bias).
                    #& Using the frequency only as a tie-breaker [count, freq].
                    freq = len(indexset[cat][val])
                    dc = DomainCounter(count=0, frequency=freq)
                    fcount[cat] |= {val: dc} if type(val) == int else {val.item(): dc}
        # for varname, dc in fcounts.items():
        # domain = anuta.domains.get(varname)
        # assert domain.kind == Kind.CATEGORICAL, "Found numerical variable in DC counter."
        
        # filtered_dc = {}
        # for key in dc:
        #     if key in domain.values:
        #         filtered_dc |= dc[key]
        # fcounts[varname] = filtered_dc
        return indexset, fcount
        

class Millisampler(Constructor):
    def __init__(self, filepath) -> None:
        self.label = 'metadc'
        log.info(f"Loading data from {filepath}")
        self.df: pd.DataFrame = pd.read_csv(filepath)
        todrop = ['rackid', 'hostid']
        for col in self.df.columns:
            if col in todrop:
                self.df.drop(columns=[col], inplace=True)
        variables = list(self.df.columns)
        #* All variables are numerical, so we don't need to specify categoricals.
        self.categoricals = []
        self.feature_marker = 'Agg'
        
        domains = {}
        for name in self.df.columns:
            domains[name] = Domain(DomainType.NUMERICAL, 
                                   Bounds(self.df[name].min().item(), 
                                          self.df[name].max().item()), 
                                   None)
        self.anuta = Anuta(variables, domains, constants={})
        return

# class Millisampler(Constructor):
#     def __init__(self, filepath: str) -> None:
#         boundsfile = f"./data/meta_bounds.json"
#         print(f"Loading data from {filepath}")
#         self.df = pd.read_csv(filepath)
        
#         variables = []
#         for col in self.df.columns:
#             if col not in ['server_hostname', 'window', 'stride']:
#                 # if len(col.split('_')) > 1 and col.split('_')[1].isdigit(): continue
#                 variables.append(col)
#         constants = {
#             'burst_threshold': round(2891883 / 7200), # round(0.5*metadf.ingressBytes_sampled.max().item()),
#         }
        
#         canaries = {
#             'canary_max10': (0, self.df.ingressBytes_aggregate.max().item()),
#             #^ Max(u1, u2, ..., u10) == canary_max10
#             'canary_premise': (0, 1),
#             'canary_conclusion': (constants['burst_threshold']+1, constants['burst_threshold']+1),
#             #^ (canary_premise > 0) => (canary_max10 + 1 ≥ burst_threshold)
#         }
#         variables.extend(canaries.keys())

#         #* Load the bounds directly from the file
#         with open(boundsfile, 'r') as f:
#             bounds = json.load(f)
#             bounds = {k: Bounds(v[0], v[1]) for k, v in bounds.items()}
#         # bounds = {}
#         # for col in metadf.columns:
#         #     if col in ['server_hostname', 'window', 'stride']: 
#         #         continue
#         #     bounds[col] = Bounds(metadf[col].min().item(), metadf[col].max().item())
#         for n, c in constants.items():
#             bounds[n] = Bounds(c, c)
#         for n, c in canaries.items():
#             bounds[n] = Bounds(c[0], c[1])
        
#         self.anuta = AnutaMilli(variables, bounds, constants, operators=[0, 1, 2])
#         pprint(self.anuta.variables)
#         pprint(self.anuta.constants)
#         pprint(self.anuta.bounds)
        
#         self.anuta.populate_kb()
#         return