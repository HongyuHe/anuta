import math
from mlxtend.frequent_patterns import apriori, fpgrowth, hmine, fpmax
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from time import perf_counter
from typing import *
from collections import defaultdict
import sympy as sp
from tqdm import tqdm
from rich import print as pprint
import itertools

from anuta.cli import FLAGS
from anuta.constructor import Constructor
from anuta.known import *
from anuta.utils import log
from anuta.grammar import DomainType
from anuta.theory import Theory


def _encode_rule_pair(antecedent: FrozenSet[str], consequent: FrozenSet[str], sep='::') -> List[Tuple[str, FrozenSet[str]]]:
    # Encode antecedents
    predicates = frozenset(f"Eq({pred.split(sep)[0]},{pred.split(sep)[-1]})" for pred in antecedent)
    predicates = set()
    for pred in antecedent:
        varname = pred.split(sep)[0]
        value = pred.split(sep)[-1]
        if '@' in varname:
            #* Abstract variable like `@Eq(SrcPt,DstPt)@`
            assert value.isdigit(), f"Invalid value for abstract variable {varname}: {value}"
            predicate = varname.replace('@', '') 
            if int(value) == 0:
                predicate = f"Not({predicate})"
            predicates.add(predicate)
        else:
            predicates.add(f"Eq({varname},{value})")
    predicates = frozenset(predicates)

    # Return list of (consequent_predicate, antecedents)
    # return [(f"Eq({pred.split(self.sep)[0]},{pred.split(self.sep)[-1]})", predicates) for pred in consequent]
    conseq_predicates = []
    for pred in consequent:
        varname = pred.split(sep)[0]
        value = pred.split(sep)[-1]
        if '@' in varname:
            assert value.isdigit(), f"Invalid value for abstract variable {varname=} {value=}"
            conseq_predicate = varname.replace('@', '') 
            if int(value) == 0:
                conseq_predicate = f"Not({conseq_predicate})"
        else:
            conseq_predicate = f"Eq({varname},{value})"
        conseq_predicates.append(conseq_predicate)
    return [(conseq_predicate, predicates) for conseq_predicate in conseq_predicates]        

def get_missing_domain_rules(examples, domains) -> List[str]:
    """
    Generate rules for missing domains based on the provided examples and domains.
    """
    rules = []
    for varname, domain in domains.items():
        if domain.kind == DomainType.CATEGORICAL:
            # Check if the variable is present in the examples
            if varname not in examples.columns:
                continue
            
            # Get unique values in the column
            unique_values = set(examples[varname].unique())
            domain_values = set(domain.values)
            missing_values = domain_values - unique_values
            if missing_values:
                # Create rules for missing values
                for value in missing_values:
                    rule = f"Ne({varname},{value})"
                    rules.append(rule)
    return rules

class AssociationRuleLearner:
    sep = '::'
    
    def __init__(self, constructor: Constructor, algorithm='hmine', 
                 limit=None, min_support=1e-10, **kwargs):
        self.algorithm = algorithm
        self.min_support = min_support
        self.kwargs = kwargs
        self.dataset = constructor.label
        #* Only support categorical variables.
        self.df = constructor.df[constructor.categoricals]
        
        if limit and limit < self.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            self.df = self.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = 'all'
        
        log.info(f"Converting dataset to transactions...")
        transactions = self.df.astype(str).apply(
            lambda row: [f"{col}{self.sep}{val}" for col, val in row.items()], axis=1).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.df = pd.DataFrame(te_ary, columns=te.columns_)
        self.domains = constructor.anuta.domains
        self.learned_rules = list(constructor.anuta.prior_kb) if constructor.anuta else []
        # pprint(self.learned_rules)

    def learn(self, min_threshold=1) -> pd.DataFrame:
        log.info(f"Learning from {len(self.df)} examples and {self.df.shape[1]} items with {self.algorithm}.")
        method = None
        match self.algorithm:
            case 'apriori':
                method = apriori
            case 'fpgrowth':
                method = fpgrowth
            case 'hmine':
                method = hmine
            # case 'fpmax':
            #     method = fpmax
            case _:
                raise ValueError("Unsupported algorithm: {}".format(self.algorithm))
        
        start = perf_counter()
        # # Pre-stringify entire DataFrame once (avoids repeated astype calls)
        # df_str = self.df.astype(str)
        
        # bucket_size = 2
        # extracted_rules = []
        # rulecolsets = []
        # while bucket_size < len(self.df.columns):
        #     adfs = []
        #     total_combo = math.comb(len(df_str.columns), bucket_size)
        #     log.info(f"Generating size-{bucket_size} itemsets ({total_combo} combinations)...")
        #     itemsets = itertools.combinations(df_str.columns, bucket_size)
        #     pruned_itemsets = list(itemsets)
        #     for itemset in tqdm(itemsets, desc=f"Pruning size-{bucket_size} itemsets", total=total_combo):
        #         itemset = set(itemset)
        #         for rulecolset in rulecolsets:
        #             if itemset.issubset(rulecolset) or itemset.issuperset(rulecolset):
        #                 pruned_itemsets.remove(itemset)
        #                 break
        #     log.info(f"Pruned {total_combo - len(pruned_itemsets)} itemsets; {len(pruned_itemsets)} remain.")
        #     # itemsets = pruned_itemsets    
        #     rulecolsets = []
        #     for cols in tqdm(
        #         pruned_itemsets,
        #         desc=f"Mining size-{bucket_size} itemsets",
        #         total=len(pruned_itemsets)
        #     ):
        #         subset = df_str[list(cols)]

        #         # Vectorized transaction building: each row becomes tuple of "col<sep>value"
        #         col_array = subset.columns.to_numpy()[np.newaxis, :]
        #         transactions = (col_array + self.sep + subset.to_numpy()).tolist()

        #         te = TransactionEncoder()
        #         te_ary = te.fit(transactions).transform(transactions)
        #         itemsetdf = pd.DataFrame(te_ary, columns=te.columns_)
                
        #         frequent_itemsets= method(
        #             itemsetdf, min_support=self.min_support, use_colnames=True, **self.kwargs)
        #         # log.info(f"Frequent itemsets found: {len(frequent_itemsets)}")
                
        #         adf = association_rules(frequent_itemsets, 
        #                                     metric="confidence",
        #                                     #* Learn hard rules by default (min_threshold=1)
        #                                     min_threshold=min_threshold,) #, support_only=True
        #         adfs.append(adf)
        #         if len(adfs) > 30:
        #             break
                
        #     aruledf = pd.concat(adfs, ignore_index=True)
        #     arules, colsets = self.extract_rules(aruledf)
        #     extracted_rules.extend(arules)
        #     rulecolsets.extend(colsets)
        #     # pprint(arules)
        #     # pprint(colsets)
        #     log.info(f"Association rules learned: {len(extracted_rules)}")
        #     log.info(f"Rule column sets: {len(rulecolsets)}")
        #     bucket_size += 1
        #     if bucket_size > 4:
        #         exit(0)
        
        frequent_itemsets= method(
            self.df, min_support=self.min_support, use_colnames=True, **self.kwargs)
        log.info(f"Frequent itemsets found: {len(frequent_itemsets)}")
        
        aruledf = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    #* Learn hard rules by default (min_threshold=1)
                                    min_threshold=min_threshold,) #, support_only=True
        log.info(f"Association rules found: {len(aruledf)}")
        end = perf_counter()
        log.info(f"Association rule learning took {end - start:.2f} seconds.")
        
        start = perf_counter()
        self.learned_rules += self.extract_rules(aruledf)
        # self.learned_rules += self.extract_rules_parallel(aruledf)
        log.info(f"Extracted {len(self.learned_rules)} rules.")
        end = perf_counter()
        log.info(f"Rule extraction took {end - start:.2f} seconds.")
        
        assumptions = set()
        # for varname, domain in self.domains.items():
        #     if domain.kind == DomainType.CATEGORICAL and '@' not in varname:
        #         assumptions.add(f"{varname} >= 0")
        #         assumptions.add(f"{varname} <= {max(domain.values)}")
                
        #         full_domain = set(val for val in range(max(domain.values) + 1))
        #         missing_values = full_domain - set(domain.values)
        #         ne_predicates = []
        #         for value in missing_values:
        #             ne_predicates.append(f"Ne({varname},{value})")
        #         #* Don't add negative assumptions for port variables.
        #         keywords = ['pt', 'port']
        #         if ne_predicates and not any(keyword in varname.lower() for keyword in keywords):
        #             assumptions.add(' & '.join(ne_predicates))
        # assumptions = set(assumptions) | set(get_missing_domain_rules(self.df, self.domains))
        
        rules = set(self.learned_rules) | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        sprules = list(filter(lambda r: r not in (sp.true, sp.false), sprules))
        outputf = f"{self.algorithm}_{self.dataset}_{self.num_examples}"
        if FLAGS.label:
            outputf += f"_{FLAGS.label}.pl"
        else:            
            outputf += ".pl"
        Theory.save_constraints(sprules, outputf)
        
        return
    
    def extract_rules_parallel(self, aruledf: pd.DataFrame) -> List[str]:
        log.info("Extracting rules in parallel...")
        premisesmap = defaultdict(set)
        #* Parallel rule encoding
        num_cores = cpu_count()
        futures = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for antecedent, consequent in zip(
                aruledf.antecedents.to_numpy(), 
                aruledf.consequents.to_numpy()
            ):
                futures.append(executor.submit(_encode_rule_pair, antecedent, consequent))

            for future in as_completed(futures):
                for conseq_predicate, predicates in future.result():
                    premisesmap[conseq_predicate].add(predicates)

        #* Merge premises → consequents (serial logic remains)
        consequentsmap = defaultdict(set)
        for conseq_predicate, premise_sets in premisesmap.items():
            for premise in premise_sets:
                is_redundant = False
                for other in premise_sets - {premise}:
                    if premise > other:  # premise is a superset
                        is_redundant = True
                        break
                if not is_redundant:
                    consequentsmap[premise].add(conseq_predicate)

        #* Format rules in valid syntax
        arules = []
        for antecedents, consequents in consequentsmap.items():
            premise = ' & '.join(antecedents)
            conclusion = ' & '.join(consequents)
            arule = f"({premise}) >> ({conclusion})"
            arules.append(arule)

        return arules
    
    # def extract_rules(self, aruledf: pd.DataFrame) -> Tuple[List[str], List[Set[str]]]:
    #     def _contains_binary(colset: Set[str], sep: str) -> bool:
    #         """Return True if colset contains both ::0 and ::1 for the same predicate."""
    #         bin_tracker = defaultdict(set)  # predicate_name -> set of values seen
    #         for raw_token in colset:
    #             if '@' not in raw_token:
    #                 #* Only consider abstract variables
    #                 continue
    #             if sep in raw_token:
    #                 pred, val = raw_token.split(sep)[0], raw_token.split(sep)[-1]
    #                 if val in ("0", "1"):  # only care about binary predicates
    #                     bin_tracker[pred].add(val)
    #                     if len(bin_tracker[pred]) == 2:  # both 0 and 1 present
    #                         return True
    #         return False
        
    #     premisesmap = defaultdict(set)       # parsed_consequent -> set of parsed_antecedent_sets
    #     rawmap = defaultdict(set)            # parsed_consequent -> set of raw_antecedent_sets
    #     conseqrawmap = {}                     # parsed_consequent -> raw_consequent_token

    #     sep = self.sep

    #     for antecedent, consequent in zip(aruledf.antecedents.to_numpy(), aruledf.consequents.to_numpy()):
    #     #     ,
    #     #     desc="Encoding antecedents and consequents",
    #     #     total=len(aruledf)
    #     # ):
    #         # Parsed + raw for the antecedents
    #         parsed_preds = set()
    #         raw_preds = set()

    #         for predicate in antecedent:
    #             raw_preds.add(predicate)  # keep original token

    #             varname, value = predicate.split(sep)[0], predicate.split(sep)[-1]
    #             if '@' in varname:
    #                 assert value.isdigit(), f"Invalid value for abstract variable {varname}: {value}"
    #                 parsed_pred = varname.replace('@', '')
    #                 if int(value) == 0:
    #                     parsed_pred = f"Not({parsed_pred})"
    #             else:
    #                 parsed_pred = f"Eq({varname},{value})"

    #             parsed_preds.add(parsed_pred)

    #         parsed_preds = frozenset(parsed_preds)
    #         raw_preds = frozenset(raw_preds)

    #         # Consequents
    #         for predicate in consequent:
    #             raw_conseq = predicate  # original raw consequent
    #             varname, value = predicate.split(sep)[0], predicate.split(sep)[-1]
    #             if '@' in varname:
    #                 assert value.isdigit(), f"Invalid value for abstract variable {varname=} {value=}"
    #                 conseq_predicate = varname.replace('@', '')
    #                 if int(value) == 0:
    #                     conseq_predicate = f"Not({conseq_predicate})"
    #             else:
    #                 conseq_predicate = f"Eq({varname},{value})"

    #             premisesmap[conseq_predicate].add(parsed_preds)
    #             rawmap[conseq_predicate].add(raw_preds)
    #             conseqrawmap[conseq_predicate] = raw_conseq  # store raw form

    #     # Redundancy pruning
    #     consequentsmap = defaultdict(set)   # parsed_antecedent_set -> parsed_consequents
    #     consequentsraw = defaultdict(set)   # parsed_antecedent_set -> raw tokens (antecedents + consequents)

    #     for conseq_predicate, premise_sets in premisesmap.items():
    #     # ,
    #     #     desc="Pruning redundant premises",
    #     #     total=len(premisesmap)
    #     # ):
    #         sorted_premises = sorted(premise_sets, key=len)
    #         corresponding_raws = sorted(rawmap[conseq_predicate], key=len)

    #         non_redundant = []

    #         for parsed_set, raw_set in zip(sorted_premises, corresponding_raws):
    #             is_redundant = any(parsed_set > q for q in non_redundant)
    #             if not is_redundant:
    #                 non_redundant.append(parsed_set)
    #                 consequentsmap[parsed_set].add(conseq_predicate)

    #                 # antecedent raw tokens
    #                 consequentsraw[parsed_set].update(raw_set)
    #                 # add raw consequent token too
    #                 consequentsraw[parsed_set].add(conseqrawmap[conseq_predicate])

    #     # Format final outputs
    #     arules = []
    #     colsets = []

    #     for antecedents, consequents in consequentsmap.items():
    #     #     desc="Formatting rules",
    #     #     total=len(consequentsmap)
    #     # ):
    #         colset = consequentsraw[antecedents]
    #         if _contains_binary(colset, sep):
    #             #* Skip rules containing the same binary predicates
    #             # pprint(arule)
    #             # pprint(colset)
    #             # exit(0)
    #             continue
    #         premise = ' & '.join(antecedents)
    #         conclusion = ' & '.join(consequents)
    #         arule = f"({premise}) >> ({conclusion})"
            
    #         arules.append(arule)
    #         colsets.append(colset)

    #     return arules, colsets


    def extract_rules(self, aruledf: pd.DataFrame) -> List[str]:
        premisesmap = defaultdict(set)

        #* Build mapping from each consequent predicate to its supporting antecedent sets
        for antecedent, consequent in tqdm(zip(
            aruledf.antecedents.to_numpy(), 
            aruledf.consequents.to_numpy()
        ), desc="Encoding antecedents and consequents", total=len(aruledf)):
            #* Encode antecedents
            predicates = set()
            for predicate in antecedent:
                varname = predicate.split(self.sep)[0]
                value = predicate.split(self.sep)[-1]
                if '@' in varname:
                    #* Abstract variable like `@Eq(SrcPt,DstPt)@`
                    assert value.isdigit(), f"Invalid value for abstract variable {varname}: {value}"
                    predicate = varname.replace('@', '') 
                    if int(value) == 0:
                        predicate = f"Not({predicate})"
                    predicates.add(predicate)
                else:
                    predicates.add(f"Eq({varname},{value})")
            predicates = frozenset(predicates)
            
            #* Encode consequents and populate mapping
            for predicate in consequent:
                varname = predicate.split(self.sep)[0]
                value = predicate.split(self.sep)[-1]
                if '@' in varname:
                    assert value.isdigit(), f"Invalid value for abstract variable {varname=} {value=}"
                    conseq_predicate = varname.replace('@', '') 
                    if int(value) == 0:
                        conseq_predicate = f"Not({conseq_predicate})"
                else:
                    conseq_predicate = f"Eq({varname},{value})"
                premisesmap[conseq_predicate].add(predicates)

        # # Merge premises → consequents
        # consequentsmap = defaultdict(set)
        # for conseq_predicate, premise_sets in premisesmap.items():
        #     for premise in premise_sets:
        #         is_redundant = False
        #         for other in premise_sets - {premise}:
        #             if premise > other:  # premise is a superset
        #                 is_redundant = True
        #                 break
        #         if not is_redundant:
        #             consequentsmap[premise].add(conseq_predicate)
        
        #* Optimized redundancy pruning
        consequentsmap = defaultdict(set)
        for conseq_predicate, premise_sets in tqdm(
            premisesmap.items(),
            desc="Pruning redundant premises",
            total=len(premisesmap)
        ):
            #* Smaller sets first
            sorted_premises = sorted(premise_sets, key=len)  
            non_redundant = []

            for i, p in enumerate(sorted_premises):
                #* Filter out superset premises
                is_redundant = any(p > q for q in non_redundant)
                if not is_redundant:
                    non_redundant.append(p)
                    consequentsmap[p].add(conseq_predicate)

        #* Format final rules in valid syntax
        arules = []
        for antecedents, consequents in tqdm(
            consequentsmap.items(),
            desc="Formatting rules",
            total=len(consequentsmap)
        ):
            #* Join antecedents and consequents
            premise = ' & '.join(antecedents)
            conclusion = ' & '.join(consequents)
            arule = f"({premise}) >> ({conclusion})"
            arules.append(arule)

        return arules
    