import gc
import math
import h2o
import psutil
from h2o.estimators import H2ORandomForestEstimator
from h2o.tree import H2OTree, H2OLeafNode, H2OSplitNode
from typing import *
import itertools
from collections import defaultdict
from time import perf_counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import sympy as sp
from rich import print as pprint
import lightgbm as lgb
from lightgbm import Booster, LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from anuta.constructor import Constructor, Cidds001
from anuta.theory import Theory
from anuta.known import *
from anuta.utils import log, true, false
from anuta.cli import FLAGS


class MemoryLimitExceeded(RuntimeError):
    """Raised when system memory crosses the configured threshold."""
    pass


def get_featuregroups(
    df: pd.DataFrame, 
    colvars: Dict[str, Set[str]],
    feature_marker: str='',
) -> Dict[str, List[Tuple[str, ...]]]:
    """Generate all feature groups for the given variables."""
    log.info(f"{feature_marker=}")
    log.info(f"{FLAGS.config.MAX_COMBO_SIZE=}")
    
    featuregroups = defaultdict(list)
    variables = list(df.columns)
    for target in tqdm(variables, desc="Generating feature groups"):
        # if feature_marker in target:
        #     continue
        #* Skip targets with only one unique value
        if df[target].nunique() <= 1:
            continue
        #* Features shouldn't contain overlapping varvars with the target
        features = [v for v in variables if 
                    not (colvars[v] & colvars[target]) and
                    feature_marker in v]
        #* In any case, include the full feature set.
        featuregroups[target].append(features)
            
        combo_size = min(FLAGS.config.MAX_COMBO_SIZE, len(features) - 1)
        # log.info(f"Max combo size for {target} is {combo_size}.")
        if combo_size <= 0:
            # No combinations to generate, skip further processing
            continue 
        # skipped = set()
        for n in range(1, combo_size+1):
            _featuregroup = [list(combo) for combo in itertools.combinations(features, n)]
            featuregroup = []
            for combo in _featuregroup:
                # if tuple(combo) in skipped:
                #     #* Skip already skipped combinations
                #     continue
                # if len(combo) < 3 and df[combo].drop_duplicates().shape[0] == 1:
                #     #* h2o will fail to train a tree on a feature group with single unique value,
                #     #*  but it's a costly operation to check, so only do it for small feature groups.
                #     skipped.add(tuple(combo))
                #     continue
                # else:
                
                # #* Check if the variables in the combo contain overlapping varvars
                # total_num_vars = sum(len(colvars[v]) for v in combo)
                # all_vars = set()
                # for col in combo:
                #     all_vars |= colvars[col]
                # if total_num_vars < len(all_vars):
                #     #* If the total number of variables in the combo is less than the number of unique varvars,
                #     #*  it means there are overlapping varvars, so skip this combo.
                #     log.info(f"Skipping combo {combo} for {target} due to overlapping varvars.")
                #     continue
                featuregroup.append(combo)
            featuregroups[target] += featuregroup
        # log.info(f"{target}: Skipped {nskiped} feature groups with single unique value.")
    return featuregroups

class TreeLearner(object):
    """Base class for tree learners."""
    def __init__(self, constructor: Constructor, limit=None):
        if limit and limit < constructor.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            constructor.df = constructor.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = 'all'
            
        self.dataset = constructor.label
        supported_datasets = ['cidds', 'yatesbury', 'metadc', 'netflix', 'mawi', 'ana']
        assert self.dataset in supported_datasets, \
            f"Unsupported dataset: {self.dataset}. Supported datasets: {supported_datasets}."
        self.examples = constructor.df.copy()
        self.examples[constructor.categoricals] = \
            self.examples[constructor.categoricals].astype('category')
        self.categoricals = constructor.categoricals
        
        self.variables: List[str] = [
            var for var in self.examples.columns
            # if var in self.categoricals
        ]
        self.features = [var for var in self.variables if constructor.feature_marker in var]
        self.featuregroups = get_featuregroups(
            self.examples, constructor.colvars, constructor.feature_marker)
        self.total_treegroups = len(self.examples.columns) * \
            len(list(self.featuregroups.values())[0])  # Total number of tree groups to learn
        
        self.prior = constructor.anuta.prior_kb
        self.memory_stop_threshold = getattr(FLAGS.config, 'MEMORY_STOP_THRESHOLD', 2)
    
    def learn(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _check_memory_budget(self):
        """Raise if the process memory usage exceeds the configured threshold."""
        if not self.memory_stop_threshold:
            return
        vm = psutil.virtual_memory()
        used_ratio = 1 - (vm.available / vm.total)
        if used_ratio >= self.memory_stop_threshold:
            raise MemoryLimitExceeded(
                f"{used_ratio:.1%} of RAM used (threshold {self.memory_stop_threshold:.1%}).")

class EntropyTreeLearner(TreeLearner):
    """Tree learner based on information gain, using H2O's implementation."""
    def __init__(self, constructor: Constructor, limit=None, shuffle=False):
        super().__init__(constructor, limit)
        h2o.init(max_mem_size=FLAGS.config.JVM_MEM, nthreads=-1)  # -1 = use all available cores
        h2o.no_progress()  # Disables all progress bar output
        
        # if self.dataset == 'mawi':
        #     #* Drop port and ip variables, as they are not useful for MAWI dataset.
        #     for col in self.examples.columns:
        #         name = col.lower()
        #         if 'port' in name or 'ipsrc' in name or 'ipdst' in name:
        #             print(f"Dropping column {col} for MAWI dataset.")
        #             self.examples.drop(columns=col, inplace=True)
        #             self.variables.remove(col)
        #             self.categoricals.remove(col)
        
        if shuffle:
            self.examples = self.examples.sample(frac=1, random_state=42).reset_index(drop=True)
        self.examples: h2o.H2OFrame = h2o.H2OFrame(constructor.df)
        if self.categoricals:
            self.examples[self.categoricals] = self.examples[self.categoricals].asfactor()

        self.model_configs = {}
        # num_examples = self.examples.shape[0]
        # min_rows = int(num_examples * 0.01) if num_examples > 100 else 1
        # log.info(f"Setting {min_rows=}.")
        self.model_configs['classification'] = dict(
            # model_id="clf_tree",
            ntrees=1,                 # Build only one tree
            max_depth=len(self.features),
            min_rows=1,               # Minimum number of observations in a leaf
            min_split_improvement=0, #* Maximize over-fitting for categorical variables
            sample_rate=1.0,          # Use all rows
            mtries=-2,                # Use all features (set to -2 for all features)
            seed=42,                  # For reproducibility
            categorical_encoding="Enum"  # Native handling of categorical features
        )
        self.model_configs['regression'] = dict(
            # model_id="reg_tree",
            ntrees=1,                 # Build only one tree
            max_depth=len(self.features)//2, #TODO: To be tuned
            min_rows=100,             #TODO: To be tuned (default value 10 from H2O)
            min_split_improvement=FLAGS.config.MIN_SPLIT_GAIN,
            sample_rate=1.0,          # Use all rows
            mtries=-2,                # Use all features (set to -2 for all features)
            seed=42,                  # For reproducibility
            categorical_encoding="Enum"  # Native handling of categorical features
        )
        
        self.domains: Dict[str, Iterable] = {}
        for varname in self.variables:
            if varname in self.categoricals: 
                self.domains[varname] = sorted(list(constructor.df[varname].unique()))
            else:
                self.domains[varname] = (
                    constructor.df[varname].min().item(), 
                    constructor.df[varname].max().item()
                )
        #* dTypes: {'int', 'real', 'enum'(categorical)}
        self.dtypes = {varname: t for varname, t in self.examples.types.items()}
        self.target_trees: Dict[str, List[H2ORandomForestEstimator]] = defaultdict(list)
        #* Start with prior rules
        self.learned_rules: Set[str] = set(self.prior)  
        #* {target: [{cls_idx1: tree1_paths}, {cls_idx2: tree2_paths}, ...]}
        self.target_treepaths: Dict[str, List[Dict[str|int, Dict[str, Any]]]] = {}
        #! Asssuming one treegroup per target variable!!!
        # self.target_unclassified_idxset: Dict[str, Set[int]] = {
        #     target: set(range(self.examples.nrows)) for target in self.categoricals}
        self.target_training_frame: Dict[str, h2o.H2OFrame] = {
            target: self.examples for target in self.variables
        }
        
        # # pprint(self.domains)
        # pprint(self.dtypes)
    
    def learn(self):
        log.info(f"{self.__class__.__name__}: Training {self.total_treegroups} tree groups.")
        log.info(f"{len(self.examples)} examples and {len(self.features)} features ({len(self.categoricals)} categorical vars).")
        
        max_sc_epochs = FLAGS.config.MAX_SEPARATE_CONQUER_EPOCHS
        if max_sc_epochs > 1:
            assert self.total_treegroups == self.examples.ncols,\
                "Separate-and-conquer currently supports one tree per target variable."
        epoch = 0
        new_rule_count = float('inf')
        new_rule_counts = []
        fully_calssified = []
        unclassified_counts = [self.examples.nrows*len(self.categoricals)]
        while epoch < max_sc_epochs and new_rule_count > 0:
            epoch += 1
            print(f"\tEpochs {epoch}/{max_sc_epochs} of separate-and-conquer.")
            
            start = perf_counter()
            treeid = 1
            for target, feature_group in self.featuregroups.items():
                if target in fully_calssified:
                    log.info(f"Skipping training for {target}, already fully classified.")
                    continue
                
                log.info(f"{target=}: {len(feature_group)} feature groups.")
                print(f"... Trained {treeid}/{self.total_treegroups} ({treeid/self.total_treegroups:.1%}) tree groups.", end='\r')
                if target in self.categoricals:
                    params = self.model_configs['classification']
                else:
                    assert '@' not in target, f"Abstract variable {target} cannot be a regression target."
                    if epoch > 1: 
                        #* No need to train regression trees in more epochs
                        continue
                    params = self.model_configs['regression']
                
                #! Assuming one treegroup per target variable, otherwise they'd share the same frame!
                training_frame = self.target_training_frame[target]
                try:
                    if len(training_frame[target].unique()) <= 1:
                        log.info(f"All examples for {target} have the target same value. Skipping.")
                        fully_calssified.append(target)
                        continue
                except Exception as e:
                    log.error(f"Failed to check unique values for {target}. Skipping.")
                    fully_calssified.append(target)
                    continue
                
                for i, features in enumerate(feature_group):                    
                    treeid += 1
                    #* Special symbols in model_id could cause issues when deleting related objects in H2O.
                    # model_id = f"'{target}_tree_{i+1}'"
                    model_id = f"tree{treeid}_featuregroup{i}"
                    params['model_id'] = model_id
                    dtree = H2ORandomForestEstimator(**params)
                    
                    try:
                        dtree.train(x=list(features), y=target, training_frame=training_frame)  
                        self.target_trees[target].append(dtree)
                    except Exception as e:
                        #* H2O lib could fail to train a tree (e.g., if all values are the same, frame too small, etc.)
                        log.warning(f"Couldn't train tree for {target} with {len(features)} features.")
                        self.target_trees[target].append(None)
                        
                    print(f"... Trained {treeid}/{self.total_treegroups} ({treeid/self.total_treegroups:.1%}) tree groups.", end='\r')
                    print(f"... Trained {treeid}/{self.total_treegroups} ({treeid/self.total_treegroups:.1%}) tree groups.", end='\r')
            end = perf_counter()
            training_time = end - start
            
            start = perf_counter()
            before_nrules = len(self.learned_rules)
            self.learned_rules |= self.extract_rules_from_treepaths()
            new_rule_count = len(self.learned_rules) - before_nrules
            end = perf_counter()
            extraction_time = end - start
            new_rule_counts.append(new_rule_count)
            if new_rule_count == 0:
                log.info(f"No new rules learned in epoch {epoch}. Stopping.")
                unclassified_counts.append(unclassified_counts[-1])
                break
            print()
            
            # total_unclassified = 0
            # for target in self.categoricals:
            #     if target in fully_calssified:
            #         log.info(f"Skipping {target}, already fully classified.")
            #         continue
            #     if target not in self.target_trees:
            #         log.warning(f"No trees trained for {target}. Skipping.")
            #         continue
                
            #     #& Training frame of the next epoch is the unclassified examples of this epoch.
            #     training_frame = self.get_unclassified_examples(target)
            #     self.target_training_frame[target] = training_frame
            #     num_unclassified = training_frame.nrows
            #     total_unclassified += num_unclassified
                
            #     if num_unclassified == 0:
            #         log.info(f"All examples for {target} are classified. Skipping.")
            #         fully_calssified.append(target)
            #         continue
            #     log.info(f"{num_unclassified/self.examples.nrows:.3%} unclassified examples for {target}.")
            
            # classified_more = unclassified_counts[-1] - total_unclassified
            # unclassified_counts.append(total_unclassified)
            
            #* Remove all H2O models to free memory
            for trees in self.target_trees.values():
                for model in trees:
                    if model is not None:
                        h2o.remove(model)
            self.target_trees.clear()
            # self.target_treepaths.clear()
            gc.collect()
            
            print(f"\tTraining {self.total_treegroups} tree groups took {training_time:.2f} seconds.")
            print(f"\tExtracting rules took {extraction_time:.2f} seconds.")
            print(f"\tFully classified {len(fully_calssified)} targets.")
            # print(f"\t{classified_more} more examples classified.")
            print(f"\tLearned {new_rule_count=} in epoch {epoch}.")
            print(f"\tTotal learned rules: {len(self.learned_rules)}.")
            print()
            
            # if classified_more == 0:
            #     log.info(f"No additional examples classified in epoch {epoch}. Stopping.")
            #     break

        log.info(f"Learning completed after {epoch} epochs.")
        # print(f"\t{fully_calssified=}")
        print(f"\t{new_rule_counts=}")
        print(f"\t{unclassified_counts=}")
        
        before_merge = len(self.learned_rules)
        self.learned_rules = self._merge_rules_by_premise(self.learned_rules)
        log.info(f"Merged learned rules: {before_merge} -> {len(self.learned_rules)}")
        
        assumptions = set(self.prior)
        rules = self.learned_rules | assumptions
        sprules = []
        
        for rule in tqdm(rules, desc="... Converting learned rules to sympy"):   
            if rule in (true, false):
                continue
            try:
                sprules.append(sp.sympify(rule))
            except Exception as e:
                log.error(f"Failed to sympify rule: {rule}. Skipping.")
                continue
        # sprules = [sp.sympify(rule) for rule in rules]
        # sprules = list(filter(lambda r: r not in (true, false), sprules))
        log.info(f"Total rules saved: {len(sprules)}")
        outputf = f'dt_{self.dataset}_{self.num_examples}_e{epoch}'
        if FLAGS.label:
            outputf += f"_{FLAGS.label}.pl"
        else:            
            outputf += ".pl"
        Theory.save_constraints(sprules, outputf)
        h2o.shutdown(prompt=False)
        return

    def classify(self, target: str) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Train a single classification tree to predict `target` from all other columns.
        Returns a tuple of (path_rules, metrics).
        """
        self.categoricals.append(target)
        
        if target not in self.examples.columns:
            raise ValueError(f"Unknown target column `{target}`.")

        feature_cols = [col for col in self.examples.columns if col != target]
        if not feature_cols:
            raise ValueError("No features available after removing the target column.")

        try:
            self.examples[target] = self.examples[target].asfactor()
        except Exception as exc:
            log.warning(f"Failed to convert {target} to categorical for classification: {exc}")

        train, test = self.examples.split_frame(ratios=[0.8], seed=42)
        if test.nrows == 0:
            log.warning("Test split is empty; using the training split for evaluation.")
            test = train

        params = dict(self.model_configs['classification'])
        params['model_id'] = f"classify_{target}"
        clf = H2ORandomForestEstimator(**params)
        
        log.info(f"Training classification tree with {len(feature_cols)} features and {train.nrows} training examples.")
        clf.train(x=feature_cols, y=target, training_frame=train)

        preds = clf.predict(test)
        pred_labels = preds['predict'].as_data_frame(use_multi_thread=True).iloc[:, 0].astype(str)
        true_labels = test[target].as_data_frame(use_multi_thread=True).iloc[:, 0].astype(str)

        classes = sorted(set(true_labels.tolist()) | set(pred_labels.tolist()))
        fpr: Dict[str, float] = {}
        tpr: Dict[str, float] = {}
        for cls in classes:
            tp = int(((pred_labels == cls) & (true_labels == cls)).sum())
            fp = int(((pred_labels == cls) & (true_labels != cls)).sum())
            tn = int(((pred_labels != cls) & (true_labels != cls)).sum())
            fn = int(((pred_labels != cls) & (true_labels == cls)).sum())
            tpr[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        accuracy = float((pred_labels == true_labels).mean()) if len(true_labels) else 0.0
        log.info(
            f"{target} classification metrics: accuracy={accuracy:.3f}, "
            f"TPR={tpr}, FPR={fpr} (train={train.nrows}, test={test.nrows})"
        )

        def _conditions_to_premise(conditions: List[str]) -> Optional[str]:
            merged: Dict[str, Dict[str, Any]] = {}
            for condition in conditions:
                varname, op, varval = condition.split('$')
                varval = eval(varval)
                bucket = merged.setdefault(varname, {})
                if op == '∈':
                    current = bucket.get(op, set(varval))
                    bucket[op] = current & set(varval)
                elif op == '∉':
                    current = bucket.get(op, set())
                    bucket[op] = current | set(varval)
                elif op == '>':
                    if self.dtypes.get(varname) == 'int':
                        varval = math.floor(varval)
                    current = bucket.get(op, float('-inf'))
                    bucket[op] = max(current, varval)
                elif op == '≤':
                    if self.dtypes.get(varname) == 'int':
                        varval = math.ceil(varval)
                    current = bucket.get(op, float('+inf'))
                    bucket[op] = min(current, varval)

            predicates: List[str] = []
            for varname, conds in merged.items():
                operators = conds.keys()
                if set(['∈', '∉']) & operators:
                    assert not set(['≤', '>']) & operators, f"{varname} has mixed {conds=}"
                    _invals = conds.get('∈', set())
                    _outvals = conds.get('∉', set())
                    invals = _invals - _outvals
                    outvals = _outvals - _invals

                    predicate = ''
                    for val in invals:
                        if '@' in varname:
                            avarname = varname.replace('@', '')
                            assert val in [0, 1], f"Abstract variable {varname} isn't binary."
                            predicate += f"{avarname}|" if val == 1 else f"Not({avarname})|"
                        else:
                            predicate += f"Eq({varname}, {val})|"
                    predicate = predicate[:-1]
                    if len(invals) > 1:
                        predicate = '( ' + predicate + ' )'
                    if predicate:
                        predicates.append(predicate)

                    for val in outvals:
                        if '@' in varname:
                            avarname = varname.replace('@', '')
                            assert val in [0, 1], f"Abstract variable {varname} isn't binary."
                            predicates.append(f"Not({avarname})" if val == 1 else f"{avarname}")
                        else:
                            predicates.append(f"Ne({varname}, {val})")
                else:
                    assert set(['≤', '>']) & operators, f"{varname} has no recognized {conds=}"
                    valmin = conds.get('>', float('-inf'))
                    valmax = conds.get('≤', float('+inf'))
                    if valmin > valmax:
                        log.warning(f"[Conflicting condition]: {varname} {conds}")
                        continue
                    if valmin == valmax:
                        if valmin == 0:
                            predicates.append(f"Eq({varname}, 0)")
                        else:
                            continue
                    else:
                        if valmin > float('-inf'):
                            predicates.append(f"({varname} > {valmin})")
                        if valmax < float('+inf'):
                            predicates.append(f"({varname} <= {valmax})")

            return ' & '.join(predicates) if predicates else None

        path_rules: Set[str] = set()
        tree_paths = self.extract_tree_paths(clf, target)
        for _, records in tree_paths.items():
            for record in records:
                premise = _conditions_to_premise(record['conditions'])
                if premise:
                    # path_rules.add(premise)
                    path_rules.add(f"Not({premise})")

        log.info(f"Extracted {len(path_rules)} classificatioin rules.")
        
        #* Save path rules 
        outputf = f'dt_{self.dataset}_classify_{self.num_examples}'
        if FLAGS.label:
            outputf += f"_{FLAGS.label}.pl"
        else:            
            outputf += ".pl"
        sprules = []
        learned_rules = set(self.prior) | path_rules
        for rule in tqdm(learned_rules, desc="... Converting path rules to sympy"):   
            try:
                sprules.append(sp.sympify(rule))
            except Exception as e:
                log.error(f"Failed to sympify rule: {rule}. Skipping.")
                continue
        Theory.save_constraints(sprules, outputf)
        
        h2o.shutdown(prompt=False)
        return path_rules, {'accuracy': accuracy, 'tpr': tpr, 'fpr': fpr}
    
    def _cleanup_tree_models(self, targets: Optional[Iterable[str]] = None):
        """Remove tracked H2O models to free JVM memory."""
        targets = list(targets) if targets is not None else list(self.target_trees.keys())
        for target in targets:
            trees = self.target_trees.pop(target, [])
            for model in trees:
                if model is None:
                    continue
                try:
                    h2o.remove(model)
                except Exception as exc:
                    log.warning(f"Failed to remove H2O model `{target}`: {exc}")
            self.target_treepaths.pop(target, None)
        gc.collect()

    def _merge_rules_by_premise(self, rules: Iterable[str]) -> Set[str]:
        """Combine rules with identical premises into a single rule with OR-ed conclusions."""
        grouped: Dict[str, Set[str]] = defaultdict(set)
        for rule in rules:
            if '>>' not in rule:
                # Skip malformed entries such as the prior sentinel values.
                continue
            premise, conclusion = rule.split('>>', 1)
            grouped[premise.strip()].add(conclusion.strip())

        merged_rules: Set[str] = set()
        for premise, conclusions in grouped.items():
            if len(conclusions) == 1:
                conclusion = next(iter(conclusions))
            else:
                conclusion = f"( {' | '.join(sorted(conclusions))} )"
            merged_rules.add(f"{premise} >> {conclusion}")
        return merged_rules
    
    def get_unclassified_examples(self, target: str) -> h2o.H2OFrame:
        trees: List[H2ORandomForestEstimator] = self.target_trees.get(target, None)
        assert len(trees) == 1, "Currently only supports one tree per target."
        #TODO: Support multiple trees per target
        dtree = trees[0]
        if dtree is None:
            log.warning(f"No tree trained for {target}. Skipping.")
            #* Return an empty H2OFrame
            return h2o.H2OFrame([])

        frame: h2o.H2OFrame = self.target_training_frame[target]
        n_rows = frame.nrows
        try:
            # #! Here, we should use ALL examples, not just the remaining ones.
            # leaf_assignments: h2o.H2OFrame = dtree.predict_leaf_node_assignment(self.examples, 'Node_ID')
            leaf_assignments: h2o.H2OFrame = dtree.predict_leaf_node_assignment(frame, 'Node_ID')
        except Exception as e:
            log.error(f"Failed to get leaf node assignments for {target}.")
            return h2o.H2OFrame([])
        leaf_df: pd.DataFrame = leaf_assignments.as_data_frame(use_multi_thread=True)

        treepaths = self.target_treepaths.get(target, [])
        recognized_any = np.zeros(n_rows, dtype=bool)
        #* Loop over each class's tree paths
        for pathconditions in treepaths:
            for idx, (cls_id, records) in enumerate(pathconditions.items()):
                #! There could be a mismatch between cls_id ([0, 1, 3, 4]) and the index used by H2O.
                leaf_col: str = leaf_assignments.columns[idx]
                #* pathid: {class_id}-{node_id} or {class_id}-{path_suffix}
                yes_leaf_ids = {int(rec['pathid'].split('-')[-1]) for rec in records}
                if not yes_leaf_ids:
                    continue

                #* Rows recognized by this class tree
                yes_rows = leaf_df.index[leaf_df[leaf_col].isin(yes_leaf_ids)]
                recognized_any[yes_rows] = True
        
        unclassified_mask = ~recognized_any
        remaining_idx = np.where(unclassified_mask)[0].tolist()
        
        # #* Take the intersect with the example indices of the previous epoch
        # training_idxset = self.target_unclassified_idxset[target]
        # unclassified_idx = [idx for idx in remaining_idx if idx in training_idxset]
        # unclassified_samples = self.examples[unclassified_idx, :]
        # #* Update the index set for the next epoch
        # self.target_unclassified_idxset[target] = set(unclassified_idx)
        
        unclassified_examples = frame[remaining_idx, :]
        h2o.remove(leaf_assignments)
        return unclassified_examples
        
    def extract_rules_from_treepaths(self) -> Set[str]:
        self.extract_target_treepaths()
        assert self.target_treepaths, "Target tree paths are empty. Did you train any trees?"
        
        learned_rules = set()
        for target, all_treepaths in self.target_treepaths.items():
            num_target_rules = 0
            for treeidx, pathconditions in enumerate(all_treepaths):
                if not pathconditions: 
                    #* No valid paths found in this tree
                    continue 
                
                if target in self.categoricals:
                    ruleset = defaultdict(set)
                else:
                    ruleset = defaultdict(dict)
                    '''Collect leaf ranges for regression trees'''
                    dtree: H2ORandomForestEstimator = self.target_trees[target][treeidx]
                    try:
                        leaf_assignments = dtree.predict_leaf_node_assignment(self.examples, 'Node_ID')# 'Path')
                    except Exception as e:
                        log.error(f"Failed to get leaf node assignments for {target}.")
                        continue
                    #* Bind leaf assignments with original target column
                    hf_leaf = self.examples.cbind(leaf_assignments)
                    leaf_col = leaf_assignments.columns[0]
                    
                    #* Group by leaf and compute min/max of the target variable
                    leaf_stats = hf_leaf.group_by(leaf_col).min(target).max(target).get_frame()
                    # leaf_stats.set_names(['leaf_id', 'leaf_min', 'leaf_max'])
                    leaf_stats_df = leaf_stats.as_data_frame(use_multi_thread=True)
                    leaf_id_col = leaf_stats_df.columns[0]      
                    min_col = leaf_stats_df.columns[1]          
                    max_col = leaf_stats_df.columns[2]          
                    leaf_ranges = {
                        row[leaf_id_col]: {'min': row[min_col], 'max': row[max_col]}
                        for _, row in leaf_stats_df.iterrows()
                    }
                    h2o.remove(leaf_assignments)
                    h2o.remove(hf_leaf)
                    h2o.remove(leaf_stats)

                for targetcls, records in pathconditions.items():
                    for record in records:
                        merged_conditions = {}
                        for condition in record['conditions']:
                            varname, op, varval = condition.split('$')
                            varval = eval(varval)
                            if varname not in merged_conditions:
                                merged_conditions[varname] = defaultdict(None)

                            if op == '∈':
                                values = merged_conditions[varname].get(op, set(varval))
                                merged_conditions[varname][op] = values & set(varval) 
                            elif op == '∉':
                                values = merged_conditions[varname].get(op, set())
                                merged_conditions[varname][op] = values | set(varval)
                            elif op == '>':  
                                if self.dtypes[varname] == 'int':
                                    varval = math.floor(varval)
                                value = merged_conditions[varname].get(op, float('-inf'))
                                merged_conditions[varname][op] = max(value, varval)
                            elif op == '≤':
                                if self.dtypes[varname] == 'int':
                                    varval = math.ceil(varval)
                                value = merged_conditions[varname].get(op, float('+inf'))
                                merged_conditions[varname][op] = min(value, varval)

                        predicates = []
                        for varname, conditions in merged_conditions.items():
                            operators = conditions.keys()
                            if set(['∈', '∉']) & operators:
                                #* Categorical var
                                assert not set(['≤', '>']) & operators, f"{varname} has mixed {conditions=}"
                                _invals = conditions.get('∈', set())
                                _outvals = conditions.get('∉', set())
                    
                                invals = _invals - _outvals
                                outvals = _outvals - _invals

                                #! Below introduces correctness issues!!!
                                # #* Use the most succinct representation
                                # domain = set(self.domains[varname])
                                # if invals:
                                #     diffvals = domain - invals
                                #     if len(diffvals) < len(invals):
                                #         outvals |= diffvals
                                #         invals = set()
                                # if outvals:
                                #     diffvals = domain - outvals
                                #     if len(diffvals) < len(outvals):
                                #         invals |= diffvals
                                #         outvals = set()
                                
                                predicate = ''
                                for val in invals:
                                    if '@' in varname:
                                        avarname = varname.replace('@', '')
                                        assert val in [0, 1], f"Abstract variable {varname} isn't binary."
                                        if val == 1:
                                            predicate += f"{avarname}|"
                                        else:
                                            predicate += f"Not({avarname})|"
                                    else:
                                        predicate += f"Eq({varname}, {val})|"
                                predicate = predicate[:-1]
                                if len(invals) > 1:
                                    predicate = '( ' + predicate + ' )'
                                if predicate:
                                    predicates.append(predicate)
                    
                                for val in outvals:
                                    if '@' in varname:
                                        avarname = varname.replace('@', '')
                                        assert val in [0, 1], f"Abstract variable {varname} isn't binary."
                                        if val == 1:
                                            predicates.append(f"Not({avarname})")
                                        else:
                                            predicates.append(f"{avarname}")
                                    else:
                                        predicates.append(f"Ne({varname}, {val})")
                            else:
                                assert set(['≤', '>']) & operators, f"{varname} has no recognizd {conditions=}"
                                valmin = conditions.get('>', float('-inf'))
                                valmax = conditions.get('≤', float('+inf'))
                                if valmin > valmax:
                                    #TODO: Accumulate logits to decide which condition to take. Discard all together for now.
                                    print(f"[Conflicting condition!!!]: {varname=}:{conditions}")
                                if valmin == valmax:
                                    if valmin == 0:
                                        predicate = f"Eq({varname}, 0)"
                                    else:
                                        #* Too strict condition, skip it
                                        # predicates.append(f"Eq({varname}, {valmin})")
                                        # log.warning(
                                        #     f"Skipping overly strict rule for {target} ({varname}={valmin}).")
                                        continue
                                if valmin > float('-inf'):
                                    predicates.append(f"({varname} > {valmin})")
                                if valmax < float('+inf'):
                                    predicates.append(f"({varname} <= {valmax})")

                        if predicates:
                            premise = ' & '.join(predicates)
                            assert '@' not in premise, \
                                f"Premise {premise} contains abstract variables."
                            if target in self.categoricals:
                                if '@' in target:
                                    #* Abstract variable
                                    atarget = target.replace('@', '')
                                    assert targetcls in [0, 1], f"Abstract {target=} isn't binary."
                                    conclusion = f"{atarget}" if targetcls == 1 else f"Not({atarget})"
                                else:
                                    conclusion = f"Eq({target}, {targetcls})"
                                ruleset[premise].add(conclusion)
                            else:
                                leafid = record['pathid'].split('-')[-1]
                                #* Int ID when `Node_ID` is used, or string (like "RRL") when `Path` is used
                                leafid = int(leafid) if leafid.isdigit() else leafid
                                if leafid not in leaf_ranges:
                                    #* There could be empty splits in the tree 
                                    #*  (e.g., `min_rows=2` and both examples are classified to the child)
                                    log.warning(f"{leafid=} not found in `leaf_ranges` for {target}.")
                                else:
                                    leafmin = leaf_ranges[leafid]['min']
                                    leafmax = leaf_ranges[leafid]['max']
                                    ruleset[premise]['min'] = min(
                                        ruleset[premise].get('min', float('+inf')), 
                                        leafmin
                                    )
                                    ruleset[premise]['max'] = max(
                                        ruleset[premise].get('max', float('-inf')), 
                                        leafmax
                                    )

                rules = set()
                for premise, conclusions in ruleset.items():
                    if target in self.categoricals:
                        conclusion = '( ' + '|'.join(conclusions) + ' )' \
                            if len(conclusions) > 1 else conclusions.pop()
                    else:
                        targetmin = ruleset[premise]['min']
                        targetmax = ruleset[premise]['max']
                        if targetmin == targetmax:
                            if targetmin == 0:
                                conclusion = f"Eq({target}, 0)"
                            else:
                                #* Too strict condition, skip it
                                # log.warning(
                                #     f"Skipping overly strict rule for {target} (min==max: {targetmin}).")
                                continue
                        else:
                            conclusion = f"(({target}>={targetmin}) & ({target}<={targetmax}))"
                        # \ if targetmin != targetmax else f"Eq({target}, {targetmin})"
                    
                    rule = f"({premise}) >> {conclusion}"
                    rules.add(rule)
                num_target_rules += len(rules)
                learned_rules |= rules
            log.info(f"Extracted {num_target_rules} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules
    
    def extract_target_treepaths(self):
        """Extract path conditions from all trees."""
        target_treepaths = defaultdict(list)
        for target, trees in self.target_trees.items():
            for treeidx, dtree in enumerate(tqdm(
                trees, desc=f"Extracting tree paths for {target}"
            )):
                # print(f"Features: {self.featuregroups[target][treeidx]}")
                if dtree is None:
                    #* Skip trees that failed to train or had no remaining examples.
                    target_treepaths[target].append({})
                    continue
                paths = self.extract_tree_paths(dtree, target)
                #* Paths could be empty `{}`, but keep it 
                #*  to match the indexing of `self.trees` to `all_tree_paths`
                target_treepaths[target].append(paths)

        #* {target: [{tree1_paths}, {tree2_paths}, ...]}
        self.target_treepaths = target_treepaths
        return
    

    def extract_tree_paths(self, dtree: H2ORandomForestEstimator, target):
        MIN_LOGIT = 1-1e-6 if target in self.categoricals else 0
        treeinfo = dtree._model_json['output']['model_summary']
        assert treeinfo['number_of_trees'][0] == 1, f"H2O random forest has {treeinfo['number_of_trees']}>1 tree."
        
        logits = set()
        def recurse(node: H2OLeafNode|H2OSplitNode, path, path_suffix, cls_idex):
            if node.__class__.__name__ == 'H2OLeafNode':
                leaf_id = path_suffix or '0'
                #* Final value is a probability (unlike boosting trees that use logits). 
                #* Only take pure leaves
                logits.add(node.prediction)
                if node.prediction > MIN_LOGIT:
                    paths.append({
                        'pathid': f"{cls_idex}-{node.id}", # f"{cls_idex}-{leaf_id}",
                        'logit': node.prediction,
                        'conditions': path.copy()
                    })
                return

            varname = node.split_feature

            # Handle categorical splits
            if node.left_levels or node.right_levels:
                left_categories = sorted([int(v) for v in node.left_levels])
                right_categories = sorted([int(v) for v in node.right_levels])
                # print(f"{varname=} categorical split: ")
                # print(f"  Left levels: {left_categories}")
                # print(f"  Right levels: {right_categories}")
                if node.left_levels:
                    cond_left = f"{varname}$∈${left_categories}"
                    cond_right = f"{varname}$∉${left_categories}"
                else:
                    cond_left = f"{varname}$∉${right_categories}"
                    cond_right = f"{varname}$∈${right_categories}"
            else:
                # Numeric split
                cond_left = f"{varname}$≤${node.threshold}"
                cond_right = f"{varname}$>${node.threshold}"

            #* Resulting path IDs: 0-LRRL
            recurse(node.left_child, path + [cond_left], path_suffix + "L", cls_idex)
            recurse(node.right_child, path + [cond_right], path_suffix + "R", cls_idex)

        treepaths = {}
        tree_classes = dtree._model_json['output']['domains'][-1]
        # print(f"{target=} {tree_classes=}, variables={dtree._model_json['output']['names']}")
        if tree_classes is not None and len(tree_classes) > 2:
            #* Multi-class classification tree
            for targetcls in tree_classes:
                #! Assume always use one tree (`tree_number=0`) with RF
                paths = []
                try:
                    htree = H2OTree(model=dtree, tree_number=0, tree_class=targetcls)
                    recurse(htree.root_node, [], "", targetcls)
                except Exception as e:
                    log.error(f"Failed to extract paths from tree ({target=} {targetcls=})")
                if paths:
                    treepaths[targetcls] = paths
        else:
            '''Binomial or regression tree'''
            paths = []
            #* For binary classification, H2O predicts class 0 logits by default.
            targetcls = 0
            try:
                htree = H2OTree(model=dtree, tree_number=0)
                recurse(htree.root_node, [], "", targetcls)
            except Exception as e:
                log.error(f"Failed to extract paths from tree ({target=})")
            if paths:
                treepaths[targetcls] = paths
        
        # print(logits)
        return treepaths


class XgboostTreeLearner(TreeLearner):
    """Tree learner based on XGBoost."""
    def __init__(self, constructor: Constructor, limit=None):
        super().__init__(constructor, limit)
        
        common_config = dict(
                min_child_weight=0,    # Leaf's minimum sum of Hessian
                gamma=FLAGS.config.MIN_SPLIT_GAIN,
                grow_policy='depthwise',  # ensures full-depth growth
                # tree_method='exact',   # for most deterministic behavior
                # # subsample=1,
                # # colsample_bytree=1,
                learning_rate=1,        # set high so pure leaves dominate logits
                n_estimators=1, 
                reg_alpha=0,   # L1 regularization term on weights
                reg_lambda=1,   # L2 regularization term on weights
                enable_categorical=True)
        self.model_configs = {}
        self.model_configs['classification'] = dict(
            objective = 'multi:softprob',
            max_depth=len(self.features), # high enough to split until pure
            **common_config,
        )
        self.model_configs['regression'] = dict(
            objective = 'reg:squarederror',
            max_depth=len(self.features)//2, #TODO: To be tuned
            **common_config,
        )
                
        #TODO: Unify `Domain`
        self.domains = {}
        for varname in self.variables:
            if varname in self.categoricals: 
                self.domains[varname] = sorted(
                    [n.item() for n in constructor.df[varname].unique()])
            else:
                self.domains[varname] = (
                    self.examples[varname].min().item(), 
                    self.examples[varname].max().item(),
                )
        self.dtypes = {}
        #TODO: Unify `DomainType`
        #* dTypes: {'int', 'real', 'enum'(categorical)}
        for varname, dtype in self.examples.dtypes.items():
            if dtype.name == 'category':
                self.dtypes[varname] = 'enum'
            elif dtype.name in ['int64', 'int32']:
                self.dtypes[varname] = 'int'
            elif dtype.name in ['float64', 'float32']:
                self.dtypes[varname] = 'real'
            else:
                raise ValueError(f"Unsupported data type {dtype} for variable {varname}.")
        self.trees: Dict[str, List[XGBClassifier|XGBRegressor]] = defaultdict(list)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.learned_rules: Set[str] = set()
        # pprint(self.domains)
        pprint(self.dtypes)
    
    def learn(self):
        log.info(f"{self.__class__.__name__}: Training {self.total_treegroups} tree groups from {len(self.examples)}"
                 f" examples and {len(self.features)} features ({len(self.categoricals)} categorical vars).")
        
        start = perf_counter()
        treeid = 1
        for target, feature_group in self.featuregroups.items():
            log.info(f"Learning trees for {target} with {len(feature_group)} feature groups.")
            modelcls = None
            if target in self.categoricals:
                params = self.model_configs['classification']
                modelcls = XGBClassifier
            else:
                params = self.model_configs['regression']
                modelcls = XGBRegressor
            num_class = len(self.domains[target]) if target in self.categoricals else 1
            
            y = self.examples[target]
            if target in self.categoricals:
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                self.label_encoders[target] = encoder
            
            for i, features in enumerate(feature_group):
                model = modelcls(num_class=num_class, **params)
                X = self.examples[list(features)]
                model.fit(X, y)
                self.trees[target].append(model)
                print(f"... Trained {treeid}/{self.total_treegroups} ({treeid/self.total_treegroups:.1%}) tree groups ({target=}).", end='\r')
                treeid += 1
        end = perf_counter()
        log.info(f"Training {self.total_treegroups} tree groups took {end - start:.2f} seconds.")
        
        start = perf_counter()
        self.learned_rules = self.extract_rules_from_pathconditions()
        end = perf_counter()
        log.info(f"Learned {len(self.learned_rules)} rules from {self.total_treegroups} trees.")
        log.info(f"Extracting rules took {end - start:.2f} seconds.")
        
        assumptions = set()
        for varname, domain in self.domains.items():
            if varname in self.categoricals:
                assumptions.add(f"{varname} >= 0")
                assumptions.add(f"{varname} <= {max(domain)}")
            else:
                assumptions.add(f"{varname} >= {domain[0]}")
                assumptions.add(f"{varname} <= {domain[1]}")
        rules = self.learned_rules | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        sprules = list(filter(lambda r: r != sp.true, sprules))  # Remove trivial rules
        Theory.save_constraints(sprules, f'xgb_{self.dataset}_{self.num_examples}.pl')
        
        return
    
    def extract_rules_from_pathconditions(self) -> Set[str]:
        learned_rules = set()
        for target, pathconditions in self.extract_conditions_from_treepaths().items():
            targetvar = target
            rules = set()
            for i, (targetcls, conditions) in enumerate(pathconditions.items()):
                for record in conditions:
                    var_conditions = {}
                    for condition in record['conditions']:
                        varname, op, varval = condition.split('$')
                        varval = eval(varval)
                        if varname not in var_conditions:
                            var_conditions[varname] = defaultdict(None)

                        if op == '∈':
                            values = var_conditions[varname].get(op, set(varval))
                            var_conditions[varname][op] = values & set(varval) 
                        elif op == '∉':
                            values = var_conditions[varname].get(op, set())
                            var_conditions[varname][op] = values | set(varval)
                        elif op == '≥':
                            value = var_conditions[varname].get(op, float('-inf'))
                            # if value != float('-inf') and value < 0:
                            #     #! Ignore such conditions? How does this happen?
                            #     log.error(
                            #         f"Invalid value for {varname} with {op}: {value}")
                            #     #! Export an image of the tree to debug
                            #     from xgboost import to_graphviz
                            #     xgbmodel = self.trees[target][i]
                            #     dot = to_graphviz(xgbmodel, tree_idx=0)
                            #     dot.render("xgb_tree", format="png", cleanup=True) 
                            var_conditions[varname][op] = max(value, varval)
                        elif op == '<':
                            value = var_conditions[varname].get(op, float('+inf'))
                            # if value < 0:
                            #     log.error(
                            #         f"Invalid value for {varname} with {op}: {value}")
                            #     #! Export an image of the tree to debug
                            #     from xgboost import to_graphviz
                            #     xgbmodel = self.trees[target][i]
                            #     dot = to_graphviz(xgbmodel, tree_idx=0)
                            #     dot.render("xgb_tree", format="png", cleanup=True)
                            var_conditions[varname][op] = min(value, varval)

                    predicates = []
                    for varname, merged_conditions in var_conditions.items():
                        operators = merged_conditions.keys()
                        if set(['∈', '∉']) & operators:
                            #* Categorical var
                            assert not set(['<', '≥']) & operators, f"{varname} has mixed {merged_conditions=}"
                            _invals = merged_conditions.get('∈', set())
                            _outvals = merged_conditions.get('∉', set())
                
                            invals = _invals - _outvals
                            outvals = _outvals - _invals
                            domain = set(self.domains[varname])

                            # #* Use the most succinct representation
                            # if invals:
                            #     diffvals = domain - invals
                            #     if len(diffvals) < len(invals):
                            #         outvals |= diffvals
                            #         invals = set()
                            # if outvals:
                            #     diffvals = domain - outvals
                            #     if len(diffvals) < len(outvals):
                            #         invals |= diffvals
                            #         outvals = set()

                            predicate = ''
                            for val in invals:
                                predicate += f"Eq({varname}, {val})|"
                            predicate = predicate[:-1]
                            if len(invals) > 1:
                                predicate = '( ' + predicate + ' )'
                            if predicate:
                                predicates.append(predicate)

                            for val in outvals:
                                predicates.append(f"Ne({varname}, {val})")
                        else:
                            assert set(['<', '≥']) & operators, f"{varname} has no recognizd {merged_conditions=}"
                            valmin = merged_conditions.get('≥', float('-inf'))
                            valmax = merged_conditions.get('<', float('+inf'))
                            # assert valmin <= valmax, f"[Conflicting condition!!!]: {varname=}:{merged_conditions}"
                            if valmin > valmax:
                                log.error(
                                    f"[Conflicting condition!!!]: {varname=}:{merged_conditions}")
                                #! Export an image of the tree to debug
                                from xgboost import to_graphviz
                                xgbmodel = self.trees[target][i]
                                dot = to_graphviz(xgbmodel, tree_idx=0)
                                dot.render("xgb_tree_conflict", format="png", cleanup=True)
                            if valmin == valmax:
                                log.warning(
                                    f"Skipping overly strict rule for {target} ({varname}={valmin}).")
                                continue
                            if valmin > float('-inf'):
                                valmin = math.floor(valmin) if self.dtypes[varname] == 'int' else valmin
                                predicates.append(f"({varname} >= {valmin})")
                            if valmax < float('+inf'):
                                valmax = math.ceil(valmax) if self.dtypes[varname] == 'int' else valmax
                                predicates.append(f"({varname} < {valmax})")
                    
                    if predicates:
                        premise = ' & '.join(predicates)
                        if targetvar in self.categoricals:
                            conclusion = f"Eq({targetvar}, {targetcls})"
                        else:
                            targetmin = record['target_range']['min']
                            targetmax = record['target_range']['max']
                            if self.dtypes[targetvar] == 'int':
                                targetmin, targetmax = math.floor(targetmin), math.ceil(targetmax)
                            if targetmin == targetmax:
                                #* Too strict condition, skip it
                                # log.warning(
                                #     f"Skipping overly strict rule for {targetvar} (min==max: {targetmin}).")
                                continue
                            
                            conclusion = f"(({targetvar}>={targetmin}) & ({targetvar}<={targetmax}))"
                        rule = f"({premise}) >> {conclusion}"
                        rules.add(rule)
            learned_rules |= rules
            log.info(f"Extracted {len(rules)} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules

    def extract_conditions_from_treepaths(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        #TODO: Move to config
        MIN_LOGIT = 0
        #* {target: {label1: [conditions1, conditions2, ...], label2: [...]}}
        all_pathconditions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for target, all_treepaths in self.extract_paths_from_all_trees().items():
            all_pathconditions[target] = defaultdict(list)
            encoder = self.label_encoders.get(target, None)
            for treeidx, paths in enumerate(all_treepaths):
                xgbtree: XGBClassifier|XGBRegressor = self.trees[target][treeidx]
                n_classes = getattr(xgbtree, 'n_classes_', 1)
                useless_splits = [] # XgboostTreeLearner.get_useless_splits(xgbtree)
                label_map = dict(zip(encoder.transform(encoder.classes_), encoder.classes_)) \
                    if encoder else {}
                
                leafranges = {}  
                if not encoder:
                    #* Compute leaf ranges for regression trees
                    features = xgbtree.get_booster().feature_names
                    X = self.examples.copy()
                    #! Deal with bug in XGBoost3.0.2 where categoricals are not handled correctly at inference
                    X[self.categoricals] = X[self.categoricals].astype(int)
                    X = X[features]
                    leaves = xgbtree.apply(X)
                    # Flatten since we have only one tree
                    leaf_indices = leaves.flatten()
                    # Combine with target values
                    df = pd.DataFrame({'leaf': leaf_indices, 'target': self.examples[target]})
                    # Group by leaf and get min/max
                    leafdf = df.groupby('leaf')['target'].agg(['min', 'max', 'count']).reset_index()
                    leafdf.reset_index(inplace=True)
                    for i, row in leafdf.iterrows():
                        #TODO: Filter by min count in a leaf?
                        leafranges[int(row['leaf'])] = {
                            'min': row['min'],
                            'max': row['max'],
                        }

                #* Group paths by target class
                for path in paths:
                    if path['Logit'] <= MIN_LOGIT: continue

                    pathcondition = {
                        'pathid': path['LeafID'],
                        'logit': path['Logit'],
                        'target_range': None,  # For regression trees
                    }
                    conditions = []
                    for split in path['Path']:
                        # if split['Gain'] <= FLAGS.config.MIN_SPLIT_GAIN: continue
                        if split['NodeID'] in useless_splits: continue

                        varname = split['Feature']
                        condition = None
                        if split['Split'] is not None:
                            if int(split['Split']) < 0:
                                #* -2147483648 is reserved to represent integer NA in R (xgb.DMatrix), 
                                #*  assuming non-negative values.
                                assert int(split['Split']) == -2147483648, \
                                    f"Unexpected split value {split['Split']} for {varname}."
                                # log.warning(f"Encountered NA split (-2147483648) for {varname}.")
                            else:
                                op = '<' if split['Direction']=='Yes' else '≥'
                                condition = f"{varname}${op}${split['Split']}"
                        else:
                            assert split['Category'] is not None
                            op = '∈' if split['Direction'] == 'Yes' else '∉'
                            categories = [
                                #* Index into the original category
                                self.examples[varname].astype('category').cat.categories[int(v)] 
                                for v in split['Category']
                            ]
                            condition = f"{varname}${op}${categories}"
                        if condition is not None:
                            conditions.append(condition)

                    if conditions:
                        pathcondition['conditions'] = conditions
                        if n_classes > 1:
                            target_cls = path['Tree'] % n_classes
                            label = label_map[target_cls]
                            all_pathconditions[target][label].append(pathcondition)
                        else:
                            #* Regression tree
                            #* [0] is the tree index, [1] is the leaf index
                            leafid = int(path['LeafID'].split('-')[-1])
                            if leafid in leafranges:
                                pathcondition['target_range'] = leafranges[leafid]
                                #! Should NOT mix leafids across trees
                                label = f"{treeidx}-{leafid}"
                                all_pathconditions[target][label].append(pathcondition)
        return all_pathconditions
                        
    @staticmethod
    def get_useless_splits(model: XGBClassifier|XGBRegressor) -> Set[int]:
        """Find the split whose two leaves have the same logits."""
        tree_df = model.get_booster().trees_to_dataframe()
        useless_splits = []
        for _, node in tree_df[tree_df['Feature'] != 'Leaf'].iterrows():
            yes_leaf = tree_df[(tree_df['Tree'] == node['Tree']) & (tree_df['ID'] == node['Yes'])]
            no_leaf = tree_df[(tree_df['Tree'] == node['Tree']) & (tree_df['ID'] == node['No'])]
        
            if not yes_leaf.empty and not no_leaf.empty:
                if yes_leaf.iloc[0]['Gain'] == no_leaf.iloc[0]['Gain']:
                    useless_splits.append( node['ID'] )
                    # useless_splits.append((node['Tree'], node['ID'], node['Feature']))
        return useless_splits
    
    def extract_paths_from_all_trees(self):
        """Extract path conditions from all trees."""
        all_tree_paths = defaultdict(list)
        for target, trees in self.trees.items():
            for treeidx, model in enumerate(tqdm(
                trees, desc=f"Extracting tree paths for {target}"
            )):
                paths = self.extract_tree_paths(model)
                #* Paths could be empty `{}`, but keep it 
                #*  to match the indexing of `self.trees` to `all_tree_paths`
                all_tree_paths[target].append(paths)

        #* {target: [{tree1_paths}, {tree2_paths}, ...]}
        return all_tree_paths
    
    def extract_tree_paths(self, model: XGBClassifier|XGBRegressor):
        """
        Extracts decision paths from the last N trees in an XGBoost model, including categorical splits.
        N is the number of classes for classifiers, and 1 for regressors.

        Parameters:
        - model: Trained XGBoost model (e.g., XGBClassifier or XGBRegressor)

        Returns:
        - List of dictionaries, each representing a decision path from root to leaf.
        """
        booster = model.get_booster()
        tree_df = booster.trees_to_dataframe()

        # Preprocess node mapping per tree
        tree_group = tree_df.groupby('Tree')
        id_to_row_by_tree = {
            tree_idx: group.set_index('ID').to_dict(orient='index')
            for tree_idx, group in tree_group
        }

        leaf_nodes = tree_df[tree_df['Feature'] == 'Leaf']
        paths = []

        for _, leaf in leaf_nodes.iterrows():
            tree_index = leaf['Tree']
            leaf_id = leaf['ID']
            leaf_value = leaf['Gain']  # Leaf value

            id_to_row = id_to_row_by_tree[tree_index]
            path = []
            current_id = leaf_id
            visited = set()

            while True:
                if current_id in visited:
                    print(f"⚠️ Infinite loop detected at node {current_id} in tree {tree_index}")
                    break
                visited.add(current_id)

                # Find parent node
                parent_id = next(
                    (pid for pid, row in id_to_row.items()
                    if row.get('Yes') == current_id or row.get('No') == current_id),
                    None
                )

                if parent_id is None:
                    break  # Reached root

                parent = id_to_row[parent_id]
                direction = 'Yes' if parent['Yes'] == current_id else 'No'

                condition = {
                    'Tree': tree_index,
                    'NodeID': parent_id,
                    'Feature': parent['Feature'],
                    'Split': parent['Split'] if not np.isnan(parent['Split']) else None,
                    'Gain': parent['Gain'],
                    'Category': parent.get('Category'),
                    'Direction': direction
                }
                path.append(condition)
                current_id = parent_id

            path.reverse()

            paths.append({
                'Tree': tree_index,
                'LeafID': leaf_id,
                'Logit': leaf_value,
                'Path': path
            })

        return paths
    
class LightGbmTreeLearner(TreeLearner):
    """Tree learner based on LightGBM."""
    def __init__(self, constructor: Constructor, limit=None):
        super().__init__(constructor, limit)
        
        common_config = {
            'n_estimators': 1,  #* Force one tree per feature group (no boosting)
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,         # or 1e-6 to stabilize
            'learning_rate': 1,
            'verbose': -1,
            
            'feature_fraction': 1.0,  # use all features
            'feature_fraction_bynode': 1.0,  # use all features in each node
            # 'min_data_in_leaf': 1,        # allow small leaves
            # 'min_sum_hessian_in_leaf': 1e-10,  # loosen constraints
            'boosting_type': 'gbdt',
            'force_col_wise': True,        # deterministic feature ordering
            # 'categorical_feature': 'auto', # f"name:{','.join(self.categoricals)}",  
        }
        self.model_configs = {}
        self.model_configs['classification'] = dict(
            objective='multiclass',
            metric='multi_logloss',
            min_data_in_leaf=1,  # Minimum # of examples that must fall into a tree node for it to be added.
            max_depth=len(self.features), # high enough to split until pure
            min_split_gain=FLAGS.config.MIN_SPLIT_GAIN,
            **common_config,
        )
        self.model_configs['regression'] = dict(
            objective='regression',
            metric='l2',
            max_depth=len(self.features)//2, #TODO: To be tuned
            min_split_gain=FLAGS.config.MIN_SPLIT_GAIN,
            **common_config,
        )
        
        self.domains = {}
        for varname in self.variables:
            if varname in self.categoricals: 
                self.domains[varname] = sorted(
                    [n.item() for n in constructor.df[varname].unique()])
            else:
                self.domains[varname] = (
                    self.examples[varname].min().item(), 
                    self.examples[varname].max().item(),
                )
        self.dtypes = {}
        #TODO: Unify `DomainType`
        #* dTypes: {'int', 'real', 'enum'(categorical)}
        for varname, dtype in self.examples.dtypes.items():
            if dtype.name == 'category':
                self.dtypes[varname] = 'enum'
            elif dtype.name in ['int64', 'int32']:
                self.dtypes[varname] = 'int'
            elif dtype.name in ['float64', 'float32']:
                self.dtypes[varname] = 'real'
            else:
                raise ValueError(f"Unsupported data type {dtype} for variable {varname}.")
        self.trees: Dict[str, List[Booster]] = defaultdict(list)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.learned_rules: Set[str] = set()
        # # pprint(self.domains)
        pprint(self.dtypes)
        
    def learn(self):
        log.info(f"{self.__class__.__name__}: Training {self.total_treegroups} tree groups from {len(self.examples)}"
                 f" examples and {len(self.features)} features ({len(self.categoricals)} categorical vars).")
        
        start = perf_counter()
        treeid = 1
        for target, feature_group in self.featuregroups.items():
            log.info(f"Learning trees for {target} with {len(feature_group)} feature groups.")
            # modelcls = None
            if target in self.categoricals:
                params = self.model_configs['classification']
                # modelcls = LGBMClassifier
            else:
                params = self.model_configs['regression']
                # modelcls = LGBMRegressor
            num_class = len(self.domains[target]) if target in self.categoricals else 1
            params['num_class'] = num_class
            
            y = self.examples[target]
            if target in self.categoricals:
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                self.label_encoders[target] = encoder
            
            for i, features in enumerate(feature_group):
                X = self.examples[list(features)]
                categorical_features = [col for col in X.columns if col in self.categoricals]
                lgb_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features)
                model = lgb.train(params, lgb_data, num_boost_epoch=1)
                self.trees[target].append(model)
                print(f"... Trained {treeid}/{self.total_treegroups} ({treeid/self.total_treegroups:.1%}) tree groups ({target=}).", end='\r')
                treeid += 1
        end = perf_counter()
        log.info(f"Training {self.total_treegroups} tree groups took {end - start:.2f} seconds.")
        start = perf_counter()
        
        start = perf_counter()
        self.learned_rules = self.extract_rules_from_pathconditions()
        end = perf_counter()
        log.info(f"Learned {len(self.learned_rules)} rules from {self.total_treegroups} trees.")
        log.info(f"Extracting rules took {end - start:.2f} seconds.")
        
        assumptions = set()
        for varname, domain in self.domains.items():
            if varname in self.categoricals:
                assumptions.add(f"{varname} >= 0")
                assumptions.add(f"{varname} <= {max(domain)}")
            else:
                assumptions.add(f"{varname} >= {domain[0]}")
                assumptions.add(f"{varname} <= {domain[1]}")
        rules = self.learned_rules | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        Theory.save_constraints(sprules, f'lgbm_{self.dataset}_{self.num_examples}.pl')
        
        return
        
        
    
    def extract_rules_from_pathconditions(self) -> Set[str]:
        #TODO: Merge with XGBoost's implementation 
        #TODO: (only difference is the operators ['≤', '>'] and their corresponding predicates)
        learned_rules = set()
        for target, pathconditions in self.extract_conditions_from_all_trees().items():
            targetvar = target
            rules = set()
            for targetcls, conditions in pathconditions.items():
                for record in conditions:
                    var_conditions = {}
                    for condition in record['conditions']:
                        varname, op, varval = condition.split('$')
                        varval = eval(varval)
                        if varname not in var_conditions:
                            var_conditions[varname] = defaultdict(None)

                        if op == '∈':
                            values = var_conditions[varname].get(op, set(varval))
                            var_conditions[varname][op] = values & set(varval) 
                        elif op == '∉':
                            values = var_conditions[varname].get(op, set())
                            var_conditions[varname][op] = values | set(varval)
                        elif op == '>':
                            value = var_conditions[varname].get(op, float('-inf'))
                            var_conditions[varname][op] = max(value, varval)
                        elif op == '≤':
                            value = var_conditions[varname].get(op, float('+inf'))
                            var_conditions[varname][op] = min(value, varval)

                    predicates = []
                    for varname, merged_conditions in var_conditions.items():
                        operators = merged_conditions.keys()
                        if set(['∈', '∉']) & operators:
                            #* Categorical var
                            assert not set(['≤', '>']) & operators, f"{varname} has mixed {merged_conditions=}"
                            _invals = merged_conditions.get('∈', set())
                            _outvals = merged_conditions.get('∉', set())
                
                            invals = _invals - _outvals
                            outvals = _outvals - _invals
                            domain = set(self.domains[varname])

                            #* Use the most succinct representation
                            if invals:
                                diffvals = domain - invals
                                if len(diffvals) < len(invals):
                                    outvals |= diffvals
                                    invals = set()
                            if outvals:
                                diffvals = domain - outvals
                                if len(diffvals) < len(outvals):
                                    invals |= diffvals
                                    outvals = set()

                            predicate = ''
                            for val in invals:
                                predicate += f"Eq({varname}, {val})|"
                            predicate = predicate[:-1]
                            if len(invals) > 1:
                                predicate = '( ' + predicate + ' )'
                            if predicate:
                                predicates.append(predicate)

                            for val in outvals:
                                predicates.append(f"Ne({varname}, {val})")
                        else:
                            assert set(['≤', '>']) & operators, f"{varname} has no recognizd {merged_conditions=}"
                            valmin = merged_conditions.get('>', float('-inf'))
                            valmax = merged_conditions.get('≤', float('+inf'))
                            assert valmin <= valmax, f"[Conflicting condition!!!]: {varname=}:{merged_conditions}"
                            
                            if valmin == valmax:
                                log.warning(
                                    f"Skipping overly strict rule for {target} ({varname}={valmin}).")
                                continue
                            if valmin > float('-inf'):
                                valmin = math.floor(valmin) if self.dtypes[varname] == 'int' else valmin
                                predicates.append(f"({varname} > {valmin})")
                            if valmax < float('+inf'):
                                valmax = math.ceil(valmax) if self.dtypes[varname] == 'int' else valmax
                                predicates.append(f"({varname} <= {valmax})")
                    
                    if predicates:
                        premise = ' & '.join(predicates)
                        if targetvar in self.categoricals:
                            conclusion = f"Eq({targetvar}, {targetcls})"
                        else:
                            targetmin = record['target_range']['min']
                            targetmax = record['target_range']['max']
                            if self.dtypes[targetvar] == 'int':
                                targetmin, targetmax = math.floor(targetmin), math.ceil(targetmax)
                            if targetmin == targetmax:
                                #* Too strict condition, skip it
                                # log.warning(
                                #     f"Skipping overly strict rule for {targetvar} (min==max: {targetmin}).")
                                continue    
                            
                            conclusion = f"(({targetvar}>={targetmin}) & ({targetvar}<={targetmax}))"
                        rule = f"({premise}) >> {conclusion}"
                        rules.add(rule)
            learned_rules |= rules
            log.info(f"Extracted {len(rules)} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules

    
    def extract_conditions_from_all_trees(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Extract path conditions from all trees."""
        #* {target: {label1: [conditions1, conditions2, ...], label2: [...]}}
        all_pathconditions: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
        for target, trees in self.trees.items():
            for treeidx, _ in enumerate(tqdm(
                trees, desc=f"Extracting conditions from trees of {target}"
            )):
                # print(f"Features: {self.featuregroups[target][treeidx]}")
                pathconditions = self.extract_conditions_from_tree(treeidx, target)
                #* Merge path conditions for the same target
                for label, conditions in pathconditions.items():
                    if label not in all_pathconditions[target]:
                        all_pathconditions[target][label] = []
                    all_pathconditions[target][label].extend(conditions)
        
        return all_pathconditions
    
    def extract_conditions_from_tree(self, treeidx: int, target: str):
        #TODO: Move to config
        MIN_LOGIT = 0
        
        encoder = self.label_encoders.get(target, None)
        num_classes = len(encoder.classes_) if encoder else 1
        label_map = dict(zip(encoder.transform(encoder.classes_), encoder.classes_)) \
            if encoder else {}
        lgbm = self.trees[target][treeidx]
        tree_info = lgbm.dump_model()['tree_info']
        features = lgbm.feature_name()
        
        leafranges = {}
        if not encoder:
            X = self.examples[features]
            y = self.examples[target]
            leaf_indices = lgbm.predict(X, pred_leaf=True)
            #* Assuming only one tree, flatten the leaf indices
            leaf_indices = leaf_indices.flatten()
            leafdf = pd.DataFrame({'leaf': leaf_indices, 'target': y})
            leafdf = leafdf.groupby('leaf')['target'].agg(['min', 'max', 'count'])
            leafdf.reset_index(inplace=True)
            leafdf
            leafranges = {}
            for i, row in leafdf.iterrows():
                #TODO: Filter by min count in a leaf?
                leafranges[int(row['leaf'])] = {
                    'min': row['min'],
                    'max': row['max'],
                }
        
        def recurse(node, path_conditions, tree_idx):
            if 'leaf_index' in node:
                path_id = f"{tree_idx}-{node['leaf_index']}"
                if node['leaf_value'] > MIN_LOGIT:
                    paths.append({
                        'pathid': path_id,
                        'logit': node['leaf_value'],
                        'conditions': path_conditions.copy(),
                        'target_range': None,  # For regression trees
                    })
                return

            # Safeguard: Ensure it's a proper split node
            if 'split_feature' not in node:
                return

            feat_idx = node['split_feature']
            feat_name = features[feat_idx] if features else f"f{feat_idx}"

            decision_type = node['decision_type']
            threshold = node['threshold']

            if decision_type == '==':  # Categorical split
                left_vals = [
                    #* Index into the original category
                    self.examples[feat_name].astype('category').cat.categories[int(v)] 
                    for v in threshold.split('||')
                ]
                cond_left = f"{feat_name}$∈${left_vals}"
                cond_right = f"{feat_name}$∉${left_vals}"
            else:  # Numeric split
                cond_left = f"{feat_name}$≤${threshold}"
                cond_right = f"{feat_name}$>${threshold}"

            recurse(node['left_child'], path_conditions + [cond_left], tree_idx)
            recurse(node['right_child'], path_conditions + [cond_right], tree_idx)

        pathconditions = {}
        for tree in tree_info:
            tree_idx = tree['tree_index']
            paths = []
            recurse(tree['tree_structure'], [], tree_idx)
            if paths:
                if encoder:
                    #* Multi-class classification tree
                    target_cls = tree_idx % num_classes
                    label = label_map[target_cls]
                    pathconditions[label] = paths
                else:
                    #* Regression tree
                    for path in paths:
                        leaf_id = int(path['pathid'].split('-')[-1])
                        if leaf_id in leafranges:
                            path['target_range'] = leafranges[leaf_id]
                            label = f"{tree_idx}-{leaf_id}"
                            pathconditions[label] = paths
        return pathconditions
