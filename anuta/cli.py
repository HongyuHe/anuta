# import sys
from absl import flags
from ml_collections import config_flags


########### Command line options ###########
'''Example usage:
#* Inspect dataset
python anuta -dataset=cidds -data=data/CIDDS/cidds_wk3_4k.csv
#* Learn constraints
python anuta -learn -dataset=cidds -data=data/cidds_wk3_all.csv -baseline -limit=512
python anuta -learn -dataset=cidds -data=data/cidds_wk3_all.csv -baseline -limit=8k (-ref=data/cidds_wk4_all.csv)
#* Validate dataset
python anuta -validate -dataset=netflix -data=data/syn/netflix_rtf_syn.csv -limit=1k -rules=results/cidds/confidence/learned_10000_a3.pl
'''
FLAGS = flags.FLAGS

#* Commands (avoid using subcommands for simplicity)
flags.DEFINE_boolean("learn", False, "Learn constraints from a dataset")
flags.DEFINE_boolean("validate", False, "Validate a dataset using a learned theory")

#* Configs
flags.DEFINE_enum("logic", 'denial', ['denial', 'level'], "Logic learning method to use")
flags.DEFINE_enum("tree", None, ['dt', 'xgb', 'lgbm'], "Tree learner to use for learning constraints")
flags.DEFINE_enum("assoc", None, ['apriori', 'fpgrowth', 'hmine'], "Association rule learning algorithm to use")
flags.DEFINE_string("limit", None, "Limit on the number of examples to learn from")
#TODO: Generalize `dataset` to netflow and pcap.
flags.DEFINE_enum("dataset", None, ['cidds', 'netflix', 'cicids', 'yatesbury', 'metadc', 'mawi'], "Name of the dataset to learn from")
flags.mark_flag_as_required('dataset')
flags.DEFINE_string("data", None, "Path to the dataset to learn from or validate")
flags.mark_flag_as_required('data')
flags.DEFINE_string("ref", "", "Path to the reference dataset")
flags.DEFINE_string("rules", "", "Path to the learned rules")
flags.DEFINE_boolean("baseline", False, "Use the baseline method Valiant algorithm")
#* Use `-nodc` to disable domain counting
flags.DEFINE_boolean("dc", True, "Enable domain counting")
flags.DEFINE_integer("cores", None, "Maximum number of cores allowed to use")

config_flags.DEFINE_config_file("config", default="./configs/default.py")
# FLAGS(sys.argv)
########### End of Command line options ###########