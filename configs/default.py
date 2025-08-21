from ml_collections import config_dict


def get_config():
    default_config = config_dict.ConfigDict()
    
    default_config.MAX_PREDICATES = 12  #* Maximum number of predicates in a rule
    default_config.MAX_RULES = 100  #* Maximum number of rules to learn
    default_config.LEVELWISE_EPOCHS = 10  #* Number of epochs for levelwise learning
    
    '''Tree leaning configuration'''
    default_config.MAX_COMBO_SIZE = 0 #* If combo size is non-positive, use all combinations
    default_config.MAX_SEPARATE_CONQUER_EPOCHS = 10  #* Maximum rounds of remove-and-conquer
    default_config.MIN_SPLIT_GAIN = 1e-6  #* Minimum gain to split a node
    default_config.JVM_MEM = '150G'  #* Maximum memory for the JVM (H2O), wisconsin c220g2.
    
    '''Legacy configuration'''
    # default_config.BASELINE = False
    default_config.ARITY_LIMIT = 3
    default_config.DOMAIN_COUNTING = True
    default_config.OUTPUT_DIR = "./"

    return default_config
