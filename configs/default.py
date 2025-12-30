from ml_collections import config_dict


def get_config():
    default_config = config_dict.ConfigDict()
    
    # 24 hours
    default_config.TIMEOUT_SEC = 24 * 60 * 60
    #* Maximum number of predicates in a rule
    default_config.MAX_PREDICATES = 8
    #* Maximum number of rules to learn
    default_config.MAX_RULES = 50_000
    #* Stop if no improvement in this many seconds
    default_config.STALL_TIMEOUT_SEC = 2 * 60 * 60
    #* Enable type-based variable suppression
    default_config.ENABLE_TYPE_SUPPRESSION = False
    #* Numeric tolerance for predicate evaluation (0 disables).
    #* For Eq/Ne over numeric expressions, use math.isclose/np.isclose with these tolerances.
    default_config.NUMERIC_EQ_RTOL = 1e-6
    default_config.NUMERIC_EQ_ATOL = 1e-6
    
    #* Number of epochs for levelwise learning
    default_config.LEVELWISE_EPOCHS = 10
    
    '''Tree leaning configuration'''
    #* If combo size is non-positive, use all combinations
    default_config.MAX_COMBO_SIZE = 0
    #* Maximum rounds of remove-and-conquer
    default_config.MAX_SEPARATE_CONQUER_EPOCHS = 1
    #* Minimum gain to split a node
    default_config.MIN_SPLIT_GAIN = 1e-6
    #* Maximum memory for the JVM (H2O), wisconsin c220g2.
    default_config.JVM_MEM = '250G'
    # #* Stop learning early if the process uses this fraction of RAM (>1 disables)
    # default_config.MEMORY_STOP_THRESHOLD = 0.9
    
    # '''Legacy configuration'''
    # # default_config.BASELINE = False
    # default_config.ARITY_LIMIT = 3
    # default_config.DOMAIN_COUNTING = True
    # default_config.OUTPUT_DIR = "./"
    
    default_config.DATA_DIR = "/mnt/ann/hy/data/"

    return default_config
