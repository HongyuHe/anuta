from ml_collections import config_dict


def get_config():
    default_config = config_dict.ConfigDict()
    
    default_config.MIN_SPLIT_GAIN = 1e-6  #* Minimum gain to split a node
    default_config.MAX_COMBO_SIZE = 0 #* If combo size is non-positive, use all combinations

    # default_config.BASELINE = False
    default_config.ARITY_LIMIT = 3
    default_config.DOMAIN_COUNTING = True
    default_config.OUTPUT_DIR = "./"

    return default_config
