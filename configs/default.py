from ml_collections import config_dict


def get_config():
    default_config = config_dict.ConfigDict()
    
    default_config.TREE_ARITY_LIMIT = 4

    # default_config.BASELINE = False
    default_config.ARITY_LIMIT = 3
    default_config.DOMAIN_COUNTING = True
    default_config.OUTPUT_DIR = "./"

    return default_config