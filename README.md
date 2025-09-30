# Anuta 🏝️

## Setup
```bash
git clone https://github.com/HongyuHe/anuta.git
cd anuta
# On a clean-slate (virtual) machine; don't run this on your host.
./scripts/setup.sh
pip install -e .
```

## Configuration
Change default configurations in `configs/default.py` or create a new config file and pass it using `-config <path_to_config>` flag.


## Data Preparation

### PCAP
```bash
bash scripts/pcap2csv.sh <pcap_file> <output_csv_file>
```

### NetFlow
Convert NetFlow to a format similar to [CIDDS](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/) or [UGR](https://nesg.ugr.es/nesg-ugr16/).


## Rule Learning
* Sub-commands aren't used here for simplicity. Make sure to include the `-learn` flag for learning.
* See `anuta/cli.py` for all available flags.
```bash
python anuta -learn \
    -dataset <cidds|mawi|netflix|...> \
    -data <path_to_csv> \
    # Choose learning methods:
    # Logical learning (default)
    -logic <denial|level> \
    # Tree-based learning (best for numerical vars)
    -tree=<dt|xgb|lgbm> \
    # Association rule mining (best for categorical vars)
    -assoc=<hmine|apriori|fpgrowth> \
    # Limit the number of training examples (default: all)
    -limit <max_num_examples> \
    -cores <max_num_cores>
```

## Data Validation
```bash
python anuta -validate \
    -dataset <cidds|mawi|netflix|...> \
    -data <path_to_csv> \
    -rules path/to/rules.pl \
    ...
```

## Rule Interpretation
```bash
python scripts/interpret_rules.py <rules.pl> <netflow|pcap>
```

## Testing
```bash
pip install -r requirements_dev.txt
pytest -sv <tests/test_*.py>
```
