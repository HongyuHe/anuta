from typing import *
import json
import pytest
from tqdm import tqdm
from rich import print as pprint

from anuta.theory import Theory, ProofResult
from tests.common import get_coverage


@pytest.fixture
def cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    # path = "tests/queries/cidds_queries.json"
    path = "tests/queries/cidds_benchmark.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def theory() -> Theory:
    # modelpath = 'results/cidds/nodc/learned_8192.pl'
    # modelpath = 'rules/cidds/nodc/learned_1024_checked.pl'
    # modelpath = 'rules/cidds/nodc/learned_128_checked.pl'
    # modelpath = 'rules/cidds/nodc/learned_256.pl'
    # # modelpath = 'results/cidds/dc/learned_4096_checked.pl'
    # # modelpath = 'learned_cidds_128.pl'
    # # modelpath = 'dtnum_hmine.pl'
    # # modelpath = 'dtree_cidds_10000.pl'
    # modelpath = 'fpgrowth_cidds_10000.pl'
    # modelpath = 'rules/new/hmine_cidds_all.pl'
    # modelpath = 'xgb_cidds_1000000_1feat.pl'
    # modelpath = 'lgbm_cidds_1000.pl'
    # modelpath = 'dt_cidds_10000.pl'
    # modelpath = 'dt_cidds_all.pl'
    # modelpath = 'xgb_cidds_all.pl'
    # modelpath = 'lgbm_cidds_all.pl'
    # modelpath = 'rules/new/dt_cidds_10000.pl'
    # modelpath = 'rules/new/lgbm_cidds_all.pl'
    # modelpath = 'rules/new/dt_cidds_all.pl'
    # modelpath = 'rules/new/dt_cidds_all_old.pl'
    # modelpath = 'rules/new/xgb_cidds_all.pl'
    # modelpath = 'rules/cidds/new/learned_cidds_8192_filtered.pl'
    modelpath = 'rules/cidds/new/learned_cidds_8192_checked.pl'
    return Theory(modelpath)

def test_cidds(cases: List[Dict[str, str]], theory: Theory) -> None:
    get_coverage(cases, theory)