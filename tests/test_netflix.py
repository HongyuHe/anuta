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
    # path = "tests/queries/netflix_queries.json"
    path = "tests/queries/netflix_benchmark.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def theory() -> Theory:
    # modelpath = 'rules/netflix/nodc/learned_netflix_1024.pl'
    # modelpath = 'rules/netflix/nodc/learned_netflix_4096.pl'
    # modelpath = 'rules/netflix/dc/learned_netflix_512.pl'
    # modelpath = 'learned_netflix_128.pl'
    
    # modelpath = 'lgbm_netflix_all.pl'
    # modelpath = 'xgb_netflix_all.pl'
    modelpath = 'dt_netflix_combo2.pl'
    return Theory(modelpath)

def test_netflix(cases: List[Dict[str, str]], theory: Theory) -> None:
    get_coverage(cases, theory)