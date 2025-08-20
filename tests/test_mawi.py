from typing import *
import json
import pytest
from tqdm import tqdm
from rich import print as pprint

from anuta.theory import Theory, ProofResult
from tests.common import get_coverage, get_specificity, run_evaluation


@pytest.fixture
def possitive_cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    path = "tests/queries/mawi_benchmark.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def negative_cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    path = "tests/queries/mawi_benchmark_neg.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def theory() -> Theory:
    modelpath = 'denial_mawi_all_p8.pl'
    return Theory(modelpath)

def test_mawi(
    possitive_cases: List[Dict[str, str]], 
    negative_cases: List[Dict[str, str]], 
    theory: Theory) -> None:
    
    run_evaluation(possitive_cases, negative_cases, theory)