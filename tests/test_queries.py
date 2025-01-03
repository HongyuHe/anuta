from typing import *
import json
import pytest
from tqdm import tqdm
from rich import print as pprint

from anuta.model import Model


@pytest.fixture
def cases() -> List[Dict[str, str]]:
    """Fixture to load the queries from the JSON file."""
    path = "tests/cidds_queries.json"
    with open(path, 'r') as f:
        return json.load(f)

@pytest.fixture
def model() -> Model:
    modelpath = 'data/results/cidds/versionspace/learned_150_a3.rule'
    # modelpath = 'data/results/cidds/dc/learned_100.rule'
    # modelpath = 'learned_100_a3_mac.rule'
    return Model(modelpath)

def test_queries(cases: List[Dict[str, str]], model: Model) -> None:
    """Test that prints the description of each query."""
    successes = 0
    total_queries = 0
    total_cases = len(cases)
    coverage = 0
    for i, q in tqdm(enumerate(cases), total=len(cases)):
        cases = q['queries']
        nqueries = len(cases)
        total_queries += nqueries

        new_successes = 0
        for query in cases:
            entailed = model.entails(query, verbose=False)
            # assert entailed, f"Failed Test #{i}"
            if entailed:
                new_successes += 1
            else:
                print(f"Failed Test #{i}")
                pprint("\t", query)
                print(f"\t{q['description']}")
        if new_successes == nqueries:
            coverage += 1
            # print("Success")
        else:
            pass
            # print("Failed")
        successes += new_successes

    print(f"Coverage={coverage/total_cases} Recall={successes/total_queries}")