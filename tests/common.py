from typing import List, Dict
from rich import print as pprint
from anuta.theory import Theory, ProofResult


def get_coverage(cases: List[Dict[str, str]], theory: Theory) -> None:
    print("\033[1;34m" + "#" * 60)
    print("TESTING COVERAGE")
    print("#" * 60 + "\033[0m\n")
    successes = 0
    total_queries = 0
    total_cases = len(cases)
    coverage = 0
    failed_entailments = 0
    failed_contradictions = 0
    passed = []
    failed = []
    for i, q in enumerate(cases):
        cases = q['queries']
        nqueries = len(cases)
        total_queries += nqueries

        new_successes = 0
        for query in cases:
            entailment = theory.z3proves(query, verbose=False)
            contradiction = theory.z3proves(f"Not({query})", verbose=False)
            # result = theory.proves(query, verbose=False)
            # assert entailed, f"Failed Test #{i}"
            if entailment != ProofResult.ENTAILMENT:
                print(f"❌ Failed Test #{i+1} (Entailment):")
                pprint(f"\t{entailment=}")
                failed_entailments += 1
            if contradiction != ProofResult.CONTRADICTION:
                print(f"❌ Failed Test #{i+1} (Contradiction):")
                pprint(f"\t{contradiction=}")
                failed_contradictions += 1
            if entailment == ProofResult.ENTAILMENT and contradiction == ProofResult.CONTRADICTION:
                new_successes += 1
            else:
                print(f"\tMeaning: {q['description']}")
                # pprint("\tQuery: ", query)
        if new_successes == nqueries:
            print(f"✅ Passed Test #{i+1}")
            print(f"\t{q['description']}")
            coverage += 1
            passed.append(i)
        else:
            failed.append(i)
        successes += new_successes

    print(f"\nRules: {theory.path_to_constraints}")
    print(f"Theory consistency: {failed_entailments == failed_contradictions}")
    pprint(f"Passed {len(passed)} tests: {passed}")
    pprint(f"Failed {len(failed)} tests: {failed}")
    print(f"Coverage={coverage/total_cases:.3f}, Recall={successes/total_queries:.3f}")
    
    return len(passed), len(failed), coverage / total_cases, successes / total_queries

def get_specificity(cases: List[Dict[str, str]], theory: Theory) -> None:
    print("\033[1;34m" + "#" * 60)
    print("TESTING SPECIFICITY")
    print("#" * 60 + "\033[0m\n")
    successes = 0
    total_queries = 0
    total_cases = len(cases)
    specificity = 0
    failed_entailments = 0
    failed_contradictions = 0
    passed = []
    failed = []
    for i, q in enumerate(cases):
        cases = q['queries']
        nqueries = len(cases)
        total_queries += nqueries

        new_successes = 0
        for query in cases:
            entailment = theory.z3proves(query, verbose=False)
            contradiction = theory.z3proves(f"Not({query})", verbose=False)
            if entailment == ProofResult.ENTAILMENT:
                print(f"❌ Failed Test #{i+1} (False Entailment):")
                pprint(f"\t{entailment=}")
                failed_entailments += 1
            if contradiction == ProofResult.CONTRADICTION:
                print(f"❌ Failed Test #{i+1} (False Contradiction):")
                pprint(f"\t{contradiction=}")
                failed_contradictions += 1
            if entailment != ProofResult.ENTAILMENT and contradiction != ProofResult.CONTRADICTION:
                new_successes += 1
            else:
                print(f"\tMeaning: {q['description']}")
                # pprint("\tQuery: ", query)
        if new_successes == nqueries:
            print(f"✅ Passed Test #{i+1}")
            print(f"\t{q['description']}")
            specificity += 1
            passed.append(i)
        else:
            failed.append(i)
        successes += new_successes

    print(f"\nRules: {theory.path_to_constraints}")
    print(f"Theory consistency: {failed_entailments == failed_contradictions}")
    pprint(f"Passed {len(passed)} tests: {passed}")
    pprint(f"Failed {len(failed)} tests: {failed}")
    print(f"Specificity={specificity/total_cases:.3f}")
    
    return len(passed), len(failed), specificity / total_cases, successes / total_queries