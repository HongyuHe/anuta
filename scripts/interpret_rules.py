from anuta.theory import Theory


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python interpret_rules.py <rules_file> <rule_type>")
        print("Example: python interpret_rules.py dt_mawi_all_e1.pl pcap")
        sys.exit(1)

    rules_file = sys.argv[1]
    rules_path = rules_file.split('.')[0]
    rtype = sys.argv[2]  # e.g., 'pcap' or 'netflow'
    
    th = Theory(rules_file)
    interpreted_rules = Theory.interpret(th.constraints, dataset=rtype, save_path=f"{rules_path}.clj")
    print(f"Interpreted rules saved to {rules_path}.clj")
    
    exit(0)