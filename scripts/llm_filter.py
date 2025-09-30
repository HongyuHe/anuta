import re
import os
import csv
import json
import argparse
import time
import requests
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from openai import OpenAI

# Config
sandbox_api_key = os.environ["AI_SANDBOX_KEY"]
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2024-02-01"
models = [
    "gpt-4o", 
    "gpt-4o-mini", 
    # "o3-mini",
    # "gpt-35-turbo-16k", 
    "Meta-Llama-3-1-70B-Instruct-htzs", 
    "Meta-Llama-3-1-8B-Instruct-nwxcg", 
    "Mistral-small-zgjes",
    # "Mistral-large-ygkys"
]
model_to_be_used = models[0] # "Meta-Llama-3-1-70B-Instruct-htzs" 
print(f"Using model: {model_to_be_used}")
# input_file = "rules.txt"         # Input file: one rule per line
# output_csv = "classified_rules.csv"
max_tokens_per_batch = 12000     # Adjust conservatively below token limit (16k for gpt-4o)
timeout = 60                     # Timeout for API requests (in seconds)
retries = 10_000              # Number of retries for failed requests

# Initialize client (if using AzureOpenAI SDK for other tasks)
client = AzureOpenAI(
    api_key=sandbox_api_key,
    azure_endpoint=sandbox_endpoint,
    api_version=sandbox_api_version
)

# Prepare HTTP headers for direct image/text POST
headers = {
    'api-key': sandbox_api_key,
    'Content-Type': 'application/json'
}

# Prompt template
system_msg = {
    "role": "system",
    "content": "You are an expert in maths and computer networking who classifies logic rules extracted from network data."
}

def build_user_message(batch_rules):
    # instructions = (
    #     "You are given a list of logical rules extracted from network measurement data. "
    #     "For each rule, classify it according to the following:\n\n"
    #     "- **rtype** (rule type), which must be one of:\n"
    #     "  • `protocol`: rules derived from network protocol specifications (e.g., TCP, UDP behavior).\n"
    #     "  • `principle`: rules that arise from general performance, behavioral, or theoretical principles, "
    #     "like queueing theory or latency constraints.\n"
    #     "  • `deployment`: rules specific to a particular network's configuration, topology, or operational norms.\n\n"
    #     "- **meaningful**: a boolean indicating if the rule is semantically meaningful. For example, "
    #     "`Duration < Bytes` is likely invalid (not meaningful), while `QueueLength > 0 ⇒ PacketsReceived > 0` is valid.\n\n"
    #     "Return a JSON array. Each element must have this format:\n"
    #     "{ \"ruleid\": <line number>, \"rtype\": <protocol|principle|deployment>, \"meaningful\": <true|false> }\n\n"
    #     "Classify these rules:\n"
    # )
    instructions = (
        "You are given a list of logical rules extracted from network telemetry data. "
        "These rules are intended to capture meaningful relationships between telemetry features and will be used "
        "to guide a machine learning model (Zoom2Net) that imputes fine-grained time series from coarse-grained aggregates.\n\n"

        "Zoom2Net aims to reconstruct high-resolution network behaviors such as bursts and congestion episodes using "
        "coarse signals. Therefore, rules must reflect *real and useful* dependencies — especially those relevant to burst structure, "
        "traffic patterns, and congestion signals — and not just statistical artifacts or tautologies.\n\n"

        "**Your task is to identify which rules are semantically meaningful. Be extremely critical.** A rule is meaningful only if it "
        "represents valid network behavior or constraints that would help the model predict fine-grained telemetry. Discard any rule that:\n"
        "- Expresses an implausible or impossible relationship (e.g., Bytes > Duration)\n"
        "- Represents spurious correlations not grounded in domain knowledge\n\n"

        "**Variable definitions**:\n"
        "- `IngressBytesAgg`: Total ingress bytes over the 50ms window (sum of `IngressBytes0` through `IngressBytes49`).\n"
        "- `EgressBytesAgg`: Total egress bytes over the 50ms window (sum of `EgressBytes0` through `EgressBytes49`).\n"
        "- `InRxmitBytesAgg`: Total retransmitted ingress bytes in the 50ms window.\n"
        "- `OutRxmitBytesAgg`: Total retransmitted egress bytes in the 50ms window.\n"
        "- `InCongestionBytesAgg`: Number of ingress bytes during congestion periods in the 50ms window.\n"
        "- `ConnectionsAgg`: Number of TCP connections active in the window.\n"
        "- `IngressBytes0` through `IngressBytes49`: Per-millisecond ingress bytes.\n\n"

        "You will be shown a list of logical rules. For each rule, return `true` if the rule is semantically meaningful and could assist "
        "in imputation by encoding valid correlations or constraints. Return `false` otherwise.\n\n"

        "Respond with a JSON array. Each element must have this format:\n"
        "{ \"ruleid\": <line number>, \"meaningful\": <true|false> }\n\n"

        "Classify these rules:\n"
    )
    # instructions = (
    #     "You are given a list of logical rules extracted from network data. "
    #     "These rules aim to describe relationships between fields in network packets or flow records.\n\n"

    #     "### Task\n"
    #     "Decide whether each rule is **semantically meaningful**. "
    #     "Be very strict: a rule is meaningful only if it reflects valid network behavior or expresses a sound integrity constraint.\n\n"

    #     "### Mark a rule as NOT meaningful if it:\n"
    #     "- Describes an implausible or impossible relationship (e.g., Bytes > Duration).\n"
    #     "- Encodes spurious correlations with no basis in networking knowledge.\n"
    #     "- Relies on chance patterns instead of protocol, principle, or deployment semantics.\n\n"

    #     "### Possible rule origins (for your context only):\n"
    #     "- **protocol**: derived from protocol specifications (e.g., TCP/UDP semantics like handshake).\n"
    #     "- **principle**: based on general behavioral or theoretical principles (e.g., queueing, latency bounds).\n"
    #     "- **deployment**: tied to a particular network's configuration, topology, or policy (e.g., unused ports never receiving traffic).\n\n"

    #     "### Output format\n"
    #     "For each input rule, respond with a JSON array element of the form:\n"
    #     "{ \"ruleid\": <line number>, \"meaningful\": true|false }\n\n"

    #     "Return only a JSON array, nothing else.\n\n"

    #     "### Classify the following rules:\n"
    # )
    # instructions = (
    #     "You are given a list of logical rules extracted from network data. "
    #     "These rules aim to describe relationships between fields in network packets or flow records.\n\n"

    #     "## Task\n"
    #     "Decide whether each rule is **semantically valid**. "
    #     "A rule is valid only if it reflects valid network behavior or expresses a sound integrity constraint.\n\n"

    #     "## Output format\n"
    #     "For each input rule, respond with a JSON array element of the form:\n"
    #     "{ \"ruleid\": <rule id>, \"meaningful\": <true|false> }\n\n"

    #     "Return only a JSON array, nothing else. **Be excessively critical!**\n\n"

    #     "## Classify the following rules:\n"
    # )
    # for rule in batch_rules:
    #     instructions += f"{rule['id']}: {rule['rule']}\n"
    # return {"role": "user", "content": [{"type": "text", "text": instructions}]}
    instructions = (
        "You are given a list of logical rules extracted from network data.\n\n"
        "These rules aim to describe relationships between fields in network packets or flow records.\n\n"
        "## Task\n"
        "Identify which rules are semantically valid. A rule is valid only if it reflects "
        "real network behavior or expresses a sound integrity constraint.\n\n"
        "## Output format\n"
        "Return a Python-style list of the rule IDs that are valid. Example:\n"
        "[0, 2, 5]\n\n"
        "Do not return anything else. **Be excessively critical!**\n\n"
        "## Rules:\n"
    )
    for rule in batch_rules:
        instructions += f"{rule['id']}: {rule['rule']}\n"
    return {"role": "user", "content": [{"type": "text", "text": instructions}]}

def extract_id_list_from_response(text):
    """
    Extracts a Python-style list of integers from the LLM output.
    """
    match = re.search(r"\[.*?\]", text.strip(), re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))  # should be a list of ints
    except Exception as e:
        print(f"❌ Failed to parse ID list: {text}\nError: {e}")
        return []

def extract_json_from_response(text):
    """
    Extracts and cleans a JSON array string from a Markdown-style code block,
    removing triple backticks, optional 'json' label, and any '//' comments.
    """
    # Match the JSON block (with or without ```json)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text.strip(), re.IGNORECASE)
    json_block = match.group(1) if match else text.strip()

    # Remove inline and full-line '//' comments
    json_cleaned = re.sub(r'(?://[^\n]*)', '', json_block)

    # Optionally strip trailing commas (which are invalid in JSON)
    json_cleaned = re.sub(r',\s*(?=[}\]])', '', json_cleaned)

    return json_cleaned.strip()

# def call_api_with_rules(batch):
#     data = {
#         "messages": [
#             system_msg,
#             build_user_message(batch)
#         ],
#         "model": model_to_be_used,
#         "max_tokens": max_tokens_per_batch,
#         "temperature": 0.0,
#         # "top_p": 0.01,  # Optional: can be adjusted for more deterministic output
#     }
    
#     for attempt in range(1, retries + 1):
#         try:
#             endpoint = f"{sandbox_endpoint}openai/deployments/{model_to_be_used}/chat/completions?api-version={sandbox_api_version}"
#             response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=timeout)
#             response.raise_for_status()
#             result = response.json()
#             content = result["choices"][0]["message"]["content"]

#             try:
#                 cleaned = extract_json_from_response(content)
#                 return json.loads(cleaned)
#             except Exception as e:
#                 print(f"❌ Failed to parse LLM response:\n{content}\nError: {e}")
#                 return []

#         except requests.exceptions.Timeout:
#             wait = 1
#             print(f"⏳ Timeout on attempt {attempt}. Retrying in {wait}s...")
#             time.sleep(wait)

#         except Exception as e:
#             print(f"❌ Failed on attempt {attempt}: {e}")
#             break

#     print("❗ Max retries exceeded for this batch.")
#     return []

def call_api_with_rules(batch):
    data = {
        "messages": [system_msg, build_user_message(batch)],
        "model": model_to_be_used,
        "max_tokens": max_tokens_per_batch,
        "temperature": 0.0,
    }
    
    for attempt in range(1, retries + 1):
        try:
            endpoint = f"{sandbox_endpoint}openai/deployments/{model_to_be_used}/chat/completions?api-version={sandbox_api_version}"
            response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            return extract_id_list_from_response(content)

        except requests.exceptions.Timeout:
            print(f"⏳ Timeout on attempt {attempt}. Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed on attempt {attempt}: {e}")
            break

    return []

def split_batches(rules, token_limit=max_tokens_per_batch):
    batches = []
    batch, current_len = [], 0
    for rule in rules:
        rule_len = len(rule['rule'])  # simple char count proxy for token length
        if current_len + rule_len > token_limit and batch:
            batches.append(batch)
            batch, current_len = [], 0
        batch.append(rule)
        current_len += rule_len
    if batch:
        batches.append(batch)
    return batches

def read_rules(file_path):
    rules = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                rules.append({"id": idx, "rule": line.strip()})
    return rules

# def save_results_to_csv(results, output_file):
#     base, ext = os.path.splitext(output_file)
#     counter = 1
#     final_output = output_file

#     # Increment suffix until we find a non-existing filename
#     while os.path.exists(final_output):
#         final_output = f"{base}_{counter}{ext}"
#         counter += 1

#     with open(final_output, mode="w", newline='', encoding="utf-8") as file:
#         writer = csv.DictWriter(file, fieldnames=["ruleid", "meaningful"])
#         writer.writeheader()
#         for entry in results:
#             writer.writerow(entry)
    
#     print(f"Saved results to {final_output}")

def save_results_to_csv(results, output_file):
    base, ext = os.path.splitext(output_file)
    counter, final_output = 1, output_file
    while os.path.exists(final_output):
        final_output = f"{base}_{counter}{ext}"
        counter += 1

    with open(final_output, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ruleid"])  # header
        for rid in results:
            writer.writerow([rid])
    
    print(f"Saved results to {final_output}")


# def classify_batches_in_parallel(batches):
#     all_results = []

#     core_count = cpu_count()
#     print(f"Using {core_count} cores for parallel processing.")
#     with ProcessPoolExecutor(max_workers=core_count) as executor:
#         # Submit in order; map preserves input-output order
#         results = list(tqdm(executor.map(call_api_with_rules, batches), total=len(batches), desc="Classifying rules"))

#     # Flatten results while preserving rule order
#     for batch_result in results:
#         all_results.extend(batch_result)

#     return all_results

def classify_batches_in_parallel(batches):
    all_ids = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = list(tqdm(executor.map(call_api_with_rules, batches), total=len(batches)))
    for batch_ids in results:
        all_ids.extend(batch_ids)
    return all_ids

def main():
    parser = argparse.ArgumentParser(description="Classify logical rules using the Sandbox API.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input file containing logical rules")
    parser.add_argument("--output", "-o", help="Path to the output CSV file (optional)")
    parser.add_argument("--label", "-l", help="Label to append to output filename (optional)")

    args = parser.parse_args()
    input_file = args.input

    # Default output: <input_name>_llm.csv
    if args.output:
        output_csv = args.output
    else:
        base, _ = os.path.splitext(input_file)
        model = model_to_be_used # f"{'-'.join(model_to_be_used.split('-')[:2])}"
        label = f"_{args.label}" if args.label else ""
        output_csv = f"{base}_{model}{label}.csv"

    print(f"Reading rules from {input_file}...")
    rules = read_rules(input_file)
    batches = split_batches(rules)

    print(f"Processing {len(rules)} rules in {len(batches)} batches...")
    all_results = classify_batches_in_parallel(batches)

    save_results_to_csv(all_results, output_csv)
    # print(f"Saved classification results to {output_csv}")

if __name__ == "__main__":
    main()