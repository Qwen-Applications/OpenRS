"""
RMBench Evaluation Script

Features:
1. Load RMBench format data (3 variants for chosen/rejected each)
2. Evaluate 9 pair combinations (aa, ab, ac, ba, bb, bc, ca, cb, cc)
3. Evaluate each pair twice (swapping positions) to eliminate position bias
4. Call common evaluation interface from evaluator.py
5. Automatically generate summary statistics

Usage:
python rmbench.py --input data/rmbench.json --output results/rmbench_results.jsonl
"""

import argparse
import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple

from tqdm import tqdm

# Use common evaluation interface
from evaluator import evaluate_pairwise, evaluate_verifiable
from tools import load_jsonl, save_jsonl
from robust_utils import safe_json_dumps_robust as safe_json_dumps

logger = logging.getLogger(__name__)

# ============== Editable Parameters ==============
MAX_CONCURRENT = 10             # Max concurrency
TEMPERATURE = 0.0               # Generation temperature
# ========================================


# File lock
file_lock = threading.Lock()

# Pair labels
PAIR_LABELS: List[str] = [
    "aa", "ab", "ac",
    "ba", "bb", "bc",
    "ca", "cb", "cc",
]

# Pair difficulty classification
MODE_BY_PAIR: Dict[str, str] = {
    "aa": "normal", "bb": "normal", "cc": "normal",
    "ab": "hard", "ac": "hard", "bc": "hard",
    "ca": "easy", "cb": "easy", "ba": "easy",
}

# Variant index mapping
VARIANT_MAP: Dict[str, int] = {"a": 0, "b": 1, "c": 2}

# domain is used directly as query_type
# You can create corresponding .md files in ./prompts/pairwise_prompts/ directory
# e.g., chat.md, code.md, math.md etc.


# ============== Data Structures ==============

@dataclass
class JudgeInput:
    """Input for a single evaluation"""
    sample_id: str
    domain: str
    query: str
    pair: str
    order: int  # 1: A=chosen, B=rejected; 2: swap positions
    chosen_variant: str
    rejected_variant: str
    response_a: str
    response_b: str
    ground_truth: Optional[str] = None


@dataclass
class Counter:
    """win/tie/lose counter"""
    win: int = 0
    tie: int = 0
    lose: int = 0

    def add(self, result: str) -> None:
        if result == "win":
            self.win += 1
        elif result == "tie":
            self.tie += 1
        elif result == "lose":
            self.lose += 1

    @property
    def total(self) -> int:
        return self.win + self.tie + self.lose

    def metrics(self) -> Dict[str, Any]:
        total = self.total
        win_rate = self.win / total if total else 0.0
        tie_rate = self.tie / total if total else 0.0
        lose_rate = self.lose / total if total else 0.0
        denom = self.win + self.lose
        net_win_rate = self.win / denom if denom else 0.0
        return {
            "total": total,
            "win": self.win,
            "tie": self.tie,
            "lose": self.lose,
            "win_rate": round(win_rate, 4),
            "tie_rate": round(tie_rate, 4),
            "lose_rate": round(lose_rate, 4),
            "net_win_rate": round(net_win_rate, 4),
        }


# ============== Core Functions ==============

def get_sample_field(sample: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get sample field, supporting multiple candidate field names"""
    for key in keys:
        if key in sample:
            return sample[key]
    return default


def build_judge_inputs(sample: Dict[str, Any]) -> List[JudgeInput]:
    """Build all evaluation inputs for a single sample (9 pairs * 2 orders = 18 times)"""
    sample_id = get_sample_field(sample, "id", "question_id", default="unknown")
    domain = get_sample_field(sample, "domain", "query_type", default="general")
    query = get_sample_field(sample, "prompt", default="")
    ground_truth = get_sample_field(sample, "ground_truth", default=None)
    
    chosen = sample.get("chosen", [])
    rejected = sample.get("rejected", [])
    
    # If single response (not list), convert to single-element list (compatible with main.py data format)
    if isinstance(chosen, str):
        chosen = [chosen]
    if isinstance(rejected, str):
        rejected = [rejected]
    
    # Ensure at least one variant
    if not chosen or not rejected:
        raise ValueError(f"Sample {sample_id}: chosen or rejected is empty")
    
    # Pad to 3 variants (if insufficient)
    while len(chosen) < 3:
        chosen.append(chosen[-1])
    while len(rejected) < 3:
        rejected.append(rejected[-1])
    
    inputs: List[JudgeInput] = []
    for pair in PAIR_LABELS:
        c_key, r_key = pair[0], pair[1]
        ci, ri = VARIANT_MAP[c_key], VARIANT_MAP[r_key]
        
        # order 1: A=chosen, B=rejected
        inputs.append(JudgeInput(
            sample_id=sample_id,
            domain=domain,
            query=query,
            pair=pair,
            order=1,
            chosen_variant=c_key,
            rejected_variant=r_key,
            response_a=str(chosen[ci]),
            response_b=str(rejected[ri]),
            ground_truth=ground_truth,
        ))
        
        # order 2: Swap positions, A=rejected, B=chosen
        inputs.append(JudgeInput(
            sample_id=sample_id,
            domain=domain,
            query=query,
            pair=pair,
            order=2,
            chosen_variant=c_key,
            rejected_variant=r_key,
            response_a=str(rejected[ri]),
            response_b=str(chosen[ci]),
            ground_truth=ground_truth,
        ))
    
    return inputs


def judge_one(ji: JudgeInput, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Execute single evaluation (call common interface of evaluator.py)
    
    Process:
    1. If ground_truth exists, perform Verifiable evaluation first
    2. Perform Pairwise evaluation (A vs B)
    3. Return evaluation results
    """
    result = {
        "pair": ji.pair,
        "order": ji.order,
        "chosen_variant": ji.chosen_variant,
        "rejected_variant": ji.rejected_variant,
    }
    
    # Determine chosen and rejected positions based on order
    if ji.order == 1:
        # A = chosen, B = rejected
        chosen_response = ji.response_a
        rejected_response = ji.response_b
    else:
        # A = rejected, B = chosen
        chosen_response = ji.response_b
        rejected_response = ji.response_a
    
    # --- Part 1: Verifiable (Fact Checking) ---
    # Call evaluate_verifiable from evaluator.py
    verifiable_conclusive = False
    
    if ji.ground_truth:
        try:
            verifiable_result = evaluate_verifiable(
                query=ji.query,
                chosen=chosen_response,
                rejected=rejected_response,
                ground_truth=ji.ground_truth,
                temperature=temperature
            )
            result["verifiable"] = verifiable_result
            verdict = verifiable_result.get('verdict', 'error')
            result["verifiable_verdict"] = verdict
            
            # Considered conclusive only when a clear winner is determined
            if verdict in ["chosen_better", "rejected_better"]:
                verifiable_conclusive = True
                
                # Must map verdict (chosen/rejected) back to Pairwise Winner (A/B)
                # Order 1: A=Chosen, B=Rejected
                # Order 2: A=Rejected, B=Chosen
                if verdict == "chosen_better":
                    winner_ab = "A" if ji.order == 1 else "B"
                else: # rejected_better
                    winner_ab = "B" if ji.order == 1 else "A"

                result["verifiable_winner"] = winner_ab
                
                # Use Verifiable result directly as final result
                result["winner"] = winner_ab
                result["score"] = 1.0 if winner_ab == "A" else -1.0
                # Mark as verifiable winner, no need to run pairwise
                
        except Exception as e:
            result["verifiable_error"] = str(e)
    
    # --- Part 2: Pairwise (Pairwise Comparison) ---
    # Run Pairwise only when no GT, or Verifiable is inconclusive (Tie/Error)
    if not verifiable_conclusive:
        try:
            pairwise_result = evaluate_pairwise(
                query=ji.query,
                response_a=ji.response_a,
                response_b=ji.response_b,
                query_type=ji.domain,  # domain as query_type
                temperature=temperature
            )
            
            result["pairwise_raw_result"] = pairwise_result.get('raw_result')
            result["score"] = pairwise_result.get('score')
            result["winner"] = pairwise_result.get('winner')
            
            if 'error' in pairwise_result:
                result["error"] = pairwise_result['error']
            
        except Exception as e:
            result["winner"] = None
            result["score"] = None
            result["error"] = f"Pairwise error: {e}"
    
    return result



def aggregate_pair_result(w1: str, w2: str) -> str:
    """
    Aggregate two evaluation results (position swap)
    
    Rules:
    - win: chosen is better in both cases (order1=A wins, order2=B wins)
    - lose: rejected is better in both cases (order1=B wins, order2=A wins)
    - tie: other cases
    """
    if w1 is None or w2 is None:
        return "error"
    
    # order 1: A=chosen, B=rejected
    # order 2: A=rejected, B=chosen
    chosen_win_1 = (w1 == "A")  # A wins in order1 = chosen wins
    chosen_win_2 = (w2 == "B")  # B wins in order2 = chosen wins
    
    if chosen_win_1 and chosen_win_2:
        return "win"
    
    rejected_win_1 = (w1 == "B")
    rejected_win_2 = (w2 == "A")
    
    if rejected_win_1 and rejected_win_2:
        return "lose"
    
    return "tie"


def process_sample(
    sample: Dict[str, Any],
    output_file: str,
    raw_output_file: str,
    error_file: str,
    temperature: float = 0.0,
    workers: int = MAX_CONCURRENT,
) -> Dict[str, Any]:
    """Process all evaluations for a single sample"""
    sample_id = get_sample_field(sample, "id", "question_id", default="unknown")
    domain = get_sample_field(sample, "domain", "query_type", default="general")
    query = get_sample_field(sample, "prompt", default="")
    
    try:
        judge_inputs = build_judge_inputs(sample)
    except Exception as e:
        logger.error("%s: Build judge inputs failed - %s", sample_id, e)
        return {"id": sample_id, "domain": domain, "error": str(e)}
    
    # Execute all evaluations concurrently
    raw_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    # Limit inner concurrency to avoid thread explosion (outer layer already has main concurrency)
    inner_workers = min(workers, 4) 
    with ThreadPoolExecutor(max_workers=inner_workers) as executor:
        futures = {executor.submit(judge_one, ji, temperature): ji for ji in judge_inputs}
        
        for future in as_completed(futures):
            ji = futures[future]
            try:
                result = future.result()
                raw_results[(result["pair"], result["order"])] = result
                
                # Write raw results
                raw_record = {
                    "id": sample_id,
                    "domain": domain,
                    **result,
                }
                with file_lock:
                    with open(raw_output_file, "a", encoding="utf-8") as f:
                        f.write(safe_json_dumps(raw_record) + "\n")
                
            except Exception as e:
                error_record = {
                    "id": sample_id,
                    "domain": domain,
                    "pair": ji.pair,
                    "order": ji.order,
                    "error": repr(e),
                }
                with file_lock:
                    with open(error_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                raw_results[(ji.pair, ji.order)] = {"winner": None, "error": repr(e)}
    
    # Aggregate results for each pair
    pair_results: Dict[str, Any] = {}
    for pair in PAIR_LABELS:
        r1 = raw_results.get((pair, 1), {})
        r2 = raw_results.get((pair, 2), {})
        w1 = r1.get("winner")
        w2 = r2.get("winner")
        
        result = aggregate_pair_result(w1, w2)
        pair_results[pair] = {
            "order1_winner": w1,
            "order2_winner": w2,
            "result": result,
        }
    
    # Build final record
    final_record = {
        "id": sample_id,
        "domain": domain,
        "prompt": query,
        "pair_results": pair_results,
    }
    
    # Write results
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    
    return final_record


def compute_summary(output_file: str) -> Dict[str, Any]:
    """Compute summary statistics (merge summary_rmbench.py logic)"""
    # domain -> pair -> Counter
    domain_pair: DefaultDict[str, Dict[str, Counter]] = defaultdict(
        lambda: {p: Counter() for p in PAIR_LABELS}
    )
    # domain -> mode -> Counter
    domain_mode: DefaultDict[str, Dict[str, Counter]] = defaultdict(
        lambda: {"easy": Counter(), "normal": Counter(), "hard": Counter()}
    )
    # Global statistics
    global_pair: Dict[str, Counter] = {p: Counter() for p in PAIR_LABELS}
    global_mode: Dict[str, Counter] = {"easy": Counter(), "normal": Counter(), "hard": Counter()}
    
    # Read results
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except:
                continue
            
            domain = rec.get("domain", "unknown")
            pair_results = rec.get("pair_results", {})
            
            for pair in PAIR_LABELS:
                pr = pair_results.get(pair, {})
                result = pr.get("result")
                if result is None or result == "error":
                    continue
                
                mode = MODE_BY_PAIR[pair]
                
                domain_pair[domain][pair].add(result)
                domain_mode[domain][mode].add(result)
                global_pair[pair].add(result)
                global_mode[mode].add(result)
    
    # Build summary
    summary: Dict[str, Any] = {
        "global": {
            "by_pair": {p: global_pair[p].metrics() for p in PAIR_LABELS},
            "by_mode": {m: global_mode[m].metrics() for m in ["easy", "normal", "hard"]},
        },
        "by_domain": {},
    }
    
    for domain in sorted(domain_pair.keys()):
        summary["by_domain"][domain] = {
            "by_pair": {p: domain_pair[domain][p].metrics() for p in PAIR_LABELS},
            "by_mode": {m: domain_mode[domain][m].metrics() for m in ["easy", "normal", "hard"]},
        }
    
    return summary


def load_done_ids(output_file: str) -> set:
    """Load completed sample IDs (for resume)"""
    done_ids = set()
    if not os.path.exists(output_file):
        return done_ids
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("id")
                if sid:
                    done_ids.add(sid)
            except:
                continue
    
    return done_ids


def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data file (supports JSON and JSONL formats)"""
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            return json.loads(content)
        else:
            return [json.loads(line) for line in content.split("\n") if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="RMBench Evaluation Script")
    
    parser.add_argument("--input", required=True, help="Input data file path")
    parser.add_argument("--output", default="results/rmbench_results.jsonl", help="Output result file path")
    parser.add_argument("--raw-output", default=None, help="Raw evaluation result file path")
    parser.add_argument("--error-log", default=None, help="Error log file path")
    parser.add_argument("--summary", default=None, help="Summary statistics file path")
    
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT, help="Concurrency")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Generation temperature")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=no limit)")
    parser.add_argument("--domain", type=str, nargs="+", default=None, help="Process specific domains only (e.g. chat, code, math etc., supports multiple)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Set file paths
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_output_file = args.raw_output or str(output_dir / "rmbench_raw_results.jsonl")
    error_file = args.error_log or str(output_dir / "rmbench_errors.jsonl")
    summary_file = args.summary or str(output_dir / "rmbench_summary.json")
    
    # Load data
    logger.info("Load data: %s", args.input)
    all_data = load_data(args.input)
    logger.info("Total %d items", len(all_data))
    
    # Filter specific domains
    if args.domain:
        # Support case-insensitive matching
        target_domains = {d.lower() for d in args.domain}
        all_data = [
            s for s in all_data 
            if get_sample_field(s, "domain", "query_type", default="").lower() in target_domains
        ]
        logger.info("Filtering domains=%s: %d items", args.domain, len(all_data))
    
    # Resume
    if not args.no_resume:
        done_ids = load_done_ids(args.output)
        if done_ids:
            logger.info("Resume: %d items already completed, skipping", len(done_ids))
            all_data = [
                s for s in all_data 
                if get_sample_field(s, "id", "question_id") not in done_ids
            ]
    else:
        # Clear output files
        for f in [args.output, raw_output_file, error_file]:
            if os.path.exists(f):
                os.remove(f)
    
    # Limit number of items
    if args.limit > 0:
        all_data = all_data[:args.limit]
    
    # ... (previous code remains unchanged) ...

    if not all_data:
        logger.info("No data to process")
        return

    logger.info("Start processing %d items", len(all_data))
    logger.info("Concurrency: %d", args.workers)
    logger.info("Evaluation tasks per sample: %d (9 pairs * 2 orders)", len(PAIR_LABELS) * 2)
    
    # 1. Pre-build mapping of all task inputs
    # map: sample_id -> { (pair, order) -> result }
    # 用于最后聚合
    sample_results_map: DefaultDict[str, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    
    # 2. Generate all fine-grained tasks
    all_tasks: List[JudgeInput] = []
    logger.info("Building task list...")
    
    # Only need ID to Sample mapping for subsequent aggregation
    id_to_sample = {} 
    
    for sample in all_data:
        try:
            tasks = build_judge_inputs(sample)
            all_tasks.extend(tasks)
            sid = get_sample_field(sample, "id", "question_id", default="unknown")
            id_to_sample[sid] = sample
        except Exception as e:
            logger.warning("Skipping sample %s: %s", sample.get('id', 'unknown'), e)

    total_tasks = len(all_tasks)
    logger.info("Total %d evaluation subtasks generated", total_tasks)

    # 3. Global concurrent execution
    completed_tasks = 0
    
    # Real-time tracking: sample_id -> number of completed tasks
    sample_progress = defaultdict(int)
    # Thread lock
    progress_lock = threading.Lock()
    
    # Total tasks required per sample (usually 18, but may be fewer if build fails)
    sample_total_tasks = defaultdict(int)
    for ji in all_tasks:
        sample_total_tasks[ji.sample_id] += 1
        
    logger.info("Start stream aggregation, writing results in real-time...")
    
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(judge_one, ji, args.temperature): ji for ji in all_tasks}
        
        # Progress bar monitoring task completion
        for future in tqdm(as_completed(future_to_task), total=total_tasks, desc="Global Evaluating"):
            ji = future_to_task[future]
            sid = ji.sample_id
            
            try:
                result = future.result()
                
                # Record results to memory map (Thread safety? output map should be locked, or dict itself is atomic)
                # Use lock for safety
                with progress_lock:
                    sample_results_map[sid][(ji.pair, ji.order)] = result
                    sample_progress[sid] += 1
                    current_done = sample_progress[sid]
                    target_total = sample_total_tasks[sid]
                
                # Real-time write raw results
                raw_record = {
                    "id": sid,
                    "domain": ji.domain,
                    **result,
                }
                with file_lock:
                    with open(raw_output_file, "a", encoding="utf-8") as f:
                        f.write(safe_json_dumps(raw_record) + "\n")
                
                # CHECK: Are all tasks for this Sample completed?
                if current_done == target_total:
                    # 🚀 Trigger aggregation!
                    sample = id_to_sample.get(sid)
                    if sample:
                        # Aggregation logic reuses previous one
                        results_chunk = sample_results_map[sid]
                        pair_results = {}
                        sample_error = False
                        
                        for pair in PAIR_LABELS:
                            r1 = results_chunk.get((pair, 1), {})
                            r2 = results_chunk.get((pair, 2), {})
                            w1 = r1.get("winner")
                            w2 = r2.get("winner")
                            
                            result_verdict = aggregate_pair_result(w1, w2)
                            pair_results[pair] = {
                                "order1_winner": w1,
                                "order2_winner": w2,
                                "result": result_verdict,
                            }
                            if result_verdict == "error":
                                sample_error = True
                        
                        final_record = {
                            "id": sid,
                            "domain": get_sample_field(sample, "domain", "query_type", default="general"),
                            "prompt": get_sample_field(sample, "prompt", default=""),
                            "pair_results": pair_results,
                        }
                        
                        # Write final result
                        with file_lock:
                            with open(args.output, "a", encoding="utf-8") as f:
                                f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                        
                        if not sample_error:
                            success_count += 1
                        else:
                            error_count += 1
                            
                        # (Optional) Release memory
                        del sample_results_map[sid]
                        del id_to_sample[sid]
                        
            except Exception as e:
                # Record error
                with progress_lock:
                    sample_results_map[sid][(ji.pair, ji.order)] = {"winner": None, "error": str(e)}
                
                error_record = {
                    "id": sid,
                    "domain": ji.domain,
                    "pair": ji.pair,
                    "order": ji.order,
                    "error": str(e),
                }
                with file_lock:
                    with open(error_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    # 4. No need for unified aggregation after loop, as it's done in stream
    # Only final statistics needed

    # Compute summary
    summary = compute_summary(args.output)
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print readable report
    print("\n")
    print("=" * 70)
    print("  RMBench Evaluation Report")
    print("=" * 70)
    
    # Global statistics
    print(f"\n  {'Mode':<10s} {'Win':>6s} {'Tie':>6s} {'Lose':>6s} {'Total':>6s} {'Win Rate':>10s} {'Net Win':>10s}")
    print("  " + "-" * 62)
    
    global_stats = summary.get("global", {}).get("by_mode", {})
    g_win, g_tie, g_lose, g_total = 0, 0, 0, 0
    for mode in ["easy", "normal", "hard"]:
        stats = global_stats.get(mode, {})
        win = stats.get('win', 0)
        tie = stats.get('tie', 0)
        lose = stats.get('lose', 0)
        total = stats.get('total', 0)
        wr = stats.get('win_rate', 0)
        nwr = stats.get('net_win_rate', 0)
        g_win += win; g_tie += tie; g_lose += lose; g_total += total
        print(f"  {mode:<10s} {win:>6d} {tie:>6d} {lose:>6d} {total:>6d} {wr:>9.2%} {nwr:>9.2%}")
    
    if g_total > 0:
        print("  " + "-" * 62)
        g_wr = g_win / g_total if g_total else 0
        g_nwr = g_win / (g_win + g_lose) if (g_win + g_lose) else 0
        print(f"  {'Total':<10s} {g_win:>6d} {g_tie:>6d} {g_lose:>6d} {g_total:>6d} {g_wr:>9.2%} {g_nwr:>9.2%}")
    
    # Statistics by Domain
    by_domain = summary.get("by_domain", {})
    if by_domain:
        print(f"\n  Statistics by Domain:")
        for domain in sorted(by_domain.keys()):
            print(f"\n  [{domain}]")
            domain_modes = by_domain[domain].get("by_mode", {})
            for mode in ["easy", "normal", "hard"]:
                stats = domain_modes.get(mode, {})
                win = stats.get('win', 0)
                tie = stats.get('tie', 0)
                lose = stats.get('lose', 0)
                total = stats.get('total', 0)
                wr = stats.get('win_rate', 0)
                nwr = stats.get('net_win_rate', 0)
                print(f"    {mode:<10s} {win:>5d} {tie:>5d} {lose:>5d} {total:>5d}  win={wr:.2%}  net={nwr:.2%}")
    
    print("\n" + "=" * 70)
    print(f"  Result file: {args.output}")
    print(f"  Summary file: {summary_file}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
