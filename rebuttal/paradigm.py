"""
paradigm.py — Three-paradigm LLM agent evaluation for rebuttal experiments.

Three interaction paradigms:
  1. NoRetrieval  : No search, answer directly from model parameters.
  2. SearchR1     : Iterative search-then-answer (SearchR1 style).
  3. GlobalRAG    : Hierarchical plan → subPlan → answer (GlobalRAG style).

Usage:
    python paradigm.py \
        --dataset hotpotqa \
        --split test \
        --llm_url http://127.0.0.1:8000/v1 \
        --search_url http://127.0.0.1:8181/retrieve \
        --paradigm all \          # one of: no_retrieval | search_r1 | global_rag | all
        --topk 3 \
        --max_turns 10 \
        --output_dir ./results
"""

import re
import json
import os
from typing import List, Dict, Any, Tuple, Optional

import requests
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

NO_RETRIEVAL_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Answer the question as accurately as possible based on your internal knowledge. "
    "Put your final answer inside <answer> and </answer>. "
    "For example: <answer> Beijing </answer>."
)

NO_RETRIEVAL_USER_TMPL = "Question: {question}"

# ----------

SEARCH_R1_USER_TMPL = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {question}"
)

# ----------

GLOBAL_RAG_USER_TMPL = (
    "Answer the given question by following the steps below. "
    "You must conduct all reasoning inside <think> and </think> before producing "
    "<plan>, <search>, <subAnswer> and <answer>.\n\n"
    "Step 1: Explicitly generate one or more sub-questions within the <plan> and </plan> block.\n"
    " - Each sub-question must contain both a question and a placeholder (#1, #2, etc.) "
    "that represents the answer to that question.\n"
    " - Each sub-question should be as brief and precise as possible.\n"
    " - If a sub-question depends on the answer to a previous one, use a placeholder "
    "(#1, #2, etc.) to represent that dependency.\n"
    " - The output format of the sub-questions must follow this JSON structure:\n"
    '{{\n'
    '    "Q1": ["First sub-question", "#1"],\n'
    '    "Q2": ["Second sub-question using #1", "#2"],\n'
    '    ...\n'
    '}}\n\n'
    "Step 2: For each sub-question, create a block enclosed in <subPlan> and </subPlan>.\n"
    "Within each <subPlan> block you must:\n"
    " - In sequential order, take one sub-question from <plan> and fill it between "
    "<search> and </search>.\n"
    " - If you lack some knowledge, call a search engine using <search> query </search>. "
    "The search engine will return results enclosed in <information> and </information>. "
    "You may search as many times as needed.\n"
    " - Conclude the block with a <subAnswer> that binds the answer to the current sub-question.\n\n"
    "Step 3: Provide the final result inside <answer> and </answer>, without detailed explanations.\n\n"
    "## One-Shot Example:\n"
    "Input:\n"
    "Who was the screenwriter of the film directed by the person who created the Money in the Bank ladder match?\n\n"
    "Output:\n"
    "<think> The question involves multiple entities and relations, so it is best decomposed into smaller "
    "sub-questions. First, I need to identify the creator of the Money in the Bank ladder match. Then, I should "
    "check which film that person directed. Finally, I must find the screenwriter of that film. </think>\n"
    "<plan>\n"
    '{{ "Q1": ["Who created the Money in the Bank ladder match?", "#1"], '
    '"Q2": ["Which film was directed by #1?", "#2"], '
    '"Q3": ["Who was the screenwriter of #2?", "#3"] }}\n'
    "</plan>\n\n"
    "<subPlan>\n"
    "    <think> To start, I need to find who created the Money in the Bank ladder match. </think>\n"
    "    <search> creator of the Money in the Bank ladder match </search>\n"
    "    <information> The Money in the Bank ladder match was created by Chris Jericho. </information>\n"
    "    <think> The information shows that Chris Jericho is the creator. </think>\n"
    "    <subAnswer> #1 = Chris Jericho </subAnswer>\n"
    "</subPlan>\n\n"
    "<subPlan>\n"
    "    <think> Next, I need to find which film Chris Jericho directed, based on the previous answer. </think>\n"
    "    <search> Which film was directed by Chris Jericho </search>\n"
    "    <information> Chris Jericho directed the film \"But I'm Chris Jericho!\". </information>\n"
    "    <think> The evidence indicates that the film directed by Chris Jericho is \"But I'm Chris Jericho!\". </think>\n"
    "    <subAnswer> #2 = \"But I'm Chris Jericho!\" </subAnswer>\n"
    "</subPlan>\n\n"
    "<subPlan>\n"
    "    <think> Finally, I should determine who wrote the film \"But I'm Chris Jericho!\". </think>\n"
    "    <search> \"But I'm Chris Jericho!\" film screenwriter </search>\n"
    "    <information> The series \"But I'm Chris Jericho!\" was written by Bob Kerr and Norm Hiscock. </information>\n"
    "    <think> The results confirm that the screenwriters of the film are Bob Kerr and Norm Hiscock. </think>\n"
    "    <subAnswer> #3 = Bob Kerr and Norm Hiscock </subAnswer>\n"
    "</subPlan>\n\n"
    "<think> I have gathered all the necessary information from the sub-questions and can now provide "
    "the final answer. </think>\n"
    "<answer> Bob Kerr and Norm Hiscock </answer>\n\n"
    "## Now, it's your turn! Please answer the following question!!!\n\n"
    "Question: {question}"
)


# ──────────────────────────────────────────────
# Search helper
# ──────────────────────────────────────────────

def batch_search(queries: List[str], search_url: str, topk: int = 3,
                 max_obs_length: int = 600) -> List[str]:
    """Call the retrieval server and return formatted passage strings.
    max_obs_length: truncate each observation to this many characters."""
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    response = requests.post(search_url, json=payload, timeout=30)
    response.raise_for_status()
    results = response.json()["result"]
    passages = [_passages2string(r) for r in results]
    return [p[:max_obs_length] for p in passages]


def _passages2string(retrieval_result: List[Dict]) -> str:
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx+1}(Title: {title}) {text}\n"
    return out.strip()


# ──────────────────────────────────────────────
# Extract final answer from model output
# ──────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract the last <answer>...</answer> span from model output."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


# ──────────────────────────────────────────────
# LLM call (vLLM OpenAI-compatible endpoint)
# ──────────────────────────────────────────────

def llm_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Returns (content, stop_reason).
    stop_reason is the stop token that triggered the stop (e.g. '</search>'),
    or None if the model stopped naturally (finish_reason='stop'/'length').
    The stop token is NOT included in content by vLLM, so callers must append it.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    stop_reason: Optional[str] = getattr(choice, "stop_reason", None)
    # fall back to finish_reason when stop_reason is absent
    if stop_reason is None and choice.finish_reason == "stop" and stop:
        # detect which stop token appears at the end of content
        for tok in stop:
            if content.rstrip().endswith(tok.rstrip()):
                stop_reason = tok
                break
    return content, stop_reason


# ──────────────────────────────────────────────
# Paradigm 1: No-Retrieval
# ──────────────────────────────────────────────

def run_no_retrieval(
    question: str,
    client: OpenAI,
    model: str,
    max_tokens: int = 500,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": NO_RETRIEVAL_SYSTEM},
        {"role": "user",   "content": NO_RETRIEVAL_USER_TMPL.format(question=question)},
    ]
    output, _ = llm_chat(client, model, messages, max_tokens=max_tokens,
                         stop=["</answer>"])
    output += "</answer>"   # stop token not included by vLLM, append it back
    return {
        "paradigm": "no_retrieval",
        "question": question,
        "output": output,
        "answer": extract_answer(output),
        "num_searches": 0,
        "trajectory": [{"turn": 0, "model_output": output}],
    }


# ──────────────────────────────────────────────
# Paradigm 2: SearchR1 (iterative search)
# ──────────────────────────────────────────────


def run_search_r1(
    question: str,
    client: OpenAI,
    model: str,
    search_url: str,
    topk: int = 3,
    max_turns: int = 5,
    max_tokens: int = 500,
    max_obs_length: int = 600,
) -> Dict[str, Any]:
    user_content = SEARCH_R1_USER_TMPL.format(question=question)
    messages = [{"role": "user", "content": user_content}]
    assistant_content = ""
    num_searches = 0
    final_answer = ""
    trajectory: list[dict[str, object]] = []
    stop_tokens = ["</search>", "</answer>"]

    for turn in range(max_turns):
        messages_for_gen = messages + [{"role": "assistant", "content": assistant_content}] \
            if assistant_content else messages

        new_text, stop_reason = llm_chat(
            client, model, messages_for_gen,
            max_tokens=max_tokens, stop=stop_tokens,
        )
        # vLLM does not include the stop token in content — append it back
        if stop_reason in stop_tokens:
            new_text += stop_reason

        assistant_content += new_text

        if stop_reason == "</answer>":
            match = re.search(r"<answer>(.*?)</answer>", new_text, re.DOTALL)
            final_answer = match.group(1).strip() if match else ""
            trajectory.append({"turn": turn, "model_output": new_text, "action": "answer"})
            break
        elif stop_reason == "</search>":
            match = re.search(r"<search>(.*?)</search>", new_text, re.DOTALL)
            query = match.group(1).strip() if match else ""
            num_searches += 1
            retrieved = batch_search([query], search_url, topk, max_obs_length)
            info_block = f"\n\n<information>{retrieved[0]}</information>\n\n"
            assistant_content += info_block
            trajectory.append({
                "turn": turn,
                "model_output": new_text,
                "action": "search",
                "query": query,
                "retrieved": retrieved[0],
            })
        else:
            # max_length or no valid stop token — stop
            trajectory.append({"turn": turn, "model_output": new_text, "action": None})
            break

    return {
        "paradigm": "search_r1",
        "question": question,
        "output": assistant_content,
        "answer": final_answer or extract_answer(assistant_content),
        "num_searches": num_searches,
        "trajectory": trajectory,
    }


# ──────────────────────────────────────────────
# Paradigm 3: GlobalRAG (plan → subPlan → answer)
# ──────────────────────────────────────────────

def run_global_rag(
    question: str,
    client: OpenAI,
    model: str,
    search_url: str,
    topk: int = 3,
    max_turns: int = 5,
    max_tokens: int = 500,
    max_obs_length: int = 600,
) -> Dict[str, Any]:
    """
    GlobalRAG uses a multi-turn approach where the model:
      1. Emits a <plan> with sub-questions.
      2. For each sub-question emits a <subPlan> that may contain multiple <search> calls
         and ends with a <subAnswer>.
      3. Finally emits an <answer>.

    We feed the accumulated context back after each search so the model sees
    the retrieved information while filling sub-plans.
    """
    user_content = GLOBAL_RAG_USER_TMPL.format(question=question)
    messages = [{"role": "user", "content": user_content}]
    assistant_content = ""
    num_searches = 0
    final_answer = ""
    trajectory: list[dict[str, object]] = []
    stop_tokens = ["</search>", "</answer>"]

    for turn in range(max_turns):
        messages_for_gen = messages + [{"role": "assistant", "content": assistant_content}] \
            if assistant_content else messages

        new_text, stop_reason = llm_chat(
            client, model, messages_for_gen,
            max_tokens=max_tokens, stop=stop_tokens,
        )
        # vLLM does not include the stop token in content — append it back
        if stop_reason in stop_tokens:
            new_text += stop_reason

        assistant_content += new_text

        if stop_reason == "</answer>":
            final_answer = extract_answer(new_text)
            trajectory.append({"turn": turn, "model_output": new_text, "action": "answer"})
            break
        elif stop_reason == "</search>":
            match = re.search(r"<search>(.*?)</search>", new_text, re.DOTALL)
            query = match.group(1).strip() if match else ""
            if query:
                num_searches += 1
                retrieved = batch_search([query], search_url, topk, max_obs_length)
                info_block = f"\n<information>{retrieved[0]}</information>\n"
                assistant_content += info_block
                trajectory.append({
                    "turn": turn,
                    "model_output": new_text,
                    "action": "search",
                    "query": query,
                    "retrieved": retrieved[0],
                })
            else:
                trajectory.append({"turn": turn, "model_output": new_text, "action": "search_empty"})
        else:
            final_answer = extract_answer(new_text)
            trajectory.append({"turn": turn, "model_output": new_text, "action": None})
            break

    return {
        "paradigm": "global_rag",
        "question": question,
        "output": assistant_content,
        "answer": final_answer or extract_answer(assistant_content),
        "num_searches": num_searches,
        "trajectory": trajectory,
    }


# ──────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────

DATASET_CONFIGS = {
    "hotpotqa":       ("hotpotqa/hotpot_qa", "distractor", "validation", "question"),
    "2wikimultihopqa":("wikimedia/wikipedia",  None,        "train",     "question"),   # placeholder
    "musique":        ("drt/musique",          None,        "validation", "question"),
    "bamboogle":      ("RUC-NLPIR/FlashRAG",   "bamboogle", "test",      "question"),
    "nq":             ("nq_open",              None,        "validation", "question"),
    "triviaqa":       ("trivia_qa",            "unfiltered", "validation","question"),
}


def load_parquet(data_path: str, max_samples: int = None) -> List[Dict]:
    """Load questions from a local parquet file produced by the SearchR1 pipeline.

    Expected columns: question, golden_answers, reward_model, id (optional).
    The 'prompt' column contains the full prompt with instructions prepended —
    we ignore it and use the raw 'question' field directly.
    golden_answers is a list; we keep all candidates for multi-answer EM/F1.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.iloc[:max_samples]

    records = []
    for idx, row in df.iterrows():
        question = str(row.get("question", ""))

        # Collect all gold answers from golden_answers or reward_model.ground_truth.target
        golden = row.get("golden_answers", None)
        if golden is None:
            rm = row.get("reward_model", {})
            golden = rm.get("ground_truth", {}).get("target", [])

        if isinstance(golden, np.ndarray):
            golden = golden.tolist()
        if isinstance(golden, str):
            golden = [golden]
        golden = [str(a).strip() for a in golden if str(a).strip()]

        records.append({
            "id": str(row.get("id", idx)),
            "question": question,
            "answers": golden,          # list of all valid answers
            "answer": golden[0] if golden else "",  # primary answer
        })
    return records


def load_questions(dataset_name: str, split: str, max_samples: int = None) -> List[Dict]:
    """Load questions from a HuggingFace dataset.

    Returns a list of dicts with at least 'question', 'answer', 'answers' keys.
    """
    cfg = DATASET_CONFIGS.get(dataset_name.lower())
    if cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )

    hf_name, hf_config, default_split, q_field = cfg
    actual_split = split or default_split

    if hf_config:
        ds = load_dataset(hf_name, hf_config, split=actual_split, trust_remote_code=True)
    else:
        ds = load_dataset(hf_name, split=actual_split, trust_remote_code=True)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    records = []
    for idx, row in enumerate(ds):
        question = row.get(q_field, row.get("question", ""))
        answer = (
            row.get("answer")
            or row.get("answers")
            or row.get("answer_aliases")
            or ""
        )
        if isinstance(answer, list):
            answers = [str(a).strip() for a in answer if str(a).strip()]
            answer = answers[0] if answers else ""
        else:
            answers = [str(answer).strip()] if answer else []
        records.append({
            "id": row.get("id", str(idx)),
            "question": question,
            "answer": answer,
            "answers": answers,
        })
    return records


# ──────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em(pred: str, answers: List[str]) -> float:
    """Return 1.0 if pred matches any answer in the list."""
    return float(any(exact_match(pred, g) for g in answers))


def best_f1(pred: str, answers: List[str]) -> float:
    """Return the maximum F1 over all candidate answers."""
    return max((f1_score(pred, g) for g in answers), default=0.0)


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────

PARADIGM_FNS = {
    "no_retrieval": run_no_retrieval,
    "search_r1":    run_search_r1,
    "global_rag":   run_global_rag,
}


def run_experiment(args):
    client = OpenAI(base_url=args.llm_url, api_key="EMPTY")
    model  = args.model
    print(f"[INFO] Using model: {model} @ {args.llm_url}")

    if args.data_path:
        records = load_parquet(args.data_path, args.max_samples)
        dataset_tag = os.path.splitext(os.path.basename(args.data_path))[0]
        print(f"[INFO] Loaded {len(records)} questions from '{args.data_path}'")
    else:
        records = load_questions(args.dataset, args.split, args.max_samples)
        dataset_tag = f"{args.dataset}_{args.split}"
        print(f"[INFO] Loaded {len(records)} questions from '{args.dataset}/{args.split}'")

    paradigms = (
        list(PARADIGM_FNS.keys()) if args.paradigm == "all" else [args.paradigm]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries: List[Dict[str, Any]] = []

    # Records used by retrieval paradigms — filtered to hard questions after no_retrieval runs
    retrieval_records = records   # default: use all records
    no_ret_results_map: Dict[str, Dict] = {}  # id -> result, for filtering

    for paradigm in paradigms:
        print(f"\n{'='*60}")
        print(f"[Paradigm] {paradigm}")
        print(f"{'='*60}")

        # For search_r1 / global_rag: only evaluate on questions no_retrieval got wrong
        if paradigm in ("search_r1", "global_rag") and no_ret_results_map:
            eval_records = [r for r in records if no_ret_results_map.get(str(r["id"]), {}).get("em", 1) == 0]
            print(f"[INFO] Filtering to hard questions: {len(eval_records)}/{len(records)} "
                  f"(no_retrieval EM=0)")
        else:
            eval_records = records

        results = []
        em_total, f1_total = 0.0, 0.0

        pbar = tqdm(eval_records, desc=f"[{paradigm}]", unit="q")
        for i, record in enumerate(pbar):
            question = record["question"]
            answers  = record.get("answers") or [record["answer"]]

            # Dispatch
            if paradigm == "no_retrieval":
                res = run_no_retrieval(
                    question, client, model,
                    max_tokens=args.max_tokens_no_retrieval,
                )
            elif paradigm == "search_r1":
                res = run_search_r1(
                    question, client, model,
                    search_url=args.search_url,
                    topk=args.topk,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    max_obs_length=args.max_obs_length,
                )
            elif paradigm == "global_rag":
                res = run_global_rag(
                    question, client, model,
                    search_url=args.search_url,
                    topk=args.topk,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens_global_rag,
                    max_obs_length=args.max_obs_length,
                )
            else:
                raise ValueError(f"Unknown paradigm: {paradigm}")

            pred = res["answer"]
            em = best_em(pred, answers)
            f1 = best_f1(pred, answers)
            em_total += em
            f1_total += f1

            res["answers"] = answers
            res["em"] = em
            res["f1"] = f1
            res["id"] = record.get("id", str(i))
            results.append(res)

            # Build lookup for downstream filtering
            if paradigm == "no_retrieval":
                no_ret_results_map[str(res["id"])] = res

            pbar.set_postfix(
                EM=f"{em_total/(i+1):.4f}",
                F1=f"{f1_total/(i+1):.4f}",
                searches=f"{sum(r['num_searches'] for r in results)/(i+1):.2f}",
            )

        total_n = len(records)
        n = len(results)
        summary = {
            "paradigm":    paradigm,
            "dataset":     dataset_tag,
            "num_total":   total_n,
            "num_eval":    n,   # for search_r1/global_rag: hard questions only
            "EM":          em_total / n,
            "F1":          f1_total / n,
            "avg_searches": sum(r["num_searches"] for r in results) / n,
        }
        print(f"\n[Summary] {summary}")
        all_summaries.append(summary)

        # Save
        out_path = os.path.join(
            args.output_dir, f"{paradigm}_{dataset_tag}.jsonl"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_path = os.path.join(
            args.output_dir, f"{paradigm}_{dataset_tag}_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Results saved to {out_path}")

    # ── Final comparison table ──────────────────────────────────────
    col_w = 16
    print(f"\n{'='*68}")
    print(f"  Final Results on [{dataset_tag}]")
    print(f"  (search_r1 / global_rag: evaluated only on hard questions where no_retrieval EM=0)")
    print(f"{'='*68}")
    print(f"  {'Paradigm':<{col_w}} {'N':>6} {'EM':>8} {'F1':>8} {'Avg Searches':>14}")
    print(f"  {'-'*col_w} {'------':>6} {'--------':>8} {'--------':>8} {'----------':>14}")
    for s in all_summaries:
        print(f"  {s['paradigm']:<{col_w}} {s['num_eval']:>6} {s['EM']:>8.4f} {s['F1']:>8.4f} {s['avg_searches']:>14.2f}")
    print(f"{'='*68}\n")


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

class Config:
    data_path              = "/ssd2/lini03/Search-R1-infer/test_model/data/bamboogle.parquet"
    dataset                = "hotpotqa"       # fallback if data_path is None
    split                  = "validation"     # fallback if data_path is None
    llm_url                = "http://127.0.0.1:8112/v1"
    search_url             = "http://127.0.0.1:8181/retrieve"
    model                  = "qwen3-32b"
    paradigm               = "all"            # no_retrieval | search_r1 | global_rag | all
    topk                   = 3
    max_turns              = 5
    max_tokens_no_retrieval = 500             # no_retrieval uses shorter output
    max_tokens             = 2048             # search_r1
    max_tokens_global_rag  = 8192             # global_rag: first turn must emit <think>+<plan>+<subPlan>+<search> before any stop token fires
    max_obs_length         = 600
    max_samples            = None             # None = all
    output_dir             = "./rebuttal_results"


if __name__ == "__main__":
    run_experiment(Config())
