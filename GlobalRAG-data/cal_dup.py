import json
import re
import string
import pandas as pd


def extract_plan(output_str):
    """Extract content between <plan> and </plan>."""
    if not isinstance(output_str, str):
        return None
    pattern = r"<plan>(.*?)</plan>"
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    # 原代码里的值是1，因为prompt和output拼接起来，至少会存在prompt中的一个answer，这里
    # 只有output，所有修改为0
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(pred, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(pred)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def main():
    input_file = "bamboogle.jsonl"
    df = pd.read_json(input_file, lines=True)
    print("file load successfully")
    print("Total rows:", len(df))

    # ===============================
    # 1. 按 inputs 归类
    # ===============================
    grouped = df.groupby("input")

    total_after_dedup = 0
    correct_after_dedup = 0

    for input_value, group_df in grouped:
        # plan -> representative row
        plan_map = {}

        for _, row in group_df.iterrows():
            data = row.to_dict()
            output_text = data.get("output", "")
            golden_answers = data.get("golden_answer", [])

            plan = extract_plan(output_text)
            if plan is None:
                continue

            # 只保留第一次出现的 plan（去重）
            if plan not in plan_map:
                plan_map[plan] = data

        # ===============================
        # 2. 在去重后的 plan 上计算 EM
        # ===============================
        for plan, data in plan_map.items():
            output_text = data.get("output", "")
            golden_answers = data.get("golden_answer", [])

            pred_answer = extract_solution(output_text)
            if pred_answer is None:
                continue

            total_after_dedup += 1
            correct_after_dedup += em_check(pred_answer, golden_answers)

    # ===============================
    # 3. 输出统计结果
    # ===============================
    if total_after_dedup == 0:
        print("No valid predictions after deduplication.")
    else:
        accuracy = correct_after_dedup / total_after_dedup
        print(
            f"After deduplication by inputs + plan:\n"
            f"Total: {total_after_dedup}, "
            f"Correct: {correct_after_dedup}, "
            f"EM Accuracy: {accuracy:.4f}"
        )

if __name__ == "__main__":
    main()
