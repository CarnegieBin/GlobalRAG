import json
import re
import string
import pandas as pd

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
    input_file = "./searchr1_hotpotqa_qwen_3b_it_wf/0.jsonl"
    total = 0
    correct = 0
    df = pd.read_json(input_file, lines=True)
    print("file load successfully")
    print(len(df))

    for idx, row in df.iterrows():  # row 是 Series
        data = row.to_dict()  # 转成字典
        output_text = data.get("output", "")
        golden_answers = data.get("golden_answer", [])

        pred_answer = extract_solution(output_text)
        if pred_answer is None:
            continue

        total += 1
        correct += em_check(pred_answer, golden_answers)

    if total == 0:
        print("No valid predictions found.")
    else:
        accuracy = correct / len(df)
        print(f"Total: {total}, Correct: {correct}, EM Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

