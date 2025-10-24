# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

rm_prompt="""Background Knowledge:
The following is a deep-research scenario:
Lines beginning with user indicate user questions.
Lines beginning with assistant represent the deep-research agent’s reasoning content.
The content inside <think>xxx</think> represents the agent’s reasoning process.
The content inside <serach>xxx</search> shows the query which the deep-research agent decides to invoke a search tool to retrieve after reasoning.
The content inside <information>xxx</information> indicates the search result returned from the retriever.
The content inside <answer>xxx</answer> represents the final answer.

Task [TASK]: You are a superintelligent agent expert—smarter than the deep-research agent. Your task is to evaluate the deep-research agent's reasoning and tool usage based on the following scoring criteria:

【Evaluation Dimensions】
1. **Logical Reasoning Quality** (0–5 points): 
    - Evaluate hypothesis formulation, evidence use, and consistency of conclusions.
    - 5 points: Fully deductive, tightly justified reasoning chains with evidence support.
    - 3 points: Acceptable logical leaps, but not rigorously justified.
    - 0 points: Broken chains, circular reasoning, or major logical flaws.
2. **Search Strategy Intelligence** (0–5 points):
    - Evaluate the accuracy of the searched entity, the inevitable connection between the search query and problem-solving, and appropriateness of time or other filters.
    - 5 points: Keyword variation and improvement across search rounds that meaningfully enhance information retrieval.
    - 3 points: Basic keyword search with no advanced filtering.
    - 0 points: Repeated or irrelevant query; invalid tool usage.

【Input Data】
Research Process Record: {process_str}

【Output Requirements】
1. Output must be in JSON format with three evaluation scores.
2. Use the following keys:
    - "Logical Reasoning Quality"
    - "Search Strategy Intelligence"
3. Append a brief defect analysis (at most 100 words) in English.

【Correct Example】
Scoring shows limited search coverage (3/5) and reasoning includes unverified assumptions (4/5). Defect Analysis: The main limitations include: (1) Lack of recent clinical trials after 2023. (2) Unverified assumptions about pharmacokinetic parameters.
Final Score Output:
\\boxed{{
    "Logical Reasoning Quality": 0,
    "Search Strategy Intelligence": 0,
    "analysis": "Explanation in both English..."
}}
"""

import re
import string
import random
from typing import Optional, List

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


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False

# from openai import OpenAI
# client = OpenAI(
#     base_url = 'http://127.0.0.1:8866/v1',
#     api_key='dummy_key', # required, but unused
# )
# def rm_judge(text: str) -> Optional[float]:
#     try:
#         prompt = rm_prompt.format(process_str=text)
#         response = client.chat.completions.create(
#             model="/data3/workhome/wangzhizhou/data/tss/Qwen3-30B-A3B",
#             messages=[
#                 {"role": "user", "content": prompt},
#             ]
#         ).choices[0].message.content
#         # print("Response:", response)

#         response = response.split("</think>", 1)[-1]
#         think_score = re.search(r'"Logical Reasoning Quality":\s*(\d+)', response)
#         search_score = re.search(r'"Search Strategy Intelligence":\s*(\d+)', response)
#         if think_score and search_score:
#             think_score = int(think_score.group(1))
#             search_score = int(search_score.group(1))
#             print(f"think_score: {think_score}, search_score: {search_score}")
#             if think_score >= 0 and think_score <= 5  and search_score >= 0 and search_score <= 5:
#                 return 0.1 * (think_score + search_score)
#         return 0.6
#     except Exception:       # 任何异常都吃掉
#         return 0.6

def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, rm_coef=0, rm_score = 0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    # retrieval_correct = False
    # if is_valid_format:
    #     retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"RM score: {rm_score}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    outcome_score = 0.0
    if em_check(answer, ground_truth['target']):
        outcome_score = 1.0
    
    score = outcome_score + rm_score
    return score
    # if answer is None:
    #     if is_valid_format:
    #         return structure_format_score # 0.2
    #     else:
    #         return 0
    # else:
    #     if em_check(answer, ground_truth['target']):
    #         if is_valid_format:
    #             return score # 1
    #         else:
    #             return score - structure_format_score # 0.8
    #     elif rm_coef > 0:
    #             if is_valid_format:
    #                 return structure_format_score + rm_score * rm_coef
    #             else:
    #                 return rm_score * rm_coef
    #     else:
    #         if is_valid_format:
    #             return structure_format_score
    #         else:
    #             return 0
