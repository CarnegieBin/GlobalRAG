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

import re
import string
import random
import re
import json
from typing import List, Tuple
import Levenshtein
import numpy as np
import traceback

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
    if prediction is None:
        return 0
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


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


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
    
    # If there are 0 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


OPEN_TAG_RE = re.compile(r'<\s*(think|search|information|answer)\b', re.IGNORECASE)


def extract_tag_sequence(text: str) -> List[str]:
    """提取文本中按出现顺序排列的打开标签名（小写）"""
    return [m.group(1).lower() for m in OPEN_TAG_RE.finditer(text)]


def check_sequence(tags: List[str]) -> bool:
    """
    校验是否满足模式：
       (<think><search><information>)* <think><answer>
    并确保：<answer> 恰好出现 1 次，且位于最后两项。
    """
    # 1) <answer> 只能出现一次
    if tags.count('answer') != 1:
        return False

    i, n = 0, len(tags)

    # 2) 消耗 0 次或多次的三元组：think, search, information
    while i + 2 < n and tags[i:i+3] == ['think', 'search', 'information']:
        i += 3

    # 3) 现在必须正好剩下两个标签：think, answer
    if i + 1 >= n:
        return False
    if tags[i:i+2] != ['think', 'answer']:
        return False

    # 4) 且没有多余标签
    return (i + 2) == n


def check_format(text: str) -> bool:
    """
    返回 (是否符合, 说明/错误原因)
    """
    errors = []
    # 1. 检查 <plan> 是否存在，且前面有 <think>
    plan_match = re.search(r"<think>.*?</think>\s*<plan>.*?</plan>", text, re.S)
    if not plan_match:
        errors.append("Missing <plan> ... </plan> block or missing preceding <think>.")

    # 2. 检查 <answer> 是否存在，且前面有 <think>
    answer_match = re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", text, re.S)
    if not answer_match:
        errors.append("Missing <answer> ... </answer> block or missing preceding <think>.")

    # 3. 检查 subPlan 是否存在（至少 1 个）
    subplans = re.findall(r"<subPlan>.*?</subPlan>", text, re.S)
    if not subplans:
        errors.append("No <subPlan> ... </subPlan> blocks found.")

    # 4. 检查每个 subPlan 内部结构
    for i, sp in enumerate(subplans, 1):
        if len(re.findall(r"<think>.*?</think>", sp, re.S)) < 1:
            errors.append(f"<subPlan> {i} missing <think> block(s).")
        if not re.search(r"<search>.*?</search>", sp, re.S):
            errors.append(f"<subPlan> {i} missing <search> ... </search>.")
        if not re.search(r"<information>.*?</information>", sp, re.S):
            errors.append(f"<subPlan> {i} missing <information> ... </information>.")
        if not re.search(r"<subAnswer>.*?</subAnswer>", sp, re.S):
            errors.append(f"<subPlan> {i} missing <subAnswer> ... </subAnswer>.")
    return True if not errors else False


def replace_placeholders(d: dict) -> dict:
    """ Replace placeholders like <#1> with <A1> in each value of the dictionary."""
    new_dict = {}
    for k, v in d.items():
        new_list = []
        for item in v:
            # 用正则替换 <A数字> → #数字
            new_item = re.sub(r"#(\d+)", r"<A\1>", item)
            new_list.append(new_item)
        new_dict[k] = new_list
    return new_dict


def plan_similar(predict_plan, golden_plan):
    """
    The scoring function for planning.
    Args:
        predict_plan: plan of rollout
        golden_plan: golden plan
    """
    ## 直接用Levenshtein 计算编辑距离相似度
    if len(predict_plan) != len(golden_plan):
        print(f'plan_similar, plan length not equal, predict: {len(predict_plan)}, golden: {len(golden_plan)}')
        return 0.0

    predict_plan = replace_placeholders(predict_plan)  # 特殊的占位符统一
    predict_plan_querys = [' '.join(val) for key, val in predict_plan.items()]
    golden_plan_querys = [' '.join(val) for key, val in golden_plan.items()]
    similarity_score = 0.0
    for predict_query in predict_plan_querys:
        for golden_query in golden_plan_querys:
            similarity_score += Levenshtein.ratio(predict_query, golden_query)
    length = len(predict_plan_querys) ** 2
    return similarity_score / length


def compute_plan_score(solution_str, meta_data):
    """The scoring function for planning.

    Args:
        solution_str: the solution text
        meta_data: the meta-data for each sample
    """
    # plan_pattern = re.compile(r'```json(.+?)```', re.DOTALL)
    plan_pattern = re.compile(r'<plan>(.+?)</plan>', re.DOTALL)
    _match = plan_pattern.findall(solution_str)
    if not _match: # 没有执行Query拆解
        return 0.0
    try:
        plan = json.loads(_match[-1].replace("json", "").replace("```", ""))
        if not meta_data.get('plan', {}):  # golden plan is empty
            return 1.0 if len(plan) > 1 else 0.0
        plan_score = plan_similar(plan, meta_data['plan'])
        return plan_score
    except Exception as e:
        print(f'compute_plan_score, plan parse error: {e}')
    return 0.0


def compute_format_score(solution_str):
    """
        The scoring function for formatting.

        Args:
            solution_str: the solution text
    """
    try:
        # _ouput_idx = solution_str.rindex('<subquery>')
        # response = solution_str[_ouput_idx:]
        if check_format(solution_str):
            return 1.0
    except Exception as e:
        print(f'compute_format_score, not good response: {solution_str}')
    return 0.0


def fix_label(golden_label):
    """  过滤掉golden label中出现value为None的key, 防止PSE计算报异常"""
    new_dict = {}
    for key, value in golden_label.items():
        if value is None:
            continue
        new_dict[key] = value
    return new_dict


def compute_pse_reward(solution: str, meta_data, pse_evaluator):
    """ The reward func for PSE."""
    golden_plan = meta_data.get('plan', {})
    golden_graph = meta_data.get('graph', [])
    if isinstance(golden_graph, np.ndarray):  # 兼容np.ndarray类型
        golden_graph = golden_graph.tolist()

    if not golden_plan or not golden_graph:  # 没有golden plan or graph
        print('compute_pse_reward, no golden_plan or golden_graph')
        return 0.0, 0.0, 0.0

    golden_plan = fix_label(golden_plan)
    golden_graph = [fix_label(g) for g in golden_graph]

    try:
        # 解析全局Plan
        plan_pattern = re.compile(r'<plan>(.+?)</plan>', re.DOTALL)
        _match = plan_pattern.findall(solution)
        if len(_match) == 0: # plan在solution中不存在
            print(f'compute_pse_reward, predict plan not found: {solution}')
            return 0.0, 0.0, 0.0
        predict_plan = json.loads(_match[-1].replace("json", "").replace("```", ""))
        predict_plan = replace_placeholders(predict_plan)  # 特殊的占位符统一

        # 解析推理过程subPlan
        predict_graph = {}
        sub_plan_pattern = re.compile(r'<subPlan>(.+?)</subPlan>', re.DOTALL)
        sub_plans = sub_plan_pattern.findall(solution, re.DOTALL)
        for sub_plan in sub_plans:
            # 解析 <subAnswer> #1 = Chris Jericho </subAnswer> 中的 #1 和 Chris Jericho
            ans_pattern = r"<subAnswer>\s*#(\d+)\s*=\s*(.*?)\s*</subAnswer>"
            ans_match = re.findall(ans_pattern, sub_plan, re.DOTALL)
            if len(ans_match) > 0:
                question_id = ans_match[0][0]
                sub_answer = ans_match[0][1].strip()
                predict_graph[f'Q{question_id}'] = {
                    'answer': sub_answer
                }
        if len(predict_graph) == 0:
            print(f'compute_pse_reward, predict graph not found: {solution}')
        predict_graph = [predict_graph]
        print_info = {
            'predict_plan': predict_plan, 'predict_graph': predict_graph,
            'golden_plan': golden_plan, 'golden_graph': golden_graph
        }
        print(f'compute_pse_reward, print_info: {print_info}')
        plan_sim_score, plan_structure_score, step_score = pse_evaluator.calculate_pse_score(predict_plan,
                                                                                             golden_plan,
                                                                                             predict_graph,
                                                                                             golden_graph)
        return plan_sim_score, plan_structure_score, step_score
    except Exception as e:
        print(f'compute_pse_reward, error: {e}')
        traceback.print_exc()
    return 0.0, 0.0, 0.0


def a_t_func(t, T=50, k=10.0, center=0.9):
    """
    Dynamic weight schedule from the paper image:
        a_t = 1 / (1 + exp((t - center*T) / k))
    Args:
        t: current training step (scalar or numpy array)
        T: total number of training steps
        k: temperature (controls steepness). Default 10.
        center: where the curve is centered as a fraction of T. Default 0.9 (i.e., 90% of T)
    """
    t = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + np.exp((t - center * T) / k)).item()


def compute_score_em(solution_str, ground_truth, meta_data={}, pse_evaluator=None, step=0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        meta_data: the meta-data for each sample
        pse_evaluator: the pse evaluator
        step: the current training step
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    _format_score = compute_format_score(solution_str)
    answer_score = em_check(answer, ground_truth['target'])
    if pse_evaluator:
        plan_sim_score, plan_structure_score, step_score = compute_pse_reward(solution_str, meta_data, pse_evaluator)
    else:
        plan_sim_score = compute_plan_score(solution_str, meta_data)
        plan_structure_score = 0.0
        step_score = 0.0

    decay_weight = a_t_func(step)
    final_score = decay_weight * (0.1 * _format_score + 0.5 * plan_sim_score + 0.5 * plan_structure_score + 0.5 * step_score) + answer_score
    # final_score = (0.1 * _format_score + 0.5 * plan_sim_score + 0.5 * plan_structure_score + 0.5 * step_score) + answer_score
    # final_score = decay_weight * (0.1 * _format_score + 0.5 * plan_sim_score + 0.5 * step_score) + answer_score
    # final_score = decay_weight * (0.1 * _format_score + 0.5 * plan_structure_score + 0.5 * step_score) + answer_score
    print(f"format_score: {_format_score}, "
          f"plan_sim_score: {plan_sim_score}, "
          f"plan_structure_score: {plan_structure_score}, "
          f"step_score: {step_score}, "
          f"answer_score: {answer_score}, "
          f"final_score: {final_score}")
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    return {
        'format_score': _format_score,
        'plan_sim_score': plan_sim_score,
        'plan_structure_score': plan_structure_score,
        'step_score': step_score,
        'answer_score': answer_score,
        'final_score': final_score
    }


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
