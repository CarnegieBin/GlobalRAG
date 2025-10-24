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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""


from verl import DataProto
import torch
from verl.utils.reward_score import qa_em_format_rm
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import traceback, sys
import numpy as np

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
    "analysis": "Explanation in English..."
}}
"""


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format_rm.compute_score_em
    else:
        raise NotImplementedError

from openai import OpenAI
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

def rm_judge(text: str) -> Optional[float]:
    try:
        client = OpenAI(
            base_url = 'http://127.0.0.1:8866/v1',
            api_key='dummy_key', # required, but unused
        )

        prompt = rm_prompt.format(process_str=text)
        response = client.chat.completions.create(
            model="/home/ssd3/jchluo/llm_models/Qwen2.5-3B-Instruct",
            messages=[
                {"role": "user", "content": prompt},
            ]
        ).choices[0].message.content
        print("Response:", response)

        response = response.split("</think>", 1)[-1]
        think_score = re.search(r'"Logical Reasoning Quality":\s*(\d+)', response)
        search_score = re.search(r'"Search Strategy Intelligence":\s*(\d+)', response)
        if think_score and search_score:
            think_score = int(think_score.group(1))
            search_score = int(search_score.group(1))
            print(f"think_score: {think_score}, search_score: {search_score}")
            if think_score >= 0 and think_score <= 5  and search_score >= 0 and search_score <= 5:
                return 0.1 * (think_score + search_score)
        return 0.6
    except Exception as e:       # 任何异常都吃掉
        traceback.print_exc(file=sys.stderr)
        return 0.6
    
def call_rm(sequences: List[str], ground_truths: List[str]) -> List[float]:
    """
    并行打分
    """
    if len(sequences) != len(ground_truths):
        raise ValueError("sequences 和 ground_truths 长度必须一致")
    
    # 并发上限，可根据本地/远端并发能力调整
    max_workers = 64

    rm_scores: List[float] = [0.0] * len(sequences)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(rm_judge, seq): idx
            for idx, seq in enumerate(sequences)
        }

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                rm_scores[idx] = fut.result()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                rm_scores[idx] = 0.6

    return rm_scores

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., format_score=0., rm_coef=0.,) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.rm_coef = rm_coef

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     return data.batch['rm_scores']

        if  self.rm_coef > 0:
            tmp_seq = []
            tmp_gt = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                tmp_gt.append(ground_truth)
                tmp_seq.append(sequences_str)
            rm_scores = call_rm(tmp_seq, tmp_gt)
        else:
            rm_scores = ["" for _ in range(len(data))]
        

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                     structure_format_score=self.structure_format_score, 
                                     rm_coef=self.rm_coef,
                                     rm_score=rm_scores[i] if self.rm_coef > 0 else None)

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              rm_coef=config.reward_model.rm_coef,)

    # Note that we always use function-based RM for validation
    from verl.trainer.main_ppo import RewardManager as TestRewardManager
    val_reward_fn = TestRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
