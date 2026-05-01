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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import os
import re
import numpy as np
import torch
import ray
import hydra

from verl import DataProto


# ---- GlobalRAG PSE-based RewardManager ----

def _select_rm_score_fn(data_source):
    from verl.utils.reward_score import qa_em
    if data_source in ['2WikiMultihopQA', 'Multihop-RAG', 'nq', 'triviaqa', 'popqa',
                       'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'multi_hop_qa']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class PSERewardManager():
    """GlobalRAG reward manager with PSE (Plan-Step Evaluation) scoring."""

    def __init__(self, tokenizer, num_examine, e5_model_path, format_score=0., validation=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.validation = validation

        from verl.utils.reward_score.pse.pse_score import PseScore
        from verl.utils.reward_score.pse.flashrag.config.config import Config as PSE_Config
        config = PSE_Config(config_dict={'e5_model_path': e5_model_path})
        self.PSE_Evaluator = PseScore(config)
        print(f"PSE_Evaluator initialized. e5_model_path: {e5_model_path}")

    def __call__(self, data: DataProto, step: int = 0):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            meta_data = data_item.non_tensor_batch.get('metadata', {})
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score_dict = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                meta_data=meta_data,
                pse_evaluator=self.PSE_Evaluator,
                step=step,
            )

            if self.validation:
                reward_tensor[i, valid_response_length - 1] = score_dict['answer_score']
            else:
                reward_tensor[i, valid_response_length - 1] = score_dict['final_score']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


# ---- custom reward function loader ----

def get_custom_reward_fn(config):
    import importlib.util, sys
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

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
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # ---- reward manager selection ----
        # Use PSERewardManager when e5_model_path is configured, otherwise use standard managers
        e5_model_path = config.get('e5_model_path', None)
        if e5_model_path:
            reward_fn = PSERewardManager(
                tokenizer=tokenizer, num_examine=0, e5_model_path=e5_model_path)
            val_reward_fn = PSERewardManager(
                tokenizer=tokenizer, num_examine=1, e5_model_path=e5_model_path, validation=True)
        else:
            reward_manager_name = config.reward_model.get("reward_manager", "naive")
            if reward_manager_name == 'naive':
                from verl.workers.reward_manager import NaiveRewardManager
                reward_manager_cls = NaiveRewardManager
            elif reward_manager_name == 'prime':
                from verl.workers.reward_manager import PrimeRewardManager
                reward_manager_cls = PrimeRewardManager
            elif reward_manager_name == 'batch':
                from verl.workers.reward_manager import BatchRewardManager
                reward_manager_cls = BatchRewardManager
            elif reward_manager_name == 'dapo':
                from verl.workers.reward_manager import DAPORewardManager
                reward_manager_cls = DAPORewardManager
            else:
                raise NotImplementedError

            compute_score = get_custom_reward_fn(config)
            reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
            reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                           num_examine=1,
                                           compute_score=compute_score,
                                           reward_fn_key=config.data.reward_fn_key,
                                           **reward_kwargs)
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                               num_examine=1,
                                               compute_score=compute_score,
                                               reward_fn_key=config.data.reward_fn_key)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()

