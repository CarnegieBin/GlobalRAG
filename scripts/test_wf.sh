train_data_name=nq_hotpotqa_train
test_data_name=nq_search

export CUDA_VISIBLE_DEVICES=2,3,4,6
export TRAIN_DATA_DIR=/home/ssd3/wanfan01/pse_v2/Search-R1-main/data/nq_hotpotqa_train # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
export TEST_DATA_DIR=/home/ssd3/chengmingquan/Search-R1-main/test_grpo_models/searchr1_data

WAND_PROJECT='Search-R1-GroupSize'
# export BASE_MODEL='/home/ssd3/wanfan01/pse_v2/Search-R1-main/verl_checkpoints/nq_hotpotqa_train-search-r1-grpo-qwen2.5-3b-2025-09-19/actor/global_step_180'
#export BASE_MODEL=/home/ssd3/wanfan01/pse_v2/Search-R1-main/verl_checkpoints/nq_hotpotqa_train-search-r1-grpo-2025-11-20/actor/global_step_160
export BASE_MODEL=/home/ssd3/jchluo/llm_models/Qwen2.5-3B-Instruct

export EXPERIMENT_NAME=searchr1_groupsize_qwen_3b

export WANDB_MODE=offline      # bash/zsh
/home/ssd3/chengmingquan/Search-R1-main/data/nq_search

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/wf_train.parquet \
    data.val_files="[$TEST_DATA_DIR/nq_100.parquet,$TEST_DATA_DIR/bamboogle.parquet]" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console','wandb'] \
    +trainer.validation_data_dir=results_val/${EXPERIMENT_NAME} \
    +trainer.rollout_data_dir=results_rollout/${EXPERIMENT_NAME}/ \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8121/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

