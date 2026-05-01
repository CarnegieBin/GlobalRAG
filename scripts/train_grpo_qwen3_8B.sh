


export TEST_DATA_DIR="/ssd2/lini03/Search-R1-infer/test_model/data"
# export TEST_FILES="[${TEST_DATA_DIR}/hotpotqa_wf.parquet,${TEST_DATA_DIR}/2WikiMultihopQA_wf.parquet,${TEST_DATA_DIR}/bamboogle_wf.parquet,${TEST_DATA_DIR}/musique_wf.parquet,${TEST_DATA_DIR}/wikihop_wf.parquet]"
export SWANLAB_PROJECT="GlobalRAG-8A800"

export TEST_FILES="[${TEST_DATA_DIR}/bamboogle_wf.parquet,${TEST_DATA_DIR}/musique_wf.parquet,${TEST_DATA_DIR}/wikihop_wf.parquet]"


export SWANLAB_MODE=offline
export RAY_TMPDIR="/ssd1/tcbian/ray_tmp"
mkdir -p $RAY_TMPDIR
export BASE_MODEL="/ssd2/llm_models/Qwen3-8B"
export EXPERIMENT_NAME="Qwen3-8B-GlobalRAG"
export CHECKPOINT_DIR="/ssd1/tcbian/GlobalRAG"
# To resume training, set RESUME_STEP to the global_step number of the checkpoint to resume from,
# and set actor_rollout_ref.model.path to the corresponding checkpoint path, e.g.:
# export RESUME_STEP=100
# export BASE_MODEL="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}/actor/global_step_${RESUME_STEP}"
export embedding_model_path="/ssd2/llm_models/e5-base-v2"

#export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="/home/work/tcbian/GlobalRAG/data/train.parquet" \
    data.val_files="${TEST_FILES}" \
    +data.train_data_num=null \
    +data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    +data.max_start_length=2048 \
    +data.max_obs_length=600 \
    +data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +algorithm.no_think_rl=false \
    +actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    +actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[console,swanlab] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=$SWANLAB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXPERIMENT_NAME} \
    trainer.resume_mode=disable \
    +trainer.balance_batch=true \
    +trainer.log_val_generations=0 \
    +max_turns=5 \
    +do_search=true \
    +n_val=1 \
    +e5_model_path=${embedding_model_path} \
    +retriever.url="http://127.0.0.1:8121/retrieve" \
    +retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
