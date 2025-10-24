import os
import re
import json
import time
import datetime
import importlib
from transformers import AutoConfig

# from flashrag.dataset.dataset import Dataset

from verl.utils.reward_score.pse.flashrag.dataset.dataset import Dataset

def get_dataset(config):
    """Load dataset from config."""

    dataset_path = config["dataset_path"]
    all_split = config["split"]

    split_dict = {split: None for split in all_split}

    for split in all_split:
        split_path = os.path.join(dataset_path, f"{split}.jsonl")
        set_logger(config, info=[f"{split_path} file exists!", ""])
        if not os.path.exists(split_path):
            set_logger(config, info=[f"{split_path} file not exists!", ""])
            continue

        split_dict[split] = Dataset(
            config,
            split_path,
            sample_num=config["test_sample_num"],
            random_sample=config["random_sample"]
            )

    return split_dict


def get_generator(config, **params):
    """Automatically select generator class based on config."""
    if config["framework"] == "vllm":
        print("start framework vllm!")
        return getattr(importlib.import_module("flashrag.generator"), "VLLMGenerator")(config, **params)
    elif config["framework"] == "fschat":
        return getattr(importlib.import_module("flashrag.generator"), "FastChatGenerator")(config, **params)
    elif config["framework"] == "hf":
        model_config = AutoConfig.from_pretrained(config["generator_model_path"])
        arch = model_config.architectures[0]
        if "t5" in arch.lower() or "bart" in arch.lower():
            return getattr(importlib.import_module("flashrag.generator"), "EncoderDecoderGenerator")(config, **params)
        else:
            return getattr(importlib.import_module("flashrag.generator"), "HFCausalLMGenerator")(config, **params)
    elif config["framework"] == "openai":
        return getattr(importlib.import_module("flashrag.generator"), "OpenaiGenerator")(config, **params)
    else:
        raise NotImplementedError


def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("flashrag.retriever"), "BM25Retriever")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)


def get_reranker(config):
    model_path = config["rerank_model_path"]
    # get model config
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if "forsequenceclassification" in arch.lower():
        return getattr(importlib.import_module("flashrag.retriever"), "CrossReranker")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "BiReranker")(config)


def get_judger(config):
    judger_name = config["judger_name"]
    if "skr" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "SKRJudger")(config)
    elif "adaptive" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "AdaptiveJudger")(config)
    else:
        assert False, "No implementation!"


def get_refiner(config, retriever=None, generator=None):
    DEFAULT_PATH_DICT = {
        "recomp_abstractive_nq": "fangyuan/nq_abstractive_compressor",
        "recomp:abstractive_tqa": "fangyuan/tqa_abstractive_compressor",
        "recomp:abstractive_hotpotqa": "fangyuan/hotpotqa_abstractive",
    }
    REFINER_MODULE = importlib.import_module("flashrag.refiner")

    refiner_name = config["refiner_name"]
    refiner_path = (
        config["refiner_model_path"]
        if config["refiner_model_path"] is not None
        else DEFAULT_PATH_DICT.get(refiner_name, None)
    )

    try:
        model_config = AutoConfig.from_pretrained(refiner_path)
        arch = model_config.architectures[0].lower()
        print(arch)
    except Exception as e:
        print("Warning", e)
        model_config, arch = "", ""

    if "recomp" in refiner_name or "bert" in arch:
        if model_config.model_type == "t5":
            refiner_class = "AbstractiveRecompRefiner"
        else:
            refiner_class = "ExtractiveRefiner"
    elif "lingua" in refiner_name:
        refiner_class = "LLMLinguaRefiner"
    elif "selective-context" in refiner_name or "sc" in refiner_name:
        refiner_class = "SelectiveContextRefiner"
    elif "kg-trace" in refiner_name:
        return getattr(REFINER_MODULE, "KGTraceRefiner")(config, retriever, generator)
    else:
        raise ValueError("No implementation!")

    return getattr(REFINER_MODULE, refiner_class)(config)


def hash_object(o) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    import hashlib
    import io
    import dill
    import base58

    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def set_logger(config, info : list):
    config["log_info"] += info
    print("\n".join(info))


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    CST = datetime.timezone(t_delta, 'CST')
    now = datetime.datetime.now(CST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def str2json(string):
    
    try:  # output format: {"Response": answer}
        json_data = eval(string)  
        return json_data
    
    except Exception as e:  # output format: ```json{"Response": answer}```
        json_matches = re.findall(r'```json(.*?)```', string, re.DOTALL)
        last_json_str = ""
        if json_matches:
            last_json_str = json_matches[-1].strip()
        else:
            json_matches = re.findall(r'({.*?})', string, re.DOTALL)
            if json_matches:
                last_json_str = json_matches[-1].strip()

        try:
            json_data = json.loads(last_json_str)
            return json_data
        except json.JSONDecodeError as e:
            return None

def run_pipeline(config, pipeline, data, pred_process_fun=None):

    rerank_name = ""
    if config['use_reranker']:
        rerank_name = "+" + config['rerank_model_name']

    set_logger(config,
               info=[
                   f"====== Current experiment configuration ======",
                   f"Dataset: {config['dataset_name']}",
                   f"Method: {config['method_name']}",
                   f"Retriever: {config['retrieval_method']}{rerank_name}",
                   f"Generator: {config['generator_model']}",
                   f"=============================================="
                ])

    start = time.time()

    # try:
    if 1:
        result = pipeline.run(data, pred_process_fun=pred_process_fun)
        set_logger(config, info=["", f"[{print_now(1)}] process state: SUCCESS"])
    # except Exception as e:
    #     print(e)
    #     set_logger(config, info=["", f"[{print_now(1)}] process state: ERROR"])
        
    end = time.time()

    set_logger(config, info=[f"[{print_now(1)}] time cost: {round(end - start, 4)} s"])

    # save log
    current_time = datetime.datetime.now()
    with open(f"{config['save_dir']}/{current_time.strftime('%Y_%m_%d_%H_%M')}.log", "w") as f:
        f.write("\n".join(config["log_info"]))
