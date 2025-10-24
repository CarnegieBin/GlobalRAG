<h1 align="center">🚀🚀🚀 <b>GlobalRAG</b>: Enhancing <b><i>Global</i></b> Reasoning in Multi-hop Question Answering via Reinforcement Learning</h1>



**GlobalRAG** is a reinforcement learning framework designed for **multi-hop question answering**. It decomposes complex questions into **sub-goals**, enabling coordinated retrieval and reasoning while **iteratively optimizing evidence utilization**.
To foster **global planning** and **reliable execution**, the framework introduces two complementary reward signals — a *planning quality reward* and a *sub-goal completion reward*. These jointly balance **process-oriented** and **outcome-oriented** objectives through **progressive weight annealing**.

Built upon [Search-R1](https://github.com/PeterGriffinJin/Search-R1), **GlobalRAG** extends its **multi-hop reasoning** capabilities with a **structured reinforcement signal** that explicitly models planning quality.

<p align="center">
          🤗 <a href="https://huggingface.co/datasets/Carnegie-Bin/GlobalRAG-data">Models</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2510.20548">Paper</a>&nbsp&nbsp ｜ 📊 <a href="https://huggingface.co/datasets/Carnegie-Bin/GlobalRAG-data">Datasets</a>&nbsp&nbsp 
</p>


## ⭐️&nbsp;Features

<table style="width:100%; table-layout:fixed;">
<tr>
<td style="vertical-align:top; width:50%; padding-right:20px;">

- 📝 **Propose a question-answering paradigm**  
    - Explicitly demonstrates **global planning**  
    - Enables the model to **learn to plan before retrieving information**
<br>
<br>
<br>

- 🧩 **Enhancing multi-hop question answering**  
    - Integrates **dense process supervision signals**  
    - Utilizes **reinforcement learning techniques** to improve reasoning**
<br>
<br>
<br>

- 📈 **Performance improvements**  
    - Achieves **+14.2 EM and F1 points** compared to current state-of-the-art methods  
    - Evaluated on multi-hop QA datasets: **2Wiki, HotpotQA, Musique, Bamboogle, Wikihop**


</td>
<td style="vertical-align:top; width:50%; text-align:left;">
<img src="public/description.png" style="width:50%; height:50%;">
</td>
</tr>
</table>


<p align="center">
  <img src="public/intro.png" alt="introduction" />
</p>




## 🔄&nbsp;Pipeline

<p align="center">
  <img src="public/pipeline.png" alt="pipeline" />
</p>



## 📰&nbsp;News


- [2025.11] We opensource GlobalRAG checkpoint.
- [2025.11] We opensource GlobalRAG train and test data.
- [2025.11] We opensource GlobalRAG code.


## ⚙️&nbsp;Installation

### GlobalRAG environment (fork from Search-R1)
```bash
conda create -n globalrag python=3.9
conda activate globalrag
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install swanlab
```

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```


## ▶️&nbsp;Quick start

Train a reasoning + search LLM  with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Process the dataset for training.
```bash
hf download Carnegie-Bin/GlobalRAG-data --local-dir Carnegie-Bin/GlobalRAG-data
```

(3) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) Run RL training (GRPO) on GlobalRAG.
```bash
conda activate globalrag
bash scripts/train_grpo_3B.sh
```

## 🤔&nbsp;Inference
(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Run inference.
```bash
conda activate globalrag
python scripts/eval_grpo_3B.sh
```

## 🗂️&nbsp;Use your own dataset

### QA data
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

### Corpora

It is recommended to make your corpus a jsonl file, where each line (a dictionary with "id" key and "contents" key) corresponds to one passage. You can refer to ```example/corpus.jsonl``` for an example.

The "id" key corresponds to the passage id, while the "contents" key corresponds to the passage content ('"' + title + '"\n' + text).
For example:
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**Index your corpora (optional).**
If you would like to use a local retriever as the search engine, you can index your own corpus by:
```
bash search_r1/search/build_index.sh
```
You can change ```retriever_name``` and ```retriever_model``` to your interested off-the-shelf retriever.

## 🔍&nbsp;Use your own search engine

Our codebase supports local sparse retriever (e.g., BM25), local dense retriever (both flat indexing with GPUs and ANN indexing with CPUs) and online search engine (e.g., Google, Bing, etc). More details can be found [here](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md).

The main philosophy is to launch a local or remote search engine server separately from the main RL training pipeline. 

The LLM can call the search engine by calling the search API (e.g., "http://127.0.0.1:8121/retrieve").

You can refer to ```search_r1/search/retriever_server.py``` for an example of launching a local retriever server.



## 🎓&nbsp;Case Study
Here we present a case study illustrating a complete trajectory generated by our model.

<img src="public/case.png" style="width:100%; height:auto;" />





## ❕&nbsp;Acknowledge

The concept of GlobalRAG is inspired by [Search-R1](https://github.com/PeterGriffinJin/Search-R1).
Its implementation is built upon [veRL](https://github.com/volcengine/verl) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). 
We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.


## Citations

```bibtex
@article{luo2025globalrag,
  title={GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning},
  author={Luo, Jinchang and Cheng, Mingquan and Wan, Fan and Li, Ni and Xia, Xiaoling and Tian, Shuangshuang and Bian, Tingcheng and Wang, Haiwei and Fu, Haohuan and Tao, Yan},
  journal={arXiv preprint arXiv:2510.20548},
  year={2025}
}
```

