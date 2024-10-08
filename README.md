# Scholar Interest Prediction

## Prerequisites

- Linux
- Python 3.7
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/oag-author-tagging.git
cd oag-author-tagging
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1Mjd8KH5hutHu-R6oyBNJSw) (with password bqdt) or [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/entity-tagging/oag-author-tagging/raw_data.zip). Put the _raw_data_ directory into project directory.

## How to run

```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
export PYTHONPATH="`pwd`:$PYTHONPATH"

# processing
python aca/code/create_file.py
python aca/code/create_citenet.py
python aca/code/expand_paper.py
python aca/code/expand_author_press.py

# LSI model
python aca/code/task2_main.py --method lsi

# aca method
python aca/code/task2_main.py --method aca

# sentence-bert
python ptm/pretrain_models_sim.py --method sbert
optional methods:["sbert", "simcse-roberta", "e5-large", "bert", "deberta-base", "deberta-v3-large", "gte-large", "bge-large", "e5-large-v2", "sentence-t5-xxl", "simcse-bert"]

# evaluation
python evaluate.py --method lsi
python evaluate.py --method aca
python evaluate.py --method sbert

```

## Results

|       | Hit@3 |
| ----- | ----- |
| LSI   | 24.37 |
| SBERT | 15.97 |
| ACA   | 30.32 |

Note that part of the code is adapted from https://github.com/geekinglcq/aca (top solutions from https://www.biendata.xyz/competition/scholar/)


## References
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@inproceedings{zhang2024oag,
  title={OAG-bench: a human-curated benchmark for academic graph mining},
  author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6214--6225},
  year={2024}
}
```
