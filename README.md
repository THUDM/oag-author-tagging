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

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1Mjd8KH5hutHu-R6oyBNJSw) (with password bqdt). Put the _raw_data_ directory into project directory.

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

```

Note that part of the code is adapted from https://github.com/geekinglcq/aca (top solutions from https://www.biendata.xyz/competition/scholar/)
