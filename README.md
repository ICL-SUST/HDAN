# HDAN: Hierarchical Dense Attention Network

This repository contains the source code for the paper: **"Neuro-developmental assessment in preterm infants via deep learning-based brain MRI segmentation"**.

## 1. Requirements
* Python 3.8+
* PyTorch >= 1.8.0
* Libraries: `pip install -r requirements.txt`

## 2. Dataset
The model is trained on the iSeg-2019 dataset. 
1. Download the dataset from the official challenge website.
2. Run `python data/preprocess.py` to prepare the data.
3. Update the data path in `config.py`.

## 3. Usage
### Training
To train the model from scratch:
```bash
python train.py --gpu 0