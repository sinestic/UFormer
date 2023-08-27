# Introduction
This project is an unofficial implementation of the article [UFormer: A General U-Shaped Transformer for Image Restoration](https://arxiv.org/abs/2106.03106).
In this implementation I use same train methods, metrics and dataset explained in the article for **Deblurring task** and uses the PyTorch trasformers to build the main block of the architecture.<br>

# Prerequisites
To install all prerequisites is needed python >= 3.8 and launch the following command, preferring a virtual environment <br> ``python3 -m pip install -r requirements.txt``. <br>

# Dataset
The dataset used to train the model is the GoPRO and it can be downloaded from [here](https://drive.google.com/drive/folders/1Zpi7nKMZb_30fZE8BH_CVsCqNQKgYn38?usp=drive_link).
Put the dataset under the `dataset` folder.

# Training
To train the model, start the training by <br>
``python3 train.py``<br>

# Model weight
A pretrained model weights based on the max value of PSNR (Peak Signal to Noise Ratio) during training, are available to download from [here] (https://drive.google.com/file/d/1SYvNHHKtOB6cUZwPL8u-cCttht4Fasvv/view?usp=drive_link)

