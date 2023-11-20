<div align="center">

# Fast Particle-based Anomaly Detection Algorithm with Variational Autoencoder

<figure>
    <img src="https://fastmachinelearning.org/hls4ml/_images/hls4ml_logo.png" width="250"/>
</figure>


[Machine Learning and Physical Science at Neurips 2023 Presentation](https://nips.cc/virtual/2023/76170)
    
[arXiv paper]()

[Author Contact](mailto:liuryan30@berkeley.edu)

</div>

Welcome to code repository for Fast Particle-based Anomaly Detection Algorithm with Variational Autoencoder

## Installation 
```
git clone https://github.com/ryanliu30/FastAnomalyDetection.git --recurse-submodules
cd FastAnomalyDetection
conda create -n FastAnomalyDetection python=3.11
conda activate FastAnomalyDetection
pip install -r requirements.txt
pip install -e .
pip install -e weaver-core/
```
To download the dataset, run:
```
particle_transformer/get_datasets.py JetClass -d data
```
This will put the dataset under the default folder `./data`. If this is not desired, please change `-d` to the desired directory and update `fastAD/configs/loader_config.yaml` accordingly.
## Usage
To begin with, run the following command:
```
python train.py --cfg experiments/SetVAE_deepset.yaml
```
This will train a deepset SetVAE model. To train with ClipVAE, run:
```
python train.py --cfg experiments/ClipVAE_deepset.yaml
```
Under the `experiments` directory, there are two other configuration files `SetVAE_transformer.yaml` and `ClipVAE_transformer.yaml` that can be used to train transformer based models.

To evaluate the trained models, run
```
python eval.py --ckpt PATH_TO_CKPT --model [DeepsetSetVAE|DeepsetClipVAE|TransformerSetVAE|TransformerClipVAE] --save_name FIG_NAME
```
We supplied four model checkpoints from our training that can be used out of the box. Note that by evaluating a checkpoint from `SetVAE` model while passing in `--model ClipVAE`, the will give the performance of the same model but with KL-divergence as anomaly score instead of reconstruction loss.
## Citation
If you use this work in your research, please cite:
```
BibTex here
```
