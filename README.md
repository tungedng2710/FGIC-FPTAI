# FGIC-FPTAI
Fine-Grained Image Classification using Deep Learning
Reference: https://github.com/ZF4444/MMAL-Net (Official implementation of MMAL-Net)
## Installation
Clone this repository and move to project folder
```bat
git clone https://github.com/tungedng2710/FGIC-FPTAI.git
cd FGIC-FPTAI
```
Install the requirements via pip
```bat
pip install -r requirements.txt
```
Due to the limitation size of GitHub upload, I just put the annotation files under ```data``` folder and the images folder is still empty. You need to take the images data at the following link https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/


## Training
In your terminal, run 
```bat
python main.py
```
For more configurations, pay attention to ```config.py``` and the comment inside.

## Pretrained model
[ZF4444](https://github.com/ZF4444) provide the checkpoint model trained by ourselves, you can download if from [here](https://drive.google.com/file/d/1-LD1Jz6Dh-P6Ibtl17scfrTFQTrW4Zy3/view) for FGVC-Aircraft. You should put it under **weights/pretrained** folder.