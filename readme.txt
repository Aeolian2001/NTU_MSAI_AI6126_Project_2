This project is based on Real-ESRGAN repo: https://github.com/xinntao/Real-ESRGAN

1 Information
CodaLab username: 	
Matriculation number: 	

2 Files Submitted
`infer.py`: Infer HQ pics on the test dataset
`realesrgan_dataset.py`:   Degradation functions
`train_SRResNet_x4_FFHQ_900k.yml` :  Hyper-parameters of model
`net_g_855000.pth`:  Best model weights
`test_real.png`:  Results of test dataset
`best_psnr_score.png`:  Screen shot of best PSNR on Codalab
`train_train_SRResNet_x4_FFHQ_900k.log`:  Train log

3 Third-party Libraries
Python 3.11
PyTorch: 2.5.1+cu121
TorchVision: 0.20.1+cu121
Others see Test Step 2

4 Test
----------------------------------------------------Test----------------------------------------------------
step1. Clone repo
###
git clone https://github.com/xinntao/Real-ESRGAN.git
###

step2. Install dependent packages
###
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
###

Step3. Put the files into their corresponding folders
###
1) Put `train_SRResNet_x4_FFHQ_900k.yml` under the `options` folder.
2) Replace `realesrgan_dataset.py` under the `realesrgan/data` folder.
3) Put `infer.py` and `evaluate.py` under the main folder
4) Put train, val and test dataset under `dataset` folder
Note: `realesrgan_dataset.py` submitted is the same as the `ffhqsub_dataset.py` you given.
###
Real-ESRGAN/  # main folder
├── infer.py
├── evaluate.py
├── dataset/
│     ├── train/
│     ├── val/
│     └── test/
├── options/
│   └── train_SRResNet_x4_FFHQ_900k.yml
└── realesrgan/
│     └── data/
│             └── realesrgan_dataset.py
...
###

Step4: Infer
Infer HQ pics
###
python infer.py
###
Results are put in `results`


Note: 
###
If you have following problem:
'''from torchvision.transforms.functional import rgb_to_grayscale'''

Use following to solve:  
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py

# replace '/usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py' to your path
###
