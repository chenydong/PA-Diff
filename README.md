# PA-Diff
The code of paper "Learning A Physical-aware Diffusion Model Based on  Transformer for Underwater Image Enhancement"

#### Dependencies
- [PyTorch](https://pytorch.org/) 
- [einops](https://github.com/arogozhnikov/einops)
- opencv-python
#### Installation
You should install Pytorch first (This installation command may not be available, please find the appropriate command in [PyTorch](https://pytorch.org/) )
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Use requirements.txt to install other requirements
```
pip install -r requirements.txt
```

### Datasets
[LSUI](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement?tab=readme-ov-file#Training)｜[UIEB](https://li-chongyi.github.io/proj_benchmark.html)｜[U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-)｜[Realworld Images]()

Prepare dataset:
1. You should make sure your image size is 256*256
2. Your directory structure should look like this:
```
train:
DatasetName_train_16_256  (16 and 256 are set in config)
├── hr_256 (GT images)
└── sr_256 (input images)

test:
DatasetName_val_16_256 
├── hr_256
└── sr_256

```

### Test
1. Download our checkpoints from [Baidu Netdisk]()
2. Change "resume_state" in the config file to the path of your checkpoints
3. Change test "dataroot" in the config file to the path like `xxx/DatasetName_val_16_256`
4. Run :` python infer.py`

### Train
1. Change train "dataroot" in the config file to the path like `xxx/DatasetName_train_16_256`
2. Run :` python train.py`

### Citation
```
@article{zhao2024learning,
  title={Learning A Physical-aware Diffusion Model Based on Transformer for Underwater Image Enhancement},
  author={Zhao, Chen and Dong, Chenyu and Cai, Weiling},
  journal={arXiv preprint arXiv:2403.01497},
  year={2024}
}
```

