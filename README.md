# HCD-IRN (Invertible Image Rescaling)
Pytorch implementation for "Downscaled Representation Matters: Improving Image Rescaling with Collaborative Downscaled Images".

## Requirements
- PyTorch >= 1.0
- Python >= 3.6
- NVIDIA GPU + CUDA

## Inference for image rescaling
```
python my_train_patch.py -opt /options/test/test_IRN_x2.yml
```
## Acknowledgement
The code is based on [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling), and the pretrained model comes from [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling).
