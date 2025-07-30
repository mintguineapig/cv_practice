# README

## Overview
이 저장소는 CIFAR-10, CIFAR-100, TinyImageNet 데이터셋에서 여러 Vision 모델을 훈련하고 비교한 실험 모음입니다.  
각 실험은 **weak** 또는 **default** augmentation 설정을 적용했고, **Adam** 옵티마이저(일부 실험은 **SGD**)를 사용했습니다.  
실험별 `exp-name` 규칙은 다음과 같습니다:

{데이터셋코드}_{모델명}[_입력크기]_augmentation_Optimizer

- **데이터셋코드**: `10`=CIFAR-10, `100`=CIFAR-100, `200`=TinyImageNet  
- **입력크기**: 기본 요구 크기(224×224) 사용 시 생략, 32×32로 강제 조정 시 `32x32` 표기  
- **augmentation**: `weak` 또는 `default` (ResNet-18만 `strong`도 실행)  
- **Optimizer**: `Adam` 또는 `SGD`

---

## Installation
```bash
# 시스템 요구
Python 3.8  
PyTorch ≥1.12  
CUDA 12.8  

# 필수 패키지
pip install \
  wandb==0.20.0 \
  pydantic==1.10.2


## Docker

docker run --gpus all -it -h cv_practice_gpu \
  -p 1290:1290 \
  --ipc=host \
  --name cv_practice_gpu \
  -v /m2:/projects \
  nvcr.io/nvidia/pytorch:22.12-py3 bash


# Usage
--dataname {CIFAR10|CIFAR100|TinyImagenet}
--num-classes {10|100|200}
--model-name <timm 모델 이름 또는 resnet18>
--opt-name {Adam|SGD}
--aug-name {weak|default|strong}
--batch-size <정수>
--lr <학습률>                # (옵션) 기본값 사용 시 생략 가능
--use_scheduler
--epochs <에포크 수>
--img-size <정수>           # 32×32 강제 조정 시 지정
--exp-name <실험 이름>

##Examples
### ResNet-18 on CIFAR-10 (default augment)
python main.py \
  --dataname CIFAR10 \
  --num-classes 10 \
  --model-name resnet18 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 64 \
  --lr 0.1 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_resnet18_default_Adam

### EfficientNet-B0 on CIFAR-10 (weak augment)
./run_timm.sh efficientnet_b0 \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 64 \
  --lr 0.1 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_efficientnet_b0_weak_Adam


### ViT-Small (32×32) on CIFAR-10
./run_timm.sh vit_small_patch32_32 \
  --dataname CIFAR10 \
  --num-classes 10 \
  --model-name vit_small_patch32_32 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --img-size 32 \
  --exp-name 10_ViTsmall32x32_default_Adam

### ConvNeXt-Base on CIFAR-10 (weak augment)
python main.py \
  --model-name convnext_base \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 128 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_ConvNeXt_weak_Adam

### naflexvit_base on CIFAR-10 (weak augment)
python main.py \
  --model-name naflexvit_base_patch16_gap.e300_s576_in1k \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name weak \
  --batch-size 64 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_naflexvit_weak_Adam

### naflexvit_base on CIFAR-10 (default augment)
python main.py \
  --model-name naflexvit_base_patch16_gap.e300_s576_in1k \
  --dataname CIFAR10 \
  --num-classes 10 \
  --opt-name Adam \
  --aug-name default \
  --batch-size 64 \
  --lr 0.001 \
  --use_scheduler \
  --epochs 50 \
  --exp-name 10_naflexvit_default_Adam

### ResNet-18 on CIFAR-100 (weak augment)



