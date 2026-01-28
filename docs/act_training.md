# ACT (Action Chunking with Transformers) Training Documentation

## Overview

This document details the training process for the ACT policy on the SO-ARM101 robot arm for a pick-and-place manipulation task.

## Model Architecture

ACT combines a visual encoder with a transformer-based action decoder:

```
Input Images (2 cameras) → ResNet18 Backbone → Transformer Encoder → VAE → Transformer Decoder → Action Chunks
```

### Architecture Details

| Component | Configuration |
|-----------|---------------|
| Vision Backbone | ResNet18 (pretrained on ImageNet) |
| Transformer Encoder | 4 layers, 512 dim, 8 heads |
| Transformer Decoder | 1 layer, 512 dim, 8 heads |
| VAE Encoder | 4 layers, latent dim 32 |
| Feed-forward Dim | 3200 |
| Dropout | 0.1 |
| Action Chunk Size | 100 steps |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| ResNet18 × 2 | ~23.4M |
| Transformer Encoder | ~13.4M |
| Transformer Decoder | ~3.3M |
| VAE Encoder | ~13.4M |
| Other (embeddings, projections) | ~2-3M |
| **Total** | **~55-60M** |

## Training Configuration

### Dataset

- **Task**: Pick red cube and place in gray bowl
- **Episodes**: 60 total
  - 30 episodes with side-view external camera
  - 30 episodes with top-view external camera
- **Cameras**: 2 (front environment + handeye wrist)
- **Resolution**: 640×480 @ 30fps

### Hyperparameters

```python
training_config = {
    "steps": 100000,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
    "seed": 1000,
}
```

### Hardware

- **GPU**: NVIDIA GeForce RTX 4090 (24GB)
- **Platform**: Ubuntu 22.04 with CUDA 12.4
- **Training Time**: ~3 hours

## Training Progress

### Loss Curve

| Step | Loss | Gradient Norm | Learning Rate |
|------|------|---------------|---------------|
| 11K | 0.168 | 16.703 | 1.0e-05 |
| 13K | 0.147 | 14.849 | 1.0e-05 |
| 15K | 0.136 | 14.227 | 1.0e-05 |
| 17K | 0.121 | 12.632 | 1.0e-05 |
| 20K | 0.105 | 11.255 | 1.0e-05 |
| 22K | 0.099 | 10.590 | 1.0e-05 |
| 25K | 0.089 | 9.641 | 1.0e-05 |
| 27K | 0.087 | 9.884 | 1.0e-05 |
| 30K | 0.081 | 9.091 | 1.0e-05 |
| 35K | 0.076 | 8.380 | 1.0e-05 |
| 40K | 0.071 | 7.972 | 1.0e-05 |

### Observations

1. **Smooth convergence**: Loss decreased steadily from 0.168 to 0.071
2. **Gradient stability**: Gradient norm decreased from ~17 to ~8, indicating stable training
3. **No learning rate scheduling**: Constant LR of 1e-5 throughout training

## Evaluation Results

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| Episode Success Rate | 7/7 (100%) |
| Grasp Success Rate | 8/36 (22.2%) |
| Total Grasping Attempts | 36 |
| Human Interventions | 6 times |
| Intervention Frequency | 0.86 per episode |

### Qualitative Observations

1. **Positioning accuracy**: Generally accurate in locating the target object
2. **Gripper control issue**: Insufficient gripper opening amplitude
3. **Dependency on intervention**: Required frequent human assistance to complete episodes
4. **Edge case handling**: Poor performance when object is at boundary of visual field

## Failure Analysis

### Primary Failure Mode

The most common failure was **insufficient gripper opening**, causing the gripper to miss the cube's bottom edge during grasping.

### Root Cause Hypothesis

The handeye (wrist-mounted) camera's field of view does not fully capture the gripper, leading to:
- Incomplete perception of gripper state
- Model cannot accurately predict required opening amplitude
- Conservative gripper actions learned from demonstrations

### Recommendations

1. Adjust handeye camera mounting angle
2. Include more wide-gripper demonstrations in training data
3. Consider adding gripper state as explicit input feature

## Reproduction

### Training Command

```bash
python -m lerobot.scripts.train \
    --dataset.repo_id=xjhu-76/so101_pick_place \
    --policy.type=act \
    --output_dir=outputs/train/act_so101 \
    --job_name=act_so101_pick_place \
    --steps=100000 \
    --batch_size=8 \
    --policy.device=cuda \
    --wandb.enable=false
```

### Inference Command

```bash
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=COM24 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --policy.path=outputs/train/act_so101/checkpoints/100000/pretrained_model
```

## Files

- Model checkpoint: `outputs/train/act_so101/checkpoints/100000/pretrained_model`
- Training config: `configs/act_config.yaml`
- Training logs: `results/act_training_log.txt`
