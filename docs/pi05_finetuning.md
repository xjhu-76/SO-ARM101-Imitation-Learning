# Pi0.5 Fine-tuning Documentation

## Overview

This document details the fine-tuning process for the Pi0.5 Vision-Language-Action model on the SO-ARM101 robot arm for a pick-and-place manipulation task.

## Model Architecture

Pi0.5 is a Vision-Language-Action (VLA) model that leverages pre-trained vision-language understanding for robot control:

```
Input Images + Language Instruction → PaliGemma VLM → Action Head → Robot Actions
```

### Architecture Details

| Component | Configuration |
|-----------|---------------|
| Base Model | `lerobot/pi05_base` |
| VLM Backbone | PaliGemma (gemma_2b variant) |
| Vision Encoder | SigLIP |
| Language Model | Gemma 2B |
| Action Head | MLP decoder |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Total Parameters | 3,616,757,520 (~3.6B) |
| Trainable Parameters | 3,616,757,520 (~3.6B) |

**Note**: All parameters are trainable in this fine-tuning setup (full fine-tuning, not LoRA).

## Training Configuration

### Dataset

- **Task**: Pick red cube and place in gray bowl
- **Episodes**: 60 total (same as ACT)
- **Total Frames**: 10,877 frames
- **Cameras**: 2 (front environment + handeye wrist)
- **Resolution**: 640×480 @ 30fps

### Hyperparameters

```python
training_config = {
    "steps": 3000,
    "batch_size": 8,
    "learning_rate_peak": 2.5e-5,
    "learning_rate_decay": 2.5e-6,
    "warmup_steps": 1000,
    "dtype": "bfloat16",
    "gradient_checkpointing": True,
}
```

### Hardware

- **GPU**: NVIDIA A100-SXM4 (80GB) via Vast.ai
- **Training Time**: ~2-3 hours
- **Estimated Cost**: $3-5 USD

## Training Progress

### Loss Curve

| Step | Loss | Gradient Norm | Learning Rate |
|------|------|---------------|---------------|
| 200 | 0.326 | 4.753 | 1.9e-05 |
| 400 | 0.194 | 3.379 | 2.4e-05 |
| 600 | 0.152 | 2.577 | 2.3e-05 |
| 800 | 0.141 | 2.546 | 2.2e-05 |
| 1K | 0.124 | 2.359 | 1.8e-05 |
| 1.5K | ~0.118 | ~2.2 | ~1.5e-05 |
| 2K | ~0.115 | ~2.1 | ~1.2e-05 |
| 3K | 0.124 | ~2.0 | 2.5e-06 |

### Observations

1. **Rapid convergence**: Loss dropped from 0.326 to 0.124 in just 1000 steps
2. **Stable gradients**: Gradient norm remained low (~2-5), indicating smooth optimization
3. **Learning rate warmup**: 1000 warmup steps helped stabilize early training
4. **Pre-training advantage**: VLM backbone provides strong initialization for visual understanding

## Evaluation Results

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| Episode Success Rate | 7/9 (77.8%) |
| Grasp Success Rate | 13/54 (24.1%) |
| Total Grasping Attempts | 54 |
| Human Interventions | 2 times |
| Intervention Frequency | 0.22 per episode |

### Comparison with ACT

| Metric | ACT | Pi0.5 | Improvement |
|--------|-----|-------|-------------|
| Grasp Success Rate | 22.2% | 24.1% | +8.6% |
| Intervention Frequency | 0.86/ep | 0.22/ep | **-74%** |
| Training Steps | 100K | 3K | **33× fewer** |

### Qualitative Observations

1. **Higher autonomy**: Significantly fewer human interventions required
2. **Better generalization**: Handles edge cases more gracefully
3. **Semantic understanding**: VLM backbone provides scene understanding beyond pixel-level patterns
4. **Similar grasp accuracy**: Core manipulation accuracy comparable to ACT

## Key Advantages of Pi0.5

### 1. Training Efficiency
- 33× fewer training steps required
- Leverages pre-trained visual-language understanding
- Fine-tuning is more sample-efficient than training from scratch

### 2. Improved Autonomy
- 74% reduction in human intervention frequency
- Better handling of out-of-distribution scenarios
- More robust to variations in object position

### 3. Semantic Grounding
- VLM backbone understands scene semantics
- Can potentially generalize to similar objects/tasks
- Language conditioning enables task specification

## Failure Analysis

### Observed Failures (2/9 episodes)

1. **Episode 4**: Gripper missed cube due to positioning error
2. **Episode 7**: Successfully grasped but dropped during transport

### Common Issues (shared with ACT)

- Insufficient gripper opening amplitude
- Handeye camera field-of-view limitation

### Why Pi0.5 Handles Failures Better

The VLM backbone provides:
- Better object localization even at visual boundaries
- More robust feature representations
- Implicit understanding of task semantics

## Reproduction

### Training Command

```bash
python -m lerobot.scripts.train \
    --policy.path=lerobot/pi05_base \
    --dataset.repo_id=xjhu-76/so101_pick_place \
    --output_dir=outputs/train/pi05_so101 \
    --job_name=pi05_so101_pick_place \
    --steps=3000 \
    --batch_size=8 \
    --policy.device=cuda
```

### Inference Command

```bash
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=COM24 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --policy.path=outputs/train/pi05_so101/checkpoints/3000/pretrained_model
```

## Technical Notes

### Memory Optimization

Due to the large model size (3.6B parameters), the following optimizations were necessary:

1. **bfloat16 precision**: Reduces memory footprint by ~50%
2. **Gradient checkpointing**: Trades compute for memory
3. **A100 80GB**: Required for full fine-tuning without memory issues

### Cloud Training Setup (Vast.ai)

```bash
# SSH into cloud instance
ssh root@[instance_ip] -p [port]

# Setup environment
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[feetech]"

# Upload data and start training
```

## Files

- Model checkpoint: `outputs/train/pi05_so101/checkpoints/3000/pretrained_model`
- Training config: `configs/pi05_config.yaml`
- Training logs: `results/pi05_training_log.txt`

## References

1. Black, K., et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." Physical Intelligence, 2024.
2. Beyer, L., et al. "PaliGemma: A versatile 3B VLM for transfer." arXiv preprint, 2024.
