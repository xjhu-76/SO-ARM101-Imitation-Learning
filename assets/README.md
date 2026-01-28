# Assets Directory

This directory contains images and media for documentation.

## Required Files

- `demo.gif` - Main demonstration GIF for README header
- `task_setup_above.png` – Top-down view of the task setup
- `task_setup_side.png` – Side view of the task setup


## Creating demo.gif from video

```bash
# Using ffmpeg to convert video to GIF
ffmpeg -i videos/inference_demo.mp4 -vf "fps=10,scale=480:-1:flags=lanczos" -c:v gif demo.gif

# Or use a specific time range
ffmpeg -i videos/inference_demo.mp4 -ss 00:00:05 -t 00:00:10 -vf "fps=10,scale=480:-1:flags=lanczos" demo.gif
```
