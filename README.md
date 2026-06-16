# Eye and Gaze Monitoring

## Setup

```bash
conda create -n gaze python=3.10
conda activate gaze

pip install -r requirements.txt
```

## GPU Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Required Model

Place:

L2CSNet_gaze360.pkl

inside:

L2CSNet/models/