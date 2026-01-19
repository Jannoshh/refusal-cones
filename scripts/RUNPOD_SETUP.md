# RunPod Setup Guide for Refusal Cones

This guide walks you through setting up a RunPod instance for training unified affine RDO models.

## Quick Start (2 minutes)

1. **Create RunPod Instance**
   - Go to https://www.runpod.io/console/pods
   - Select a GPU (RTX 4090 or A40 recommended)
   - Use template: `PyTorch 2.1` or `RunPod PyTorch`
   - Container Disk: 50GB minimum
   - Launch pod

2. **Setup GitHub Authentication (for private repo)**
   ```bash
   # In the RunPod terminal:
   # Option 1: Use GitHub CLI (recommended)
   gh auth login
   # Follow the prompts to authenticate with your GitHub account

   # Option 2: Use SSH key
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Add the key to your GitHub account: https://github.com/settings/keys
   ```

3. **Run Setup Script**
   ```bash
   # Download and run the setup script
   curl -fsSL https://raw.githubusercontent.com/Jannoshh/refusal-cones/main/scripts/runpod_setup.sh | bash

   # Or clone the repo first (if you have SSH set up):
   mkdir -p ~/workspace && cd ~/workspace
   git clone git@github.com:Jannoshh/refusal-cones.git
   cd refusal-cones/scripts
   ./runpod_setup.sh
   ```

4. **Activate Environment**
   ```bash
   exec zsh  # Restart shell
   rdc       # Activate refusal-cones environment
   ```

5. **Start Training**
   ```bash
   python examples/example_train_with_eval.py
   ```

## What the Setup Script Does

The `runpod_setup.sh` script automates:

### 1. System Dependencies
- Essential utilities: `htop`, `nvtop`, `btop`, `jq`, etc.
- Build tools: `gcc`, `make`, etc.
- Shell: `zsh`, `tmux`, `git`
- GitHub CLI (`gh`) for authentication

### 2. Python Environment
- Installs `uv` (fast Python package manager)
- Python 3.11
- Virtual environment at `~/workspace/refusal-cones/.venv`

### 3. GitHub Authentication
- Sets up `gh` CLI for private repo access
- Supports both HTTPS (via gh) and SSH authentication
- Validates authentication before cloning

### 4. ML Dependencies
- PyTorch 2.1+ with CUDA 12.1
- Transformers, PEFT, Datasets, Accelerate
- Scientific computing: NumPy, SciPy, scikit-learn
- Utilities: `simple-gpu-scheduler`, `ipykernel`

### 5. Dotfiles
- Your personal dotfiles from https://github.com/Jannoshh/dotfiles
- Zsh with oh-my-zsh, powerlevel10k theme
- Tmux configuration
- Vim configuration

### 6. Refusal Cones Project
- Clones private repo to `~/workspace/refusal-cones`
- Sets up Jupyter kernel
- Creates convenience aliases

## Directory Structure

After setup:
```
~/workspace/
├── refusal-cones/          # Main project
│   ├── .venv/              # Virtual environment
│   ├── src/                # Source code
│   ├── examples/           # Example scripts
│   └── ...
├── dotfiles/               # Your dotfiles
└── start_refusal_cones.sh  # Convenience script
```

## Usage

### Activating the Environment

Option 1: Use the alias (recommended)
```bash
rdc  # Short for "refusal direction cones"
```

Option 2: Manual activation
```bash
cd ~/workspace/refusal-cones
source .venv/bin/activate
```

### Training Commands

**Quick test (2-3 minutes)**
```bash
python examples/example_baseline_fitting.py
```

**Full training with validation (~15-30 min on RTX 4090)**
```bash
python examples/example_train_with_eval.py
```

**Custom training**
```python
from src.training import train_with_validation

model, results = train_with_validation(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    n_train_harmful=1000,
    n_train_harmless=1000,
    num_epochs=5,
    use_baseline=True,
    projection_alpha=1.0,
    addition_alpha=1.0
)
```

### Monitoring

**GPU usage**
```bash
nvidia-smi           # Static view
nvtop                # Interactive (like htop for GPU)
watch -n 1 nvidia-smi  # Auto-refresh every 1s
```

**Training progress**
```bash
# In another terminal/tmux pane
cd ~/workspace/refusal-cones
source .venv/bin/activate
tensorboard --logdir unified_rdo_demo
```

## GPU Recommendations

| GPU | VRAM | Training Time* | Cost/hr | Total Cost* |
|-----|------|---------------|---------|-------------|
| RTX 4090 | 24GB | 15-20 min | $0.69 | $0.17-0.23 |
| RTX 3090 | 24GB | 20-25 min | $0.44 | $0.15-0.18 |
| A40 | 48GB | 18-22 min | $0.79 | $0.24-0.29 |
| A100 40GB | 40GB | 12-15 min | $1.89 | $0.38-0.47 |

\* For Qwen2.5-0.5B-Instruct, 1000 train examples, 5 epochs

**Recommendation**: RTX 4090 offers best price/performance for this model size.

## Troubleshooting

### Setup script fails

**Problem**: Script exits with errors

**Solution**:
```bash
# Run sections manually
cd ~/workspace
git clone https://github.com/Jannoshh/refusal-cones.git
cd refusal-cones

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install torch transformers peft datasets accelerate
```

### Out of memory during training

**Problem**: CUDA OOM error

**Solutions**:
1. Reduce batch size in training args
2. Enable gradient checkpointing (already enabled by default)
3. Use smaller model or fewer training examples
4. Use GPU with more VRAM

### Can't find CUDA

**Problem**: PyTorch can't find GPU

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA support
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Training is slow

**Problem**: Training takes much longer than expected

**Checks**:
1. Verify GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```

2. Check GPU utilization: `nvidia-smi` (should be 70-100%)

3. Reduce dataset size for testing:
   ```python
   train_with_validation(
       n_train_harmful=100,  # Start small
       n_train_harmless=100,
       num_epochs=2
   )
   ```

## Cost Optimization

### Tips for Minimizing Costs

1. **Use smaller model for prototyping**
   - Qwen2.5-0.5B-Instruct is fast and cheap
   - Scale to larger models (Gemma-2-2B, Llama-3.1-8B) after validation

2. **Save checkpoints frequently**
   - Training saves every N steps (configurable)
   - Can resume if interrupted

3. **Stop pod when not training**
   - RunPod charges by the minute
   - Data persists in storage volume
   - Can restart later with same environment

4. **Use spot instances**
   - ~50% cheaper than on-demand
   - May be interrupted (rare for short jobs)
   - Good for experiments <1 hour

### Example Workflow (Total cost: ~$0.50)

1. Prototype with small data (100 examples) - 5 min - $0.06
2. Validate approach with medium data (1K examples) - 20 min - $0.23
3. Final run with full data (10K examples) - 90 min - $1.04

Total development cost: ~$1.33 (vs $20+ on M2 Pro in time)

## Advanced: Multi-GPU Training

If you have multiple GPUs:

```python
# In your training script
from accelerate import Accelerator

accelerator = Accelerator()

# Accelerate will automatically:
# - Detect all GPUs
# - Distribute batches
# - Synchronize gradients

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

Or use `torchrun`:
```bash
torchrun --nproc_per_node=4 examples/example_train_with_eval.py
```

## SSH Access (Optional)

For better experience, connect via SSH:

1. In RunPod pod settings, note the SSH command
2. In your local terminal:
   ```bash
   ssh -p <PORT> root@<IP>
   ```

3. (Optional) Setup SSH key for passwordless access

## VSCode Remote (Optional)

Connect VSCode to RunPod:

1. Install "Remote - SSH" extension
2. Add SSH host from RunPod
3. Connect to pod
4. Open folder: `/root/workspace/refusal-cones`
5. Select Python interpreter: `.venv/bin/python`

Now you have full IDE with remote execution!

## Further Reading

- [RunPod Documentation](https://docs.runpod.io/)
- [Project README](../README.md)
- [Training Examples](../examples/)
- [CLAUDE.md](../CLAUDE.md) - Project overview and guidelines

## Questions?

- Check the main [README.md](../README.md)
- See examples in `examples/`
- Review code documentation in `src/training/`

Happy training! 🚀
