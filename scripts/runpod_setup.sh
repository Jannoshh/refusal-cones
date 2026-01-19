#!/bin/bash
set -euo pipefail

# =============================================================================
# RunPod Setup Script for Refusal Cones Project
# =============================================================================
# This script sets up a RunPod instance with:
# 1. System dependencies and utilities
# 2. Dotfiles (zsh, tmux, etc.)
# 3. Python environment with ML dependencies (PyTorch, transformers, PEFT)
# 4. Refusal-cones project repository
# =============================================================================

echo "========================================="
echo "RunPod Setup for Refusal Cones Project"
echo "========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 1. Setup system dependencies
# =============================================================================
log_info "Installing system dependencies..."

# Ensure sudo is available
if ! command -v sudo &> /dev/null; then
    log_warn "sudo not found, installing..."
    su -c 'apt-get update && apt-get install -y sudo'
fi

# Install essential utilities
sudo apt-get update -y
sudo apt-get install -y \
    less nano htop ncdu nvtop lsof rsync btop jq \
    zsh tmux git curl wget \
    build-essential

log_info "System dependencies installed ✓"

# =============================================================================
# 2. Setup Python environment with uv
# =============================================================================
log_info "Setting up Python environment with uv..."

# Install uv (fast Python package manager)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    source $HOME/.local/bin/env
fi

# Install Python 3.11
log_info "Installing Python 3.11..."
uv python install 3.11

log_info "Python environment setup complete ✓"

# =============================================================================
# 3. Setup GitHub authentication (for private repo)
# =============================================================================
log_info "Setting up GitHub authentication..."

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    log_info "Installing GitHub CLI..."
    sudo apt-get install -y gh
fi

# Check if already authenticated
if ! gh auth status &> /dev/null; then
    log_warn "GitHub authentication required for private repository access"
    log_info "Please authenticate with GitHub:"
    log_info "  Option 1 (Recommended): gh auth login"
    log_info "  Option 2: Set up SSH key: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    log_info ""
    log_info "After authentication, run this script again."

    # Try to authenticate
    gh auth login

    if ! gh auth status &> /dev/null; then
        log_error "GitHub authentication failed. Please set up authentication manually."
        exit 1
    fi
else
    log_info "GitHub authentication verified ✓"
fi

# =============================================================================
# 4. Clone and setup refusal-cones project
# =============================================================================
log_info "Setting up refusal-cones project..."

# Create workspace directory
mkdir -p ~/workspace
cd ~/workspace

# Clone refusal-cones repository
if [ -d "refusal-cones" ]; then
    log_warn "refusal-cones directory already exists, skipping clone"
    cd refusal-cones
else
    log_info "Cloning refusal-cones repository (private)..."

    # Try HTTPS with GitHub CLI first
    if gh auth status &> /dev/null; then
        gh repo clone Jannoshh/refusal-cones
    else
        # Fallback to SSH if available
        if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
            git clone git@github.com:Jannoshh/refusal-cones.git
        else
            log_error "No valid authentication method found. Please set up gh or SSH."
            exit 1
        fi
    fi

    cd refusal-cones
fi

# Create virtual environment
log_info "Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install ML dependencies
log_info "Installing ML dependencies (PyTorch, transformers, PEFT, etc.)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers peft datasets accelerate
uv pip install scikit-learn scipy numpy matplotlib jaxtyping
uv pip install ipykernel simple-gpu-scheduler

# Install ipykernel for Jupyter/VSCode
log_info "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=refusal-cones

log_info "Refusal-cones project setup complete ✓"

# =============================================================================
# 5. Setup dotfiles (zsh, tmux, etc.)
# =============================================================================
log_info "Setting up dotfiles..."

cd ~/workspace

# Clone dotfiles repository
if [ -d "dotfiles" ]; then
    log_warn "dotfiles directory already exists, skipping clone"
else
    log_info "Cloning dotfiles repository..."
    git clone https://github.com/Jannoshh/dotfiles.git
fi

cd dotfiles

# Run install script (zsh, tmux, oh-my-zsh)
log_info "Installing zsh and tmux configurations..."
./install.sh --zsh --tmux

# Deploy configurations
log_info "Deploying dotfiles..."
./deploy.sh --vim

# Change default shell to zsh
log_info "Changing default shell to zsh..."
sudo chsh -s $(which zsh) $(whoami)

log_info "Dotfiles setup complete ✓"

# =============================================================================
# 6. Create convenience startup script
# =============================================================================
log_info "Creating startup script..."

cat > ~/workspace/start_refusal_cones.sh << 'EOF'
#!/bin/bash
# Quick startup script for refusal-cones project

cd ~/workspace/refusal-cones
source .venv/bin/activate

echo "========================================="
echo "Refusal Cones Environment Activated"
echo "========================================="
echo ""
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Quick commands:"
echo "  - Train unified RDO: python examples/example_train_with_eval.py"
echo "  - Run baseline fitting: python examples/example_baseline_fitting.py"
echo "  - Check GPU: nvidia-smi"
echo ""
EOF

chmod +x ~/workspace/start_refusal_cones.sh

log_info "Startup script created at ~/workspace/start_refusal_cones.sh"

# =============================================================================
# 7. Setup activation in .zshrc
# =============================================================================
log_info "Adding activation to .zshrc..."

# Add activation to .zshrc if not already present
if ! grep -q "start_refusal_cones.sh" ~/.zshrc; then
    cat >> ~/.zshrc << 'EOF'

# Auto-activate refusal-cones environment
if [ -f ~/workspace/start_refusal_cones.sh ]; then
    alias rdc='source ~/workspace/start_refusal_cones.sh'
    echo "💡 Tip: Run 'rdc' to activate refusal-cones environment"
fi
EOF
    log_info "Added 'rdc' alias to .zshrc"
fi

# =============================================================================
# 8. Display summary
# =============================================================================
echo ""
echo "========================================="
echo "✓ RunPod Setup Complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✓ System dependencies installed"
echo "  ✓ Python 3.11 + uv installed"
echo "  ✓ GitHub authentication configured"
echo "  ✓ Refusal-cones project cloned to ~/workspace/refusal-cones"
echo "  ✓ ML dependencies installed (PyTorch, transformers, PEFT)"
echo "  ✓ Dotfiles configured (zsh, tmux, vim)"
echo "  ✓ Jupyter kernel installed"
echo ""
echo "Quick Start:"
echo "  1. Restart your shell or run: exec zsh"
echo "  2. Activate environment: rdc"
echo "  3. Start training: python examples/example_train_with_eval.py"
echo ""
echo "Useful commands:"
echo "  - rdc                  Activate refusal-cones environment"
echo "  - nvidia-smi           Check GPU status"
echo "  - nvtop                Interactive GPU monitor"
echo "  - htop                 System monitor"
echo ""
echo "Training tips:"
echo "  - Use Qwen2.5-0.5B-Instruct (default) for fast prototyping"
echo "  - Validation uses fast refusal token proxy (no generation)"
echo "  - Test evaluation uses HarmBench (slow but accurate)"
echo "  - Estimated time: ~15-30 min for 3 epochs on RTX 4090"
echo ""
echo "Happy training! 🚀"
echo "========================================="
