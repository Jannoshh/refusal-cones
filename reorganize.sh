#!/bin/bash
set -e

# Create directory structure
mkdir -p docs/{setup,architecture,discovery,training,integrations,legacy}
mkdir -p src/{discovery,training,measurement,utils}
mkdir -p examples
mkdir -p tests
mkdir -p scripts
mkdir -p legacy

# Move documentation files
# Setup guides
mv GPU_QUICKSTART.md docs/setup/
mv COLAB_QUICKSTART.md docs/setup/
mv NEXT_STEPS.md docs/setup/

# Architecture docs
mv SUMMARY.md docs/architecture/
mv IMPLEMENTATION_SUMMARY.md docs/architecture/
mv PROJECTION_AS_LORA.md docs/architecture/
mv WEIGHT_MODIFICATION_ANALYSIS.md docs/architecture/

# Discovery docs
mv GRADIENT_BASED_DISCOVERY.md docs/discovery/
mv ADAPTIVE_GEOMETRY_DISCOVERY.md docs/discovery/
mv EFFICIENT_HYPERSPHERE_EXPLORATION.md docs/discovery/
mv EXPLORATION_METHODS_COMPARISON.md docs/discovery/
mv GEOMETRY_COMPARISON.md docs/discovery/
mv SAMPLING_SPACE_EXPLAINED.md docs/discovery/

# Training docs
mv RDO_WITH_PEFT.md docs/training/
mv SFT_WITH_PEFT.md docs/training/
mv PER_LAYER_TRAINING.md docs/training/
mv ADVERSARIAL_RL_README.md docs/training/

# Integration docs
mv VLLM_INTEGRATION.md docs/integrations/
mv TRL_GRPO_INTEGRATION.md docs/integrations/
mv RL_FRAMEWORK_COMPARISON.md docs/integrations/

# Legacy docs
mv PORTING_NOTES.md docs/legacy/
mv CONVERSION_TEST_RESULTS.md docs/legacy/
mv COST_COMPARISON.md docs/legacy/

# Move source files
# Discovery
mv gradient_discovery.py src/discovery/
mv efficient_discovery.py src/discovery/
mv adaptive_geometry_discovery.py src/discovery/
mv adaptive_to_training.py src/discovery/

# Training
mv rdo_peft_adapter.py src/training/
mv rdo_peft_trainer.py src/training/
mv projection_adapter.py src/training/
mv per_layer_training.py src/training/
mv sft_peft_trainer.py src/training/
mv rl_vector_optimization.py src/training/
mv rl_adversarial.py src/training/
mv rl_grpo_trl_adapted.py src/training/

# Measurement
mv vllm_hybrid_measurement.py src/measurement/
mv scoring.py src/measurement/
mv example_measure_function.py src/measurement/

# Utils
mv model_utils.py src/utils/
mv generate_utils.py src/utils/
mv conversion_utils.py src/utils/

# Move example files
mv example_full_pipeline.py examples/
mv example_adversarial_training.py examples/
mv example_per_layer_training.py examples/
mv example_projection_adapter_training.py examples/
mv example_trl_comparison.py examples/
mv example_weight_vs_hooks_training.py examples/

# Move test files
mv test_port.py tests/
mv test_smooth_max.py tests/
mv test_subspaces.py tests/
mv test_conversion_equivalence.py tests/

# Move scripts
mv visualize_geometry.py scripts/

# Move legacy files
mv directopt.py legacy/
mv old_directopt.py legacy/
mv crossovereffects.py legacy/
mv plots.py legacy/
mv sampling.py legacy/
mv properties.py legacy/
mv targets.py legacy/
mv surrogate_scores.py legacy/
mv repind_gcg.py legacy/
mv repind_gcg_run.py legacy/
mv new_repind_gcg.py legacy/

# Rename Claude.md to README.md
mv Claude.md README.md

echo "Reorganization complete!"
