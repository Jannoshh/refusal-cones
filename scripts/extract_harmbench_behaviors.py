#!/usr/bin/env python3
"""
Extract HarmBench behaviors used in REINFORCE attacks paper.

Based on: https://github.com/sigeisler/reinforce-attacks-llms
They used indices 0-199 from harmbench_behaviors_text_test.csv
"""

import pandas as pd
import json
from pathlib import Path

# Load HarmBench test set from REINFORCE repo
harmbench_path = "/tmp/reinforce-attacks-llms/data/behavior_datasets/harmbench_behaviors_text_test.csv"

print("Loading HarmBench test set...")
df = pd.read_csv(harmbench_path)

print(f"Total behaviors in test set: {len(df)}")

# Extract the first 200 behaviors (indices 0-199) as used in REINFORCE paper
behaviors_used = df.iloc[:200]

print(f"Behaviors used in REINFORCE paper: {len(behaviors_used)}")

# Analyze distribution
print("\n" + "=" * 70)
print("CATEGORY DISTRIBUTION (FunctionalCategory)")
print("=" * 70)

func_dist = behaviors_used['FunctionalCategory'].value_counts()
for category, count in func_dist.items():
    print(f"  {category:<20} {count:>3} ({count/len(behaviors_used)*100:.1f}%)")

print("\n" + "=" * 70)
print("SEMANTIC CATEGORY DISTRIBUTION")
print("=" * 70)

sem_dist = behaviors_used['SemanticCategory'].value_counts()
for category, count in sem_dist.items():
    print(f"  {category:<30} {count:>3} ({count/len(behaviors_used)*100:.1f}%)")

# Filter out copyright and contextual (as they mention in paper)
print("\n" + "=" * 70)
print("FILTERING (Standard Behaviors Only)")
print("=" * 70)

# Standard behaviors = not copyright, not contextual
standard_behaviors = behaviors_used[
    (behaviors_used['FunctionalCategory'] == 'standard')
]

print(f"Standard behaviors: {len(standard_behaviors)} / {len(behaviors_used)}")
print(f"Filtered out: {len(behaviors_used) - len(standard_behaviors)}")

# Save to our repo format
output_dir = Path("/home/user/refusal-cones/data/harmbench")
output_dir.mkdir(parents=True, exist_ok=True)

# Save all 200 behaviors
all_behaviors = []
for _, row in behaviors_used.iterrows():
    all_behaviors.append({
        'instruction': row['Behavior'],
        'category': row['SemanticCategory'],
        'functional_category': row['FunctionalCategory'],
        'behavior_id': row['BehaviorID'],
    })

with open(output_dir / "harmbench_test_200.json", "w") as f:
    json.dump(all_behaviors, f, indent=2)

print(f"\n✓ Saved all 200 behaviors to: {output_dir / 'harmbench_test_200.json'}")

# Save standard behaviors only (what they actually evaluated on)
standard_only = []
for _, row in standard_behaviors.iterrows():
    standard_only.append({
        'instruction': row['Behavior'],
        'category': row['SemanticCategory'],
        'functional_category': row['FunctionalCategory'],
        'behavior_id': row['BehaviorID'],
    })

with open(output_dir / "harmbench_test_standard.json", "w") as f:
    json.dump(standard_only, f, indent=2)

print(f"✓ Saved {len(standard_only)} standard behaviors to: {output_dir / 'harmbench_test_standard.json'}")

# Print some example behaviors
print("\n" + "=" * 70)
print("EXAMPLE BEHAVIORS")
print("=" * 70)

for i in range(min(5, len(standard_behaviors))):
    row = standard_behaviors.iloc[i]
    print(f"\n{i+1}. [{row['SemanticCategory']}]")
    print(f"   {row['Behavior'][:100]}...")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Extracted {len(behaviors_used)} behaviors (indices 0-199)")
print(f"✓ {len(standard_behaviors)} are standard (non-copyright, non-contextual)")
print(f"✓ Ready to use for experiments matching REINFORCE paper")
