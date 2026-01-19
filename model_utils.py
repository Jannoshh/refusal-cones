"""
Utility module for PyTorch hooks to replace nnsight functionality.
Provides similar API for activation interventions during forward passes and generation.
"""

import torch
from torch import nn
from typing import Callable, Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager


class HookedModel:
    """
    Wrapper around HuggingFace models that provides activation intervention capabilities
    using PyTorch hooks, similar to nnsight's LanguageModel.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize a HookedModel.

        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=kwargs.get('cache_dir'))
        self.device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        self.dtype = kwargs.get('torch_dtype', torch.float32)

        # Storage for hooks and saved values
        self.hooks = []
        self.saved_values = {}
        self.intervention_fn = None

    def requires_grad_(self, requires_grad: bool):
        """Set requires_grad for all model parameters."""
        self.model.requires_grad_(requires_grad)
        return self

    @contextmanager
    def trace(self, prompts, **tokenizer_kwargs):
        """
        Context manager for tracing a forward pass with optional interventions.

        Args:
            prompts: String or list of strings to process
            **tokenizer_kwargs: Additional arguments for tokenization
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize inputs
        default_kwargs = {'add_special_tokens': True, 'padding': True, 'truncation': False, 'return_tensors': 'pt'}
        default_kwargs.update(tokenizer_kwargs)
        inputs = self.tokenizer(prompts, **default_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Storage for this trace
        self.saved_values = {}
        self.hooks = []

        try:
            # Register hooks if intervention function is set
            if self.intervention_fn is not None:
                self._register_intervention_hooks()

            # Run forward pass
            with torch.no_grad():
                self.outputs = self.model(**inputs, output_hidden_states=True)

            yield self
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            self.intervention_fn = None

    @contextmanager
    def generate(self, prompts=None, max_new_tokens=150, do_sample=False, temperature=None, **kwargs):
        """
        Context manager for generation with optional interventions.

        Args:
            prompts: String or list of strings to generate from (can be None if using invoke)
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        """
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
        }
        if temperature is not None:
            gen_kwargs['temperature'] = temperature
        gen_kwargs.update(kwargs)

        # Create generator context
        generator = GeneratorContext(self, prompts, gen_kwargs)

        try:
            yield generator
        finally:
            # Clean up
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            self.intervention_fn = None

    def _register_intervention_hooks(self):
        """Register hooks for interventions on all layers."""
        if self.intervention_fn is None:
            return

        def create_hook(layer_idx):
            def hook(module, input, output):
                return self.intervention_fn(module, input, output, layer_idx)
            return hook

        # Register hooks on each layer
        for idx, layer in enumerate(self.model.model.layers):
            handle = layer.register_forward_hook(create_hook(idx))
            self.hooks.append(handle)


class GeneratorContext:
    """Context for managing generation with interventions."""

    def __init__(self, hooked_model, prompts, gen_kwargs):
        self.hooked_model = hooked_model
        self.prompts = prompts
        self.gen_kwargs = gen_kwargs
        self.output = None
        self.invoker = None

    @contextmanager
    def invoke(self, prompts):
        """Set the prompts to use for generation."""
        self.prompts = prompts
        self.invoker = InvokerContext(self.hooked_model, prompts, self.gen_kwargs)

        try:
            yield self.invoker
        finally:
            # Store output
            if self.invoker.output_ids is not None:
                self.output = SavedValue(self.invoker.output_ids)


class InvokerContext:
    """Context for token-by-token generation with interventions."""

    def __init__(self, hooked_model, prompts, gen_kwargs):
        self.hooked_model = hooked_model
        self.prompts = prompts
        self.gen_kwargs = gen_kwargs
        self.output_ids = None
        self.current_token_idx = 0

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        inputs = hooked_model.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors='pt'
        )
        self.input_ids = inputs['input_ids'].to(hooked_model.device)
        self.attention_mask = inputs['attention_mask'].to(hooked_model.device)

        # Initialize output_ids with input_ids
        self.output_ids = self.input_ids.clone()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class SavedValue:
    """Wrapper for saved values, similar to nnsight's saved values."""

    def __init__(self, value):
        self.value = value


def get_layer_activations(model, prompts, intervention_fn=None, batch_size=8):
    """
    Get activations from all layers with optional interventions.

    Args:
        model: HookedModel instance
        prompts: List of prompts
        intervention_fn: Function to apply interventions (takes layer, activations)
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping layer indices to their activations
    """
    all_activations = {i: [] for i in range(len(model.model.model.layers))}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = model.tokenizer(
            batch_prompts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors='pt'
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Storage for this batch
        batch_activations = {}

        def create_hook(layer_idx):
            def hook(module, input, output):
                # Store the output
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                batch_activations[layer_idx] = activation.detach()

                # Apply intervention if provided
                if intervention_fn is not None:
                    modified = intervention_fn(layer_idx, activation)
                    if modified is not None:
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified
                return output
            return hook

        # Register hooks
        handles = []
        for idx, layer in enumerate(model.model.model.layers):
            handle = layer.register_forward_hook(create_hook(idx))
            handles.append(handle)

        # Forward pass
        with torch.no_grad():
            _ = model.model(**inputs)

        # Clean up hooks
        for handle in handles:
            handle.remove()

        # Store activations
        for layer_idx, acts in batch_activations.items():
            all_activations[layer_idx].append(acts.cpu())

    # Concatenate batches
    for layer_idx in all_activations:
        if all_activations[layer_idx]:
            all_activations[layer_idx] = torch.cat(all_activations[layer_idx], dim=0)
        else:
            all_activations[layer_idx] = None

    return all_activations


def apply_intervention_to_layers(model, prompts, intervention_fn, batch_size=8):
    """
    Apply interventions to layers during forward pass.

    Args:
        model: HookedModel instance or regular HuggingFace model
        prompts: List of prompts
        intervention_fn: Function(layer_module, layer_input, layer_output, layer_idx) -> modified_output
        batch_size: Batch size for processing

    Returns:
        Model outputs
    """
    # Handle both HookedModel and raw models
    if isinstance(model, HookedModel):
        base_model = model.model
        tokenizer = model.tokenizer
        device = model.device
    else:
        base_model = model
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    all_outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        def create_hook(layer_idx):
            def hook(module, input, output):
                return intervention_fn(module, input, output, layer_idx)
            return hook

        # Register hooks
        handles = []
        for idx, layer in enumerate(base_model.model.layers):
            handle = layer.register_forward_hook(create_hook(idx))
            handles.append(handle)

        # Forward pass
        with torch.no_grad():
            outputs = base_model(**inputs)

        # Clean up hooks
        for handle in handles:
            handle.remove()

        all_outputs.append(outputs)

    return all_outputs
