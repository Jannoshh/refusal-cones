import torch
from tqdm import tqdm
import einops
from torch import Tensor
from jaxtyping import Float

def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj

def generate_completions(
    model,
    dataset: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 16,
):
    """
    Generate completions for a dataset of prompts.

    Args:
        model: HuggingFace model (or HookedModel)
        dataset: List of prompt strings
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for processing

    Returns:
        List of generated completions
    """
    all_completions = []

    # Get the actual model and tokenizer
    if hasattr(model, 'model'):
        # HookedModel wrapper
        base_model = model.model
        tokenizer = model.tokenizer
    else:
        # Direct HuggingFace model
        base_model = model
        tokenizer = model.tokenizer

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]

        # Tokenize inputs
        inputs = tokenizer(instructions, add_special_tokens=True, padding=True,
                          truncation=False, return_tensors='pt')
        start_token = inputs['input_ids'].shape[1]
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated part
        completion = tokenizer.batch_decode(outputs[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)

    return all_completions

def intervene_with_fn_vector_ablation(
    model,
    dataset: list[str],
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    batch_size: int = 16,
    do_sample=False,
    temperature=0,
):
    """
    Generate completions with directional ablation applied at every layer.

    Args:
        model: HuggingFace model (or HookedModel)
        dataset: List of prompt strings
        fn_vector: Direction vector to ablate
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for processing
        do_sample: Whether to use sampling
        temperature: Sampling temperature

    Returns:
        List of generated completions
    """
    fn_vector = fn_vector / fn_vector.norm()
    all_completions = []

    # Get the actual model and tokenizer
    if hasattr(model, 'model'):
        # HookedModel wrapper
        base_model = model.model
        tokenizer = model.tokenizer
        device = model.device
        dtype = model.dtype
    else:
        # Direct HuggingFace model
        base_model = model
        tokenizer = model.tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

    fn_vector = fn_vector.to(dtype).to(device)

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]

        # Tokenize inputs
        inputs = tokenizer(instructions, add_special_tokens=True, padding=True,
                          truncation=False, return_tensors='pt')
        start_token = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Define intervention hook
        def intervention_hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                modified_output = output[0] - projection_einops(output[0], fn_vector)
                return (modified_output,) + output[1:]
            else:
                return output - projection_einops(output, fn_vector)

        # Register hooks on all layers
        handles = []
        for layer in base_model.model.layers:
            # Hook on layer output
            handles.append(layer.register_forward_hook(intervention_hook))
            # Hook on self-attention output
            handles.append(layer.self_attn.register_forward_hook(intervention_hook))
            # Hook on MLP output
            handles.append(layer.mlp.register_forward_hook(intervention_hook))

        # Generate with interventions
        gen_kwargs = {'max_new_tokens': max_new_tokens, 'do_sample': do_sample}
        if temperature > 0:
            gen_kwargs['temperature'] = temperature
        gen_kwargs['pad_token_id'] = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = base_model.generate(**inputs, **gen_kwargs)

        # Clean up hooks
        for handle in handles:
            handle.remove()

        # Decode only the generated part
        completion = tokenizer.batch_decode(outputs[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)

    return all_completions


def intervene_with_fn_vector_addition(
    model,
    dataset: list[str],
    layer: int,
    alpha: float,
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    post_tokens: int = -1,
    batch_size: int = 16,
):
    """
    Generate completions with activation addition at a specific layer.

    Args:
        model: HuggingFace model (or HookedModel)
        dataset: List of prompt strings
        layer: Layer index to add the vector to
        alpha: Scaling factor for the vector
        fn_vector: Direction vector to add
        max_new_tokens: Maximum number of tokens to generate
        post_tokens: Number of tokens to apply intervention for (-1 for all)
        batch_size: Batch size for processing

    Returns:
        List of generated completions
    """
    fn_vector = alpha * fn_vector / fn_vector.norm()
    all_completions = []

    # Get the actual model and tokenizer
    if hasattr(model, 'model'):
        # HookedModel wrapper
        base_model = model.model
        tokenizer = model.tokenizer
        device = model.device
        dtype = model.dtype
    else:
        # Direct HuggingFace model
        base_model = model
        tokenizer = model.tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

    fn_vector = fn_vector.to(dtype).to(device)

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]

        # Tokenize inputs
        inputs = tokenizer(instructions, add_special_tokens=True, padding=True,
                          truncation=False, return_tensors='pt')
        start_token = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # For post_tokens limitation, we need to track generation step
        # Since we're using standard generation, we'll apply intervention for all steps
        # Note: Exact post_tokens tracking would require custom generation loop
        # For now, we apply the intervention throughout generation

        # Define intervention hook
        def intervention_hook(module, input, output):
            # Add vector to output
            if isinstance(output, tuple):
                modified_output = output[0] + fn_vector
                return (modified_output,) + output[1:]
            else:
                return output + fn_vector

        # Register hook on specific layer
        handle = base_model.model.layers[layer].register_forward_hook(intervention_hook)

        # Generate with intervention
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Clean up hook
        handle.remove()

        # Decode only the generated part
        completion = tokenizer.batch_decode(outputs[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)

    return all_completions