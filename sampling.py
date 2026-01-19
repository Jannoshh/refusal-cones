# %%
import torch

def sample_hypersphere_gaussian(batch_size, dim):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

from torch.quasirandom import SobolEngine
import numpy as np

def sample_hypersphere_sobol(batch_size, dim):
    # Initialize Sobol sequence generator
    sobol = SobolEngine(dimension=dim)
    
    # Generate Sobol sequence points in [0,1]^dim
    samples = sobol.draw(batch_size)
    
    # Transform to positive quadrant of normal distribution using inverse CDF
    # We use inverse error function (erfinv) for this transformation
    samples = torch.erfinv(2 * samples - 1) * np.sqrt(2)
    samples = samples.abs()  # Keep only positive values
    
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    
    return samples

def sample_prob_vectors(batch_size, dim):
    samples = torch.exp(torch.randn(batch_size, dim))
    samples = samples / samples.sum(dim=1, keepdim=True)
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

def sample_hypersphere_sobol_v2(batch_size, dim):
    # For n-sphere in first orthant, we need dim-1 angles
    sobol = SobolEngine(dimension=dim-1)
    
    # Generate Sobol points
    u = sobol.draw(batch_size)
    
    # Transform angles to account for spherical area element
    angles = torch.zeros_like(u)
    angles[:, :-1] = torch.arccos(torch.sqrt(u[:, :-1]))  # [0, π/2]
    angles[:, -1] = u[:, -1] * (np.pi/2)  # Last angle uniform in [0, π/2]
    
    # Initialize coordinates
    coords = torch.ones(batch_size, dim)
    
    # Convert to Cartesian coordinates
    for i in range(dim):
        if i < dim-1:
            coords[:, i] = torch.cos(angles[:, i])
            if i > 0:
                coords[:, i] *= torch.prod(torch.sin(angles[:, :i]), dim=1)
        else:
            coords[:, i] = torch.prod(torch.sin(angles[:, :dim-1]), dim=1)
    
    return coords
# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def generate_prime_numbers(n: int) -> torch.Tensor:
    """Generate first n prime numbers using the Sieve of Eratosthenes."""
    size = max(n * 20, 100)
    sieve = torch.ones(size, dtype=torch.bool)
    sieve[0] = sieve[1] = False
    
    primes = []
    for i in range(2, size):
        if sieve[i]:
            primes.append(i)
            if len(primes) == n:
                return torch.tensor(primes)
            
            sieve[i * i::i] = False
    
    return torch.tensor(primes)

def halton_sequence_batch(batch_size: int, dim: int, skip: int = 100) -> torch.Tensor:
    """Generate batch_size points of dim-dimensional Halton sequence."""
    primes = generate_prime_numbers(dim)
    points = torch.zeros((batch_size, dim))
    
    for d, prime in enumerate(primes):
        sequence = torch.zeros(batch_size)
        for i in range(batch_size):
            idx = i + skip + 1
            f = 1
            result = 0
            
            while idx > 0:
                f = f / prime
                result += f * (idx % prime)
                idx = idx // prime
                
            sequence[i] = result
        points[:, d] = sequence
    
    return points

def sample_hypersphere_halton(batch_size: int, dim: int) -> torch.Tensor:
    """Sample from unit hypersphere using Halton sequence with correct spherical measure."""
    # Generate raw Halton sequence points
    points = halton_sequence_batch(batch_size, dim-1)  # We need dim-1 angles
    
    # Transform angles with correct spherical measure
    angles = torch.zeros(batch_size, dim-1)
    
    # For the first n-2 angles, use arccos(1-2x) to get proper distribution on [0,π/2]
    if dim > 2:
        angles[:, :-1] = torch.arccos(1 - 2 * points[:, :-1])  # Map to [0, π/2]
    
    # Last angle is uniform in [0, π/2]
    angles[:, -1] = points[:, -1] * (np.pi/2)
    
    # Initialize coordinates
    result = torch.ones(batch_size, dim)
    
    # Convert to Cartesian coordinates
    for i in range(dim):
        if i < dim-1:
            result[:, i] = torch.cos(angles[:, i])
            if i > 0:
                result[:, i] *= torch.prod(torch.sin(angles[:, :i]), dim=1)
        else:
            result[:, i] = torch.prod(torch.sin(angles[:, :dim-1]), dim=1)
    
    return result
# Test functions for distribution verification
def compute_statistics(samples: torch.Tensor, n_bins: int = 20) -> dict:
    """Compute statistics to verify the uniformity of the distribution."""
    n_samples, dim = samples.shape
    
    # Test 1: Distance from origin (should all be ~1)
    norms = torch.norm(samples, dim=1)
    norm_stats = {
        'mean_norm': float(torch.mean(norms)),
        'std_norm': float(torch.std(norms)),
        'max_norm_error': float(torch.max(torch.abs(norms - 1)))
    }
    
    # Test 2: Histogram analysis for each coordinate
    coord_stats = []
    for d in range(dim):
        hist = torch.histogram(samples[:, d], bins=n_bins, range=(0, 1))
        counts = hist.hist
        uniformity = float(torch.std(counts) / torch.mean(counts))
        coord_stats.append(uniformity)
    
    # Test 3: Pairwise angle distributions (for 3D)
    if dim == 3:
        xy_angles = torch.atan2(samples[:, 1], samples[:, 0])
        xz_angles = torch.atan2(samples[:, 2], samples[:, 0])
        yz_angles = torch.atan2(samples[:, 2], samples[:, 1])
        
        angle_stats = {
            'xy_uniformity': float(torch.std(torch.histogram(xy_angles, bins=n_bins)[0]) / 
                                 torch.mean(torch.histogram(xy_angles, bins=n_bins)[0])),
            'xz_uniformity': float(torch.std(torch.histogram(xz_angles, bins=n_bins)[0]) / 
                                 torch.mean(torch.histogram(xz_angles, bins=n_bins)[0])),
            'yz_uniformity': float(torch.std(torch.histogram(yz_angles, bins=n_bins)[0]) / 
                                 torch.mean(torch.histogram(yz_angles, bins=n_bins)[0]))
        }
    else:
        angle_stats = {}
    
    return {
        'norm_stats': norm_stats,
        'coord_uniformity': coord_stats,
        'angle_stats': angle_stats
    }

def sample_positive_orthant_sphere_surface_torch(N, dim):
    '''
    Generate N samples uniformly distributed on the positive orthant surface of the n-dimensional unit sphere
    using the Sobol sequence in PyTorch.

    Parameters:
    n : int
        Dimension of the sphere.
    N : int
        Number of samples.

    Returns:
    samples : torch.Tensor of shape (N, n)
        The generated samples lying on the positive orthant of the unit sphere.
    '''
    # Create a Sobol sequence generator for n dimensions
    soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
    
    # Generate N samples in (0,1)
    sample = soboleng.draw(N)  # shape (N, n)
    
    # To avoid issues with log(0), clip the samples away from 0
    u = torch.clamp(sample, min=1e-10, max=1.0)
    
    # Transform to exponential variables
    x = -torch.log(u)  # shape (N, n)
    
    # Normalize each sample vector to have unit norm
    norms = torch.norm(x, p=2, dim=1, keepdim=True)
    samples = x / norms  # shape (N, n)
    
    return samples

import numpy as np
from typing import List, Optional

def van_der_corput(n: int, base: int) -> float:
    """
    Generate the nth number in the Van der Corput sequence with given base.
    
    Args:
        n: Position in sequence (starting from 0)
        base: Base for sequence generation
    
    Returns:
        float: nth number in Van der Corput sequence
    """
    vdc, denom = 0.0, 1.0
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

def halton(index: int, dimension: int) -> List[float]:
    """
    Generate point from Halton sequence.
    
    Args:
        index: Index in sequence
        dimension: Number of dimensions
    
    Returns:
        List[float]: Point coordinates
    """
    # First few prime numbers for bases
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dimension > len(bases):
        raise ValueError(f"Dimension {dimension} exceeds available prime bases")
    
    return [van_der_corput(index, bases[d]) for d in range(dimension)]

def spherical_to_cartesian(coords: List[float]) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.
    First coordinate is radius, remaining are angles.
    
    Args:
        coords: List of spherical coordinates [r, φ₁, φ₂, ..., φₙ₋₁]
    
    Returns:
        np.ndarray: Cartesian coordinates
    """
    r = coords[0]
    n = len(coords)
    x = np.zeros(n)
    
    # First coordinate
    sin_product = r
    for i in range(1, n-1):
        sin_product *= np.sin(coords[i] * np.pi)
    x[0] = sin_product * np.cos(coords[-1] * 2 * np.pi)
    
    # Middle coordinates
    for i in range(1, n-1):
        sin_product = r
        for j in range(1, i):
            sin_product *= np.sin(coords[j] * np.pi)
        sin_product *= np.cos(coords[i] * np.pi)
        x[i] = sin_product
    
    # Last coordinate
    sin_product = r
    for i in range(1, n-1):
        sin_product *= np.sin(coords[i] * np.pi)
    x[-1] = sin_product * np.sin(coords[-1] * 2 * np.pi)
    
    return x

def sample_hypersphere(n_samples: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from the unit hypersphere surface using Halton sequence.
    
    Args:
        n_samples: Number of samples to generate
        dimension: Dimension of the hypersphere
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Array of shape (n_samples, dimension) containing samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = np.zeros((n_samples, dim))
    
    for i in range(n_samples):
        # Generate Halton sequence point
        halton_point = halton(i, dim)
        
        # First coordinate is radius (always 1 for unit hypersphere surface)
        spherical_coords = [1.0]
        
        # Remaining coordinates are angles
        spherical_coords.extend(halton_point[:-1])
        
        # Convert to Cartesian coordinates
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        samples[i] = cartesian_coords
    # filter such taht all coefficients are positive
    samples = samples[samples.all(axis=1)]
    
    return torch.tensor(samples)

# Test all sampling methods
def test_sampling_methods(n_samples=1000, dim=3):
    methods = {
        # 'Halton': sample_hypersphere_halton,
        'Sobol Torch': sample_positive_orthant_sphere_surface_torch
    }
    
    results = {}
    for name, method in methods.items():
        samples = method(n_samples, dim)
        stats = compute_statistics(samples)
        results[name] = stats
        
        print(f"\n{name} Statistics:")
        print(f"Mean norm: {stats['norm_stats']['mean_norm']:.6f}")
        print(f"Norm std: {stats['norm_stats']['std_norm']:.6f}")
        print(f"Max norm error: {stats['norm_stats']['max_norm_error']:.6f}")
        print(f"Coordinate uniformity: {[f'{u:.4f}' for u in stats['coord_uniformity']]}")
        if stats['angle_stats']:
            print("Angle uniformity:")
            for k, v in stats['angle_stats'].items():
                print(f"  {k}: {v:.4f}")
    
    return results

# Update the visualization code to include Halton sampling
def visualize_sampling_methods(n_samples=64):
    # Create figures for 2D and 3D plots
    fig_2d, axes_2d = plt.subplots(1, 7, figsize=(20, 5))
    fig_3d = plt.figure(figsize=(20, 5))
    
    # Sample and plot for each method
    methods = [
        (sample_hypersphere_gaussian, "Gaussian"),
        (sample_hypersphere_sobol, "Sobol"), 
        (sample_hypersphere_sobol_v2, "Sobol v2"),
        (sample_prob_vectors, "Probability Vectors"),
        (sample_hypersphere_halton, "Halton"),
        (sample_positive_orthant_sphere_surface_torch, "Sobol Torch"),
        (sample_hypersphere, "Sobol Torch 2")
    ]
    
    # 2D plots
    for (sample_fn, title), ax in zip(methods, axes_2d):
        samples = sample_fn(n_samples, dim=2).numpy()
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
        ax.set_title(f"{title} (2D)")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.set_aspect('equal')
    
    # 3D plots
    for i, (sample_fn, title) in enumerate(methods):
        ax = fig_3d.add_subplot(1, 7, i+1, projection='3d')
        samples = sample_fn(n_samples*10, dim=3).numpy()
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5, s=10)
        ax.set_title(f"{title} (3D)")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_zlim(0, 1.1)
        ax.grid(True)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

# %%
visualize_sampling_methods(n_samples=64)
# test_sampling_methods(n_samples=1000, dim=3)

# %%
import torch

def sample_unit_sphere_surface_torch(N, dim):
    '''
    Generate N samples uniformly distributed on the surface of the n-dimensional unit sphere
    using the Sobol sequence in PyTorch.

    Parameters:
    N : int
        Number of samples.

    Returns:
    samples : torch.Tensor of shape (N, n)
        The generated samples lying on the surface of the unit sphere.
    '''
    # Create a Sobol sequence generator for n dimensions
    soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
    
    # Generate N samples in [0,1)
    sample = soboleng.draw(N)  # shape (N, n)
    
    # To avoid infinities in the inverse CDF, clip the samples away from 0 and 1
    u = torch.clamp(sample, min=1e-10, max=1.0 - 1e-10)
    
    # Map the uniform samples to standard normal samples using the inverse CDF
    normal = torch.distributions.normal.Normal(0, 1)
    z = normal.icdf(u)  # shape (N, n)
    
    # Normalize each sample vector to have unit norm
    norms = torch.norm(z, p=2, dim=1, keepdim=True)
    samples = z / norms
    
    return samples

# Example usage:
n = 5    # Dimension of the sphere
N = 1000  # Number of samples
samples = sample_unit_sphere_surface_torch(n, N)

# Verify that the samples lie on the unit sphere surface
norms = torch.norm(samples, p=2, dim=1)
print("Minimum norm:", norms.min().item())
print("Maximum norm:", norms.max().item())

# %%
