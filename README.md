# SOHIP Abel Transform and Onion Peeling Model Module

This software provides tools for analyzing and modeling physical systems using mathematical transforms and layered models. It includes (1) functions for performing the Abel transform, which is used to relate measurements of bending angles to properties such as refractive index and radius in a medium. The code can compute bending angles from input profiles and also reconstruct these profiles from observed data; (2) the functions for modeling systems with multiple layers using an onion-peeling approach, allowing users to simulate and analyze the behavior of layered materials or structures. These capabilities are useful for researchers and engineers working in fields such as optics, atmospheric science, and materials analysis, enabling them to interpret and model data from experiments or simulations relates to refraction in spherical symmetric medium.

## Quick Start

This module provides:
- forward_abel_sx3: Forward Abel transform, computing bending angle beta(a) from refractive index n(x)
  and tangent radius r_t(x), where the impact parameter a = n * r_t.
- invert_abel_sx3: Inverse Abel transform, reconstructing n and r_t from impact parameter a and beta(a).
- forward_slab: Onion-peeling forward model for multiple layers, vectorized across columns.
- invert_slab: Onion-peeling inversion with three method variants.

## Important notes

- Logic remains unchanged from the original implementation for tests in author's paper.
- Progress printing and interactive pauses are preserved.
- Inputs for Abel functions are intended as 1D arrays, for slab functions as 2D arrays with
  shape (layers, columns), where each column is processed independently.

## References

The forward Abel transform uses the form (eq(4.11) in García (2004)):
    β(a) = -2 a ∫_a^∞ [1 / √(x^2 - a^2)] [d ln(n) / dx] dx

Invert Abel uses (eq(4.12) in García (2004)):
    ln n(a) = (1 / π) ∫_a^∞ [β(x) / √(x^2 - a^2)] dx,
then n = exp(ln n), and r_t = a / n.

Reference: 
	García Fernández, M. (2004). Contributions to the 3D ionospheric sounding with GPS data. Universitat Politècnica de Catalunya. [Chapter 4]
	Xu, S., D. McGuffin, L. M. Simms, C. Laguna, H. Walter, (2025) A New Algorithm for Retrieval of Stratospheric Gravity Waves from Stellar Occultation Satellite Bending Angle Measurements (Under preparation)


## License

SOHIP Abel Transform and Onion Peeling Model Module is licensed under the MIT license, (MIT or https://opensource.org/license/mit/).

Copyrights and patents in the SOHIP project are retained by contributors.  No
copyright assignment is required to contribute to SOHIP.

Unlimited Open Source - MIT Distribution
`LLNL-CODE-2019387`
