"""
Copyright (c) 2026, Lawrence Livermore National Security, LLC.  All rights 
reserved.  LLNL-CODE-2019387
MIT License
This work was produced at the Lawrence Livermore National Laboratory (LLNL) 
under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department 
of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the 
operation of LLNL.  See license for disclaimers, notice of U.S. Government 
Rights and license terms and conditions.
@authors: Shuang Xu
"""

"""
SOHIP Abel Transform and Onion Peeling Model Module
Created on Thu Oct 16 11:29:12 2025
@author: xu27@llnl.gov

This module provides:
- forward_abel_sx3: Forward Abel transform, computing bending angle beta(a) from refractive index n(x)
  and tangent radius r_t(x), where the impact parameter a = n * r_t.
- invert_abel_sx3: Inverse Abel transform, reconstructing n and r_t from impact parameter a and beta(a).
- forward_slab: Onion-peeling forward model for multiple layers, vectorized across columns.
- invert_slab: Onion-peeling inversion with three method variants.

Important notes:
- Logic remains unchanged from the original implementation for tests in author's paper.
- Progress printing and interactive pauses are preserved.
- Inputs for Abel functions are intended as 1D arrays, for slab functions as 2D arrays with
  shape (layers, columns), where each column is processed independently.

References:
The forward Abel transform uses the form (eq(4.11) in García (2004)):
    β(a) = -2 a ∫_a^∞ [1 / √(x^2 - a^2)] [d ln(n) / dx] dx

Invert Abel uses (eq(4.12) in García (2004)):
    ln n(a) = (1 / π) ∫_a^∞ [β(x) / √(x^2 - a^2)] dx,
then n = exp(ln n), and r_t = a / n.

Reference: García Fernández, M. (2004). Contributions to the 3D ionospheric sounding with GPS data. Universitat Politècnica de Catalunya. [Chapter 4]
"""

from datetime import datetime
import warnings
from typing import Tuple

import numpy as np


def forward_abel_sx3(r_t_arr: np.ndarray, n_arr: np.ndarray, verbose: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Abel transform:
    Computes bending angle β(a) from tangent radius r_t and refractive index n. The impact parameter is a = n * r_t.

    The integral is implemented numerically as:
        β(a) = -2 a ∫_a^∞ [1 / √(x^2 - a^2)] [d ln(n) / dx] dx

    Parameters
    - r_t_arr: 1D array of tangent radius values, x-domain where n is defined.
    - n_arr:   1D array of refractive index values corresponding to r_t_arr.
    - verbose: 1 to print progress milestones, 0 to disable.

    Returns
    - impact_arr: 1D array of impact parameter a = n * r_t, mapped back to original input order.
    - beta_arr:   1D array of bending angle β(a), mapped back to original input order.

    Notes
    - The computation sorts by r_t in ascending order internally to respect the Abel integral domain.
    - d ln(n) / dx is approximated by a centered finite difference with np.gradient using the impact parameter grid.
    - The integral excludes the singular point by starting at index i+1 for each a[i].
    - Invalid regions will yield NaN in β, and the corresponding impact parameters are set to NaN.
    """
    # Validate vector lengths, preserve interactive behavior
    if len(r_t_arr) != len(n_arr):
        warnings.warn("Warning (ropp): The arrays have different numbers of elements!", UserWarning)
        input("Press Enter to continue...")
    else:
        print("(ropp): The arrays have the same number of elements.")

    # Sort by r_t ascending to satisfy integral domain [a, ∞)
    indices_sorted = np.argsort(r_t_arr)
    indices_back = np.argsort(indices_sorted)
    r_t_arr = r_t_arr[indices_sorted]
    n_arr = n_arr[indices_sorted]

    # Impact parameter a = n * r_t
    impact_arr = r_t_arr * n_arr

    # Numerical derivative d ln(n) / dx on the impact parameter grid
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ln_n_dx = np.gradient(np.log(n_arr), impact_arr)

    beta_arr = np.zeros(impact_arr.size)

    for i in np.arange(impact_arr.size):
        # Integrand: [1 / √(x^2 - a^2)] * d ln(n) / dx
        denom = np.sqrt(impact_arr**2 - impact_arr[i]**2)
        integrand = (1.0 / denom) * d_ln_n_dx

        # Trapezoidal integration over x ∈ (a, ∞), skip singular point at x = a
        beta_arr[i] = -2.0 * impact_arr[i] * np.trapz(integrand[i + 1:], impact_arr[i + 1:])

        if verbose == 1:
            if (i + 1) == int(n_arr.shape[0] * 0.01):
                print("1 % of invert steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.2):
                print("20 % of invert steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.4):
                print("40 % of invert steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.6):
                print("60 % of invert steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.8):
                print("80 % of invert steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0]):
                print("100 % of invert steps finished @", datetime.now())

    # Mark invalid impact parameters if β is NaN
    impact_arr[np.where(np.isnan(beta_arr))[0]] = np.nan

    # Map back to original input order
    return impact_arr[indices_back], beta_arr[indices_back]


def invert_abel_sx3(impact_arr: np.ndarray, beta_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse Abel transform:
    Reconstructs refractive index n and tangent radius r_t from impact parameter a and bending angle β(a).

    Implements:
        ln n(a) = (1 / π) ∫_a^∞ [β(x) / √(x^2 - a^2)] dx
    Then:
        n = exp(ln n)
        r_t = a / n

    Parameters
    - impact_arr: 1D array of impact parameters a.
    - beta_arr:   1D array of bending angles β(a) corresponding to impact_arr.

    Returns
    - r_t_arr: 1D array of tangent radius values mapped back to original input order.
    - n_arr:   1D array of refractive index values mapped back to original input order.

    Notes
    - Sorting by a in ascending order is performed internally to satisfy the integral domain.
    - The integral excludes the singular point by starting at index i+1 for each a[i].
    """
    # Validate vector lengths, preserve interactive behavior
    if len(impact_arr) != len(beta_arr):
        warnings.warn("Warning (ropp): The arrays have different numbers of elements!", UserWarning)
        input("Press Enter to continue...")
    else:
        print("(ropp): The arrays have the same number of elements.")

    # Sort by a ascending
    indices_sorted = np.argsort(impact_arr)
    indices_back = np.argsort(indices_sorted)
    impact_arr = impact_arr[indices_sorted]
    beta_arr = beta_arr[indices_sorted]

    # Left-hand side of inversion, ln n(a)
    leftside_arr = np.zeros(beta_arr.size)

    for i in np.arange(impact_arr.size):
        # Integrand: β(x) / √(x^2 - a^2)
        denom = np.sqrt(impact_arr**2 - impact_arr[i]**2)
        integrand = beta_arr / denom

        # Trapezoidal integration over x ∈ (a, ∞), skip singular point at x = a
        leftside_arr[i] = (1.0 / np.pi) * np.trapz(integrand[i + 1:], impact_arr[i + 1:])

    # Recover n and r_t
    n_arr = np.exp(leftside_arr)
    r_t_arr = impact_arr / n_arr

    # Mark invalid r_t if n is NaN
    r_t_arr[np.where(np.isnan(n_arr))[0]] = np.nan

    # Map back to original input order
    return r_t_arr[indices_back], n_arr[indices_back]


def forward_slab(r_t_arr: np.ndarray, n_arr: np.ndarray, verbose: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Onion-peeling forward model, vectorized across columns.

    Inputs are expected to be 2D arrays shaped (layers, columns), where each column represents an independent profile.
    This function:
      - Sorts each column of r_t in descending order.
      - Inserts a boundary layer at the top: r_t_top = 2*r_t[0] - r_t[1], and n_top = 1.
      - Computes the impact parameter a = n * r_t.
      - Computes the bending angle beta per layer using cumulative arcsin differences.

    Parameters
    - r_t_arr: 2D array (layers, columns) of tangent radii per column.
    - n_arr:   2D array (layers, columns) of refractive indices per column.
    - verbose: 1 to print progress milestones, 0 to disable.

    Returns
    - impact_arr: 2D array (layers-1, columns), original column order restored after removing the inserted top layer.
    - beta_arr:   2D array (layers-1, columns), original column order restored after removing the inserted top layer.

    Example
        test_r = np.array([5.5, 5.4, 5.3, 5.2, 5.1]).reshape(1, -1)
        test_n = np.array([1.00, 1.01, 1.02, 1.03, 1.04]).reshape(1, -1)

        test_i, test_b = forward_slab(test_r.T, test_n.T)
        test_i, test_b = test_i.T, test_b.T
        print(test_i, test_b)
        # Expected:
        # [[5.5   5.454 5.406 5.356 5.304]] [[0.         0.12311673 0.21393432 0.28642569 0.34640879]]

        test_r2, test_n2 = invert_slab(test_i.T, test_b.T, method_matrix=3)
        test_r2, test_n2 = test_r2.T, test_n2.T
        print(test_r2, test_n2)
        # Expected:
        # [[5.5 5.4 5.3 5.2 5.1]] [[1.   1.01 1.02 1.03 1.04]]
    """
    # Validate vector lengths, preserve interactive behavior
    if len(r_t_arr) != len(n_arr):
        warnings.warn("Warning (ropp): The arrays have different numbers of elements!", UserWarning)
        input("Press Enter to continue...")
    else:
        print("(qualified inputs): The arrays have the same number of elements.")

    # Sort each column of r_t in descending order, track indices to map back
    indices_sorted = np.argsort(-r_t_arr, axis=0)
    indices_back = np.argsort(indices_sorted, axis=0)

    # Gather sorted arrays per column
    r_t_arr = r_t_arr[indices_sorted, np.arange(r_t_arr.shape[1])]
    n_arr = n_arr[indices_sorted, np.arange(r_t_arr.shape[1])]

    # Insert top boundary layer:
    # r_t_top = 2*r_t[0] - r_t[1], n_top = 1
    r_t_arr = np.concatenate(([2.0 * r_t_arr[0] - r_t_arr[1]], r_t_arr))
    n_arr = np.concatenate(([n_arr[0]], n_arr))
    n_arr[0] = 1.0

    # Impact parameter per layer and column
    impact_arr = r_t_arr * n_arr

    # Accumulate bending angle β across layers
    beta_arr = np.zeros(n_arr.shape)

    for i in range(1, n_arr.shape[0]):
        # Clip ratios for arcsin to avoid numerical issues outside [-1, 1]
        theta_b_i = np.arcsin(
            np.clip((n_arr[i] / n_arr[1:i + 1]) * (r_t_arr[i] / r_t_arr[:i]), -1.0, 1.0)
        )
        theta_a_i = np.arcsin(
            np.clip((n_arr[i] / n_arr[:i]) * (r_t_arr[i] / r_t_arr[:i]), -1.0, 1.0)
        )

        # β_i is the cumulative 2 * sum of angle differences for layers above
        beta_arr[i] += 2.0 * (np.nansum(theta_a_i - theta_b_i, axis=0))

        if verbose == 1:
            if (i + 1) == int(n_arr.shape[0] * 0.2):
                print("20 % of forward steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.4):
                print("40 % of forward steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.6):
                print("60 % of forward steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0] * 0.8):
                print("80 % of forward steps finished @", datetime.now())
            if (i + 1) == int(n_arr.shape[0]):
                print("100 % of forward steps finished @", datetime.now())

    # Mark impact parameters as NaN where β is NaN
    impact_arr[np.where(np.isnan(beta_arr))[0]] = np.nan

    # Note: This print is part of the original logic
    print("top beta:", (beta_arr[beta_arr > 0])[0])

    # Return arrays without the inserted top layer, mapped back to original column order
    return (impact_arr[1:])[indices_back, np.arange(r_t_arr.shape[1])], (beta_arr[1:])[
        indices_back, np.arange(r_t_arr.shape[1])
    ]


def invert_slab(
    impact_arr: np.ndarray,
    beta_arr: np.ndarray,
    method_matrix: int = 1,
    verbose: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Onion-peeling inversion model with three method variants.

    Parameters
    - impact_arr: 1D or 2D array of impact parameter values. For method 3, shape (layers, columns).
    - beta_arr:   1D or 2D array of bending angle values corresponding to impact_arr.
    - method_matrix:
          1: Matrix method using per-layer theta_a and theta_b accumulation, higher precision typically.
          2: Alternative using precomputed theta_a from impact ratios only, lower precision in tests.
          3: Vectorized variant for multiple columns, equivalent logic to method 1 but space-efficient.
    - verbose: 1 to print progress milestones for method 3, 0 to disable.

    Returns
    - r_t_arr: Reconstructed tangent radius array, with the inserted top layer removed and original order restored.
    - n_arr:   Reconstructed refractive index array, with the inserted top layer removed and original order restored.

    Notes
    - Sorting in descending order and insertion of a top boundary layer are part of the original logic.
    - The first beta element is set to 0 per original code.
    
    Example
        test_r = np.array([5.5, 5.4, 5.3, 5.2, 5.1]).reshape(1, -1)
        test_n = np.array([1.00, 1.01, 1.02, 1.03, 1.04]).reshape(1, -1)

        test_i, test_b = forward_slab(test_r.T, test_n.T)
        test_i, test_b = test_i.T, test_b.T
        print(test_i, test_b)
        # Expected:
        # [[5.5   5.454 5.406 5.356 5.304]] [[0.         0.12311673 0.21393432 0.28642569 0.34640879]]

        test_r2, test_n2 = invert_slab(test_i.T, test_b.T, method_matrix=3)
        test_r2, test_n2 = test_r2.T, test_n2.T
        print(test_r2, test_n2)
        # Expected:
        # [[5.5 5.4 5.3 5.2 5.1]] [[1.   1.01 1.02 1.03 1.04]]
    """
    # Validate vector lengths, preserve interactive behavior
    if len(impact_arr) != len(beta_arr):
        warnings.warn("Warning (ropp): The arrays have different numbers of elements!", UserWarning)
        input("Press Enter to continue...")
    else:
        print("(qualified inputs): The arrays have the same number of elements.")

    # Sorting and top-layer insertion depend on whether we operate on single or multiple columns
    if method_matrix != 3:
        # Single group of observations
        indices_sorted = np.argsort(-impact_arr)
        indices_back = np.argsort(indices_sorted)
        impact_arr = impact_arr[indices_sorted]
        beta_arr = beta_arr[indices_sorted]

        # Insert top boundary layer: a_top = 2*a[0] - a[1], β_top = 0
        impact_arr = np.concatenate(([2.0 * impact_arr[0] - impact_arr[1]], impact_arr))
        beta_arr = np.concatenate(([beta_arr[0]], beta_arr))
        beta_arr[0] = 0.0

    else:
        # Multiple columns of observations
        indices_sorted = np.argsort(-impact_arr, axis=0)
        indices_back = np.argsort(indices_sorted, axis=0)
        impact_arr = impact_arr[indices_sorted, np.arange(impact_arr.shape[1])]
        beta_arr = beta_arr[indices_sorted, np.arange(impact_arr.shape[1])]

        # Insert top boundary layer: a_top = 2*a[0] - a[1], β_top = 0
        impact_arr = np.concatenate(([2.0 * impact_arr[0] - impact_arr[1]], impact_arr))
        beta_arr = np.concatenate(([beta_arr[0]], beta_arr))
        beta_arr[0] = 0.0

    # Method 1: Higher precision in tests for float32 vs float64
    if method_matrix == 1:
        n_arr = np.zeros(beta_arr.size)
        n_arr[0] = 1.0
        r_t_arr = np.zeros(beta_arr.size)
        r_t_arr[0] = impact_arr[0]

        beta_arr2d = np.zeros((beta_arr.size, beta_arr.size))
        theta_a_arr2d = np.zeros((beta_arr.size, beta_arr.size))
        theta_b_arr2d = np.zeros((beta_arr.size, beta_arr.size))

        for i in range(1, n_arr.size):
            # theta_a for layers 0..i-1
            theta_a_arr2d[i, :i] = np.arcsin(np.clip(impact_arr[i] / n_arr[:i] / r_t_arr[:i], -1.0, 1.0))

            if i > 1:
                # theta_b for layers 0..i-2
                theta_b_arr2d[i, :i - 1] = np.arcsin(np.clip(impact_arr[i] / n_arr[1:i] / r_t_arr[:i - 1], -1.0, 1.0))
                beta_arr2d[i, :i - 1] = theta_a_arr2d[i, :i - 1] - theta_b_arr2d[i, :i - 1]

            # Last element in row i from half-beta closure
            beta_arr2d[i, i - 1] = 0.5 * beta_arr[i] - np.nansum(beta_arr2d[i])

            # Back out theta_b_last and update state
            theta_b_arr2d[i, i - 1] = theta_a_arr2d[i, i - 1] - beta_arr2d[i, i - 1]

            n_arr[i] = impact_arr[i] / (r_t_arr[i - 1] * np.sin(theta_b_arr2d[i, i - 1]))
            r_t_arr[i] = r_t_arr[i - 1] * np.sin(theta_b_arr2d[i, i - 1])

    # Method 2: Alternative approach, lower precision per original notes
    if method_matrix == 2:
        n_arr = np.zeros(beta_arr.size)
        n_arr[0] = 1.0
        r_t_arr = np.zeros(beta_arr.size)
        r_t_arr[0] = impact_arr[0]

        beta_arr2d = np.zeros((beta_arr.size, beta_arr.size))
        theta_a_arr2d = np.zeros((beta_arr.size, beta_arr.size))
        theta_b_arr2d = np.zeros((beta_arr.size, beta_arr.size))

        # Precompute theta_a purely from impact ratios, respecting lower-triangular structure
        impact_2d_i = impact_arr[0:impact_arr.size, np.newaxis]
        impact_2d_j = impact_arr[np.newaxis, 0:impact_arr.size]
        theta_a_arr2d_test = np.arcsin(np.clip(impact_2d_i / impact_2d_j, -1.0, 1.0))
        theta_a_arr2d_test[np.triu_indices_from(theta_a_arr2d_test, k=1)] = 0.0
        theta_a_arr2d = theta_a_arr2d_test

        for i in range(1, n_arr.size):
            if i > 1:
                theta_b_arr2d[i, :i - 1] = np.arcsin(np.clip(impact_arr[i] / n_arr[1:i] / r_t_arr[:i - 1], -1.0, 1.0))
                beta_arr2d[i, :i - 1] = theta_a_arr2d[i, :i - 1] - theta_b_arr2d[i, :i - 1]

            beta_arr2d[i, i - 1] = 0.5 * beta_arr[i] - np.nansum(beta_arr2d[i])
            theta_b_arr2d[i, i - 1] = theta_a_arr2d[i, i - 1] - beta_arr2d[i, i - 1]

            r_t_arr[i] = r_t_arr[i - 1] * np.sin(theta_b_arr2d[i, i - 1])
            n_arr[i] = impact_arr[i] / r_t_arr[i]

    # Method 3: Space-saving variant for multiple columns, equivalent logic to method 1
    if method_matrix == 3:
        n_arr = np.zeros(beta_arr.shape)
        n_arr[0] = 1.0
        r_t_arr = np.zeros(beta_arr.shape)
        r_t_arr[0] = impact_arr[0]

        for i in range(1, n_arr.shape[0]):
            beta_arr1d = np.zeros(beta_arr.shape)
            theta_a_arr1d = np.zeros(beta_arr.shape)
            theta_b_arr1d = np.zeros(beta_arr.shape)

            # theta_a for layers 0..i-1
            theta_a_arr1d[:i] = np.arcsin(np.clip(impact_arr[i] / n_arr[:i] / r_t_arr[:i], -1.0, 1.0))

            if i > 1:
                # theta_b for layers 0..i-2
                theta_b_arr1d[:i - 1] = np.arcsin(np.clip(impact_arr[i] / n_arr[1:i] / r_t_arr[:i - 1], -1.0, 1.0))
                beta_arr1d[:i - 1] = theta_a_arr1d[:i - 1] - theta_b_arr1d[:i - 1]

            # Last element closed by half-beta constraint
            beta_arr1d[i - 1] = 0.5 * beta_arr[i] - np.nansum(beta_arr1d, axis=0)
            theta_b_arr1d[i - 1] = theta_a_arr1d[i - 1] - beta_arr1d[i - 1]

            # Update state
            n_arr[i] = impact_arr[i] / (r_t_arr[i - 1] * np.sin(theta_b_arr1d[i - 1]))
            r_t_arr[i] = r_t_arr[i - 1] * np.sin(theta_b_arr1d[i - 1])

            if verbose == 1:
                if (i + 1) == int(n_arr.shape[0] * 0.2):
                    print("20 % of invert steps finished @", datetime.now())
                if (i + 1) == int(n_arr.shape[0] * 0.4):
                    print("40 % of invert steps finished @", datetime.now())
                if (i + 1) == int(n_arr.shape[0] * 0.6):
                    print("60 % of invert steps finished @", datetime.now())
                if (i + 1) == int(n_arr.shape[0] * 0.8):
                    print("80 % of invert steps finished @", datetime.now())
                if (i + 1) == int(n_arr.shape[0]):
                    print("100 % of invert steps finished @", datetime.now())

    # Restore original order and remove the inserted top layer
    if method_matrix != 3:
        return (r_t_arr[1:])[indices_back], (n_arr[1:])[indices_back]
    else:
        return (r_t_arr[1:])[indices_back, np.arange(beta_arr.shape[1])], (n_arr[1:])[
            indices_back, np.arange(beta_arr.shape[1])
        ]

"""
Copyright (c) 2026, Lawrence Livermore National Security, LLC.  All rights 
reserved.  LLNL-CODE-2019387
MIT License
This work was produced at the Lawrence Livermore National Laboratory (LLNL) 
under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department 
of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the 
operation of LLNL.  See license for disclaimers, notice of U.S. Government 
Rights and license terms and conditions.
@authors: Shuang Xu
"""