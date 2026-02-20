"""
================================================================================
ATOM CORE v1.0
The Limit of Continuity - Nearest Neighbor Geometric Inference
================================================================================

Project Lead: Alex Kinetic
AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
License: MIT
Version: v1.0
Date: 2026-02-20

================================================================================
PHILOSOPHY
================================================================================

ATOM represents the limit case where data density approaches continuity.

When points are so numerous that neighbors are arbitrarily close, geometric
structures (simplex, politopos) become computationally wasteful. The nearest
neighbor IS the answer.

PRINCIPLES:
- Deterministic: Same query → Same neighbor → Same result
- Honest: Returns distance as "truthfulness metric"
- Efficient: O(log N) search via spatial indexing
- Transparent: No black-box interpolation, pure identity

WHEN TO USE ATOM:
- Datasets with N >> 10^6 points
- High density (avg neighbor distance << epsilon)
- Real-time inference requirements
- Memory-constrained environments (no model compression needed)

================================================================================
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Union, Tuple, Optional
import warnings


class AtomCore:
    """
    Atom Core - Nearest Neighbor Inference Engine
    
    The simplest and fastest SLRM motor. Uses spatial indexing (KDTree)
    to find the closest point in the dataset and returns its Y value.
    
    Complexity:
    - Training: O(N log N) to build tree
    - Inference: O(log N) per query
    - Memory: O(N·D) to store tree
    
    Parameters:
    -----------
    dimensions : int
        Number of input dimensions
    leafsize : int, default=16
        KDTree leaf size (tradeoff: build time vs query time)
        - Smaller: Faster queries, slower build
        - Larger: Faster build, slower queries
    verbose : bool, default=True
        Print diagnostic information
    
    Attributes:
    -----------
    tree : cKDTree
        Spatial index for fast neighbor search
    X : np.ndarray
        Feature matrix (N, D)
    Y : np.ndarray
        Target values (N,)
    N : int
        Number of training points
    bounds_min : np.ndarray
        Minimum bounds per dimension
    bounds_max : np.ndarray
        Maximum bounds per dimension
    """
    
    def __init__(self, dimensions: int, leafsize: int = 16, verbose: bool = True):
        self.d = dimensions
        self.leafsize = leafsize
        self.verbose = verbose
        
        # Data storage
        self.tree = None
        self.X = None  # Feature matrix
        self.Y = None  # Target values
        self.N = 0
        
        # Metadata
        self.bounds_min = None
        self.bounds_max = None
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """
        Build spatial index from dataset.
        
        Parameters:
        -----------
        data : np.ndarray, shape (N, D+1)
            Training data where:
            - Columns 0 to D-1: Input coordinates
            - Column D: Target values
        
        Raises:
        -------
        ValueError
            If data has wrong shape or insufficient points
        """
        data = np.array(data, dtype=np.float64)
        
        # Validation
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        if data.shape[1] != self.d + 1:
            raise ValueError(
                f"Data must have {self.d + 1} columns (D={self.d} features + 1 target), "
                f"got {data.shape[1]}"
            )
        
        # Remove NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            warnings.warn(f"Removed {n_invalid} rows with NaN values")
            data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data points after removing NaN")
        
        # Store data
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.N = len(data)
        
        # Build KDTree
        # Parallel build is automatic in scipy
        self.tree = cKDTree(self.X, leafsize=self.leafsize)
        
        # Compute bounds for metadata
        self.bounds_min = np.min(self.X, axis=0)
        self.bounds_max = np.max(self.X, axis=0)
        
        self.fitted = True
        
        if self.verbose:
            print(f"✓ Atom Core v1.0: Indexed {self.N:,} points in {self.d}D")
            print(f"  Bounds: [{self.bounds_min.min():.4f}, {self.bounds_max.max():.4f}]")
            print(f"  KDTree leaf size: {self.leafsize}")
    
    def predict(self, query_points: Union[list, np.ndarray],
                return_distance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict values using nearest neighbor search.
        
        Parameters:
        -----------
        query_points : array-like, shape (M, D) or (D,)
            Query points to predict. Can be:
            - Single point: shape (D,)
            - Multiple points: shape (M, D)
        return_distance : bool, default=False
            If True, also return distances to nearest neighbors
        
        Returns:
        --------
        predictions : np.ndarray, shape (M,)
            Predicted Y values
        distances : np.ndarray, shape (M,) [optional]
            Euclidean distances to nearest neighbors (if return_distance=True)
        
        Raises:
        -------
        RuntimeError
            If model not fitted
        ValueError
            If query_points has wrong shape
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Handle single point
        query_points = np.atleast_2d(query_points)
        
        # Validate shape
        if query_points.shape[1] != self.d:
            raise ValueError(
                f"Query points must have {self.d} columns, "
                f"got {query_points.shape[1]}"
            )
        
        # Find nearest neighbors
        # k=1: single nearest neighbor per query
        # workers=-1: use all available CPU cores for parallelization
        distances, indices = self.tree.query(query_points, k=1, workers=-1)
        
        # Get Y values of nearest neighbors
        predictions = self.Y[indices]
        
        if return_distance:
            return predictions, distances
        return predictions
    
    def predict_with_confidence(self, query_points: Union[list, np.ndarray],
                                max_distance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence metric based on distance to nearest neighbor.
        
        The confidence score represents the "truthfulness" of the prediction:
        - High confidence: Query is close to training data (reliable)
        - Low confidence: Query is far from training data (less reliable)
        
        Parameters:
        -----------
        query_points : array-like, shape (M, D) or (D,)
            Query points to predict
        max_distance : float, optional
            Maximum acceptable distance. If None, estimated from data as
            3× the median nearest-neighbor distance.
            Points beyond max_distance get confidence approaching 0.
        
        Returns:
        --------
        predictions : np.ndarray, shape (M,)
            Predicted Y values
        confidence : np.ndarray, shape (M,)
            Confidence scores in [0, 1] where:
            - 1.0 = exact match (distance ≈ 0)
            - 0.5 = distance = max_distance / 3
            - 0.0 = distance >> max_distance
        
        Notes:
        ------
        Confidence is computed as: exp(-distance / (max_distance / 3))
        This gives exponential decay with distance.
        """
        # Get predictions and distances
        predictions, distances = self.predict(query_points, return_distance=True)
        
        if max_distance is None:
            # Auto-estimate max_distance from data
            # Sample 1000 points and compute their nearest-neighbor distances
            sample_size = min(1000, self.N)
            sample_indices = np.random.choice(self.N, sample_size, replace=False)
            sample_points = self.X[sample_indices]
            
            # Query with k=2 to get distance to nearest OTHER point (skip self)
            sample_dists, _ = self.tree.query(sample_points, k=2, workers=-1)
            median_nn_dist = np.median(sample_dists[:, 1])
            
            # Use 3× median as threshold
            max_distance = median_nn_dist * 3
        
        # Confidence: exponential decay
        # exp(-d / (max_d / 3)) gives:
        # - d=0: confidence=1.0
        # - d=max_d/3: confidence=0.37
        # - d=max_d: confidence=0.05
        confidence = np.exp(-distances / (max_distance / 3))
        confidence = np.clip(confidence, 0, 1)
        
        return predictions, confidence
    
    def evaluate(self, test_data: np.ndarray) -> dict:
        """
        Evaluate performance on test data.
        
        Parameters:
        -----------
        test_data : np.ndarray, shape (M, D+1)
            Test data with same format as training data
        
        Returns:
        --------
        metrics : dict
            Dictionary containing:
            - 'MSE': Mean Squared Error
            - 'MAE': Mean Absolute Error
            - 'RMSE': Root Mean Squared Error
            - 'inference_time': Total inference time (seconds)
            - 'time_per_point': Average time per query (milliseconds)
            - 'n_points': Number of test points
        """
        import time
        
        test_data = np.array(test_data, dtype=np.float64)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        
        # Time inference
        start = time.perf_counter()
        Y_pred = self.predict(X_test)
        elapsed = time.perf_counter() - start
        
        # Compute metrics
        errors = Y_test - Y_pred
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'inference_time': elapsed,
            'time_per_point': elapsed / len(X_test) * 1000,  # Convert to ms
            'n_points': len(X_test)
        }


# ==========================================
# QUICK START EXAMPLE
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ATOM CORE v1.0 - QUICK START EXAMPLE")
    print("="*70)
    
    # Generate high-density dataset
    np.random.seed(42)
    N, D = 100_000, 10
    
    print(f"\nGenerating dataset: {N:,} points in {D}D...")
    X = np.random.uniform(0, 10, (N, D))
    Y = np.sum(X**2, axis=1)  # Target function: sum of squares
    data = np.c_[X, Y]
    
    # Create and fit Atom Core
    print(f"\n{'='*70}")
    print("TRAINING")
    print("="*70)
    atom = AtomCore(dimensions=D, verbose=True)
    atom.fit(data)
    
    # Generate test data
    print(f"\n{'='*70}")
    print("EVALUATION")
    print("="*70)
    
    n_test = 1000
    X_test = np.random.uniform(0, 10, (n_test, D))
    Y_test = np.sum(X_test**2, axis=1)
    test_data = np.c_[X_test, Y_test]
    
    # Evaluate
    metrics = atom.evaluate(test_data)
    
    print(f"\nMetrics on {metrics['n_points']:,} test points:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  Inference time: {metrics['inference_time']*1000:.2f} ms total")
    print(f"  Time per point: {metrics['time_per_point']:.4f} ms")
    
    # Test confidence scores
    print(f"\n{'='*70}")
    print("CONFIDENCE TEST")
    print("="*70)
    
    print(f"\nPredicting with confidence scores on first 5 test points:")
    query = X_test[:5]
    predictions, confidence = atom.predict_with_confidence(query)
    
    for i, (pred, conf, true) in enumerate(zip(predictions, confidence, Y_test[:5])):
        error = abs(pred - true)
        print(f"  Point {i+1}: Y_pred={pred:.4f}, Y_true={true:.4f}, "
              f"Error={error:.4f}, Confidence={conf:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ QUICK START COMPLETE")
    print("="*70 + "\n")
   
