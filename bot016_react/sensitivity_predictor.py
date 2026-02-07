"""
Binary Options Sensitivity Predictor
Predicts PUT and CALL sensitivities based on distance, time to expiry, and volatility
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import KDTree


@dataclass
class BinKey:
    """Represents a bin configuration"""
    distance_range: Tuple[float, float]
    time_range: Tuple[int, int]
    vol_range: Tuple[float, float]
    label: str
    
    def get_center(self) -> Tuple[float, float, float]:
        """Get the center point of this bin"""
        dist_center = (self.distance_range[0] + self.distance_range[1]) / 2
        # Handle infinity in distance
        if dist_center == float('inf'):
            dist_center = self.distance_range[0] + 500  # Arbitrary large value
        
        time_center = (self.time_range[0] + self.time_range[1]) / 2
        vol_center = (self.vol_range[0] + self.vol_range[1]) / 2
        # Handle infinity in volatility
        if vol_center == float('inf'):
            vol_center = self.vol_range[0] + 120  # Arbitrary large value
        
        return (dist_center, time_center, vol_center)


class SensitivityPredictor:
    """
    Predicts binary option sensitivities using multiple interpolation strategies
    """
    
    def __init__(self, data_file: str):
        """Initialize predictor with data file"""
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        
        self.bins_data = self.raw_data['bins']
        self.bin_definitions = self._create_bin_definitions()
        
        # Prepare interpolation data
        self._prepare_interpolation_data()
        
        print(f"Loaded {len(self.bins_data)} bins with {self.raw_data['total_measurements']} measurements")
    
    def _create_bin_definitions(self) -> Dict[str, BinKey]:
        """Create bin definitions from bin labels"""
        distance_bins = [
            (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
            (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
            (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
            (1280, float('inf'), "1280+")
        ]
        time_bins = [
            (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
            (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
            (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
            (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
            (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
            (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
            (2, 5, "5s-2s"), (0, 2, "last-2s")
        ]
        vol_bins = [
            (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
            (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
            (120, float('inf'), "120+"),  # Handle both 120-240 and 120+
            (240, float('inf'), "240+")
        ]
        
        # Create lookup dictionaries
        dist_dict = {label: (low, high) for low, high, label in distance_bins}
        time_dict = {label: (low, high) for low, high, label in time_bins}
        vol_dict = {label: (low, high) for low, high, label in vol_bins}
        
        # Parse all bin keys
        bin_defs = {}
        for bin_label in self.bins_data.keys():
            parts = bin_label.split('|')
            if len(parts) == 3:
                dist_label, time_label, vol_label = parts
                bin_defs[bin_label] = BinKey(
                    distance_range=dist_dict[dist_label],
                    time_range=time_dict[time_label],
                    vol_range=vol_dict[vol_label],
                    label=bin_label
                )
        
        return bin_defs
    
    def _prepare_interpolation_data(self):
        """Prepare data structures for interpolation"""
        points = []
        put_sensitivities = []
        call_sensitivities = []
        weights = []  # Based on sample count
        
        for bin_label, bin_def in self.bin_definitions.items():
            bin_data = self.bins_data[bin_label]
            
            if bin_data['count'] > 0:
                center = bin_def.get_center()
                points.append(center)
                
                # Use median for more robust predictions
                put_sensitivities.append(bin_data['put_sensitivity']['median'])
                call_sensitivities.append(bin_data['call_sensitivity']['median'])
                
                # Weight by sample count (with log to avoid over-weighting high counts)
                weights.append(np.log1p(bin_data['count']))
        
        self.points = np.array(points)
        self.put_values = np.array(put_sensitivities)
        self.call_values = np.array(call_sensitivities)
        self.weights = np.array(weights)
        
        # Create interpolators
        self._create_interpolators()
        
        # Create KDTree for nearest neighbor search
        self.kdtree = KDTree(self.points)
        
        print(f"Prepared {len(self.points)} data points for interpolation")
    
    def _create_interpolators(self):
        """Create various interpolation models"""
        # Linear interpolator (works well within convex hull)
        self.linear_interp_put = LinearNDInterpolator(self.points, self.put_values)
        self.linear_interp_call = LinearNDInterpolator(self.points, self.call_values)
        
        # Nearest neighbor fallback (for extrapolation)
        self.nearest_interp_put = NearestNDInterpolator(self.points, self.put_values)
        self.nearest_interp_call = NearestNDInterpolator(self.points, self.call_values)
    
    def predict(self, distance: float, time_to_expiry: int, volatility: float, 
                method: str = 'adaptive') -> Dict:
        """
        Predict sensitivities for given market conditions
        
        Args:
            distance: Distance from strike price (in USD)
            time_to_expiry: Time to expiry in seconds
            volatility: Implied volatility (annualized %)
            method: 'adaptive', 'linear', 'nearest', 'weighted_knn', or 'ensemble'
        
        Returns:
            Dictionary with predictions and metadata
        """
        query_point = np.array([distance, time_to_expiry, volatility])
        
        if method == 'adaptive':
            return self._predict_adaptive(query_point)
        elif method == 'linear':
            return self._predict_linear(query_point)
        elif method == 'nearest':
            return self._predict_nearest(query_point)
        elif method == 'weighted_knn':
            return self._predict_weighted_knn(query_point, k=10)
        elif method == 'ensemble':
            return self._predict_ensemble(query_point)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_adaptive(self, query_point: np.ndarray) -> Dict:
        """
        Adaptive prediction: use linear interpolation if within convex hull,
        otherwise use weighted k-NN
        """
        # Try linear interpolation
        put_linear = self.linear_interp_put(query_point[0], query_point[1], query_point[2])
        call_linear = self.linear_interp_call(query_point[0], query_point[1], query_point[2])
        
        # Check if we got valid results (NaN means outside convex hull)
        if not np.isnan(put_linear) and not np.isnan(call_linear):
            # Find nearest neighbors for confidence estimate
            distances, indices = self.kdtree.query(query_point, k=5)
            avg_distance = np.mean(distances)
            
            return {
                'put_sensitivity': float(put_linear),
                'call_sensitivity': float(call_linear),
                'method': 'linear_interpolation',
                'confidence': self._compute_confidence(avg_distance, len(indices)),
                'nearest_distance': float(distances[0]),
                'query_point': {
                    'distance': float(query_point[0]),
                    'time_to_expiry': int(query_point[1]),
                    'volatility': float(query_point[2])
                }
            }
        else:
            # Outside convex hull, use weighted k-NN
            return self._predict_weighted_knn(query_point, k=10)
    
    def _predict_linear(self, query_point: np.ndarray) -> Dict:
        """Linear interpolation with nearest neighbor fallback"""
        put_val = self.linear_interp_put(query_point[0], query_point[1], query_point[2])
        call_val = self.linear_interp_call(query_point[0], query_point[1], query_point[2])
        
        if np.isnan(put_val):
            put_val = self.nearest_interp_put(query_point[0], query_point[1], query_point[2])
            call_val = self.nearest_interp_call(query_point[0], query_point[1], query_point[2])
            method = 'nearest_neighbor_fallback'
        else:
            method = 'linear_interpolation'
        
        return {
            'put_sensitivity': float(put_val),
            'call_sensitivity': float(call_val),
            'method': method,
            'query_point': {
                'distance': float(query_point[0]),
                'time_to_expiry': int(query_point[1]),
                'volatility': float(query_point[2])
            }
        }
    
    def _predict_nearest(self, query_point: np.ndarray) -> Dict:
        """Simple nearest neighbor prediction"""
        distance, index = self.kdtree.query(query_point, k=1)
        
        return {
            'put_sensitivity': float(self.put_values[index]),
            'call_sensitivity': float(self.call_values[index]),
            'method': 'nearest_neighbor',
            'nearest_distance': float(distance),
            'nearest_point': {
                'distance': float(self.points[index][0]),
                'time_to_expiry': int(self.points[index][1]),
                'volatility': float(self.points[index][2])
            },
            'query_point': {
                'distance': float(query_point[0]),
                'time_to_expiry': int(query_point[1]),
                'volatility': float(query_point[2])
            }
        }
    
    def _predict_weighted_knn(self, query_point: np.ndarray, k: int = 10) -> Dict:
        """
        Weighted k-nearest neighbors prediction
        Weights are based on inverse distance and sample count
        """
        distances, indices = self.kdtree.query(query_point, k=min(k, len(self.points)))
        
        # Compute weights: inverse distance * sample weight
        # Add small epsilon to avoid division by zero
        distance_weights = 1.0 / (distances + 0.01)
        sample_weights = self.weights[indices]
        combined_weights = distance_weights * sample_weights
        combined_weights /= combined_weights.sum()  # Normalize
        
        # Weighted average
        put_pred = np.sum(self.put_values[indices] * combined_weights)
        call_pred = np.sum(self.call_values[indices] * combined_weights)
        
        # Compute prediction variance as confidence metric
        put_variance = np.sum(combined_weights * (self.put_values[indices] - put_pred)**2)
        call_variance = np.sum(combined_weights * (self.call_values[indices] - call_pred)**2)
        
        return {
            'put_sensitivity': float(put_pred),
            'call_sensitivity': float(call_pred),
            'put_std': float(np.sqrt(put_variance)),
            'call_std': float(np.sqrt(call_variance)),
            'method': f'weighted_knn_k{k}',
            'confidence': self._compute_confidence(np.mean(distances), k),
            'nearest_distance': float(distances[0]),
            'avg_neighbor_distance': float(np.mean(distances)),
            'query_point': {
                'distance': float(query_point[0]),
                'time_to_expiry': int(query_point[1]),
                'volatility': float(query_point[2])
            }
        }
    
    def _predict_ensemble(self, query_point: np.ndarray) -> Dict:
        """
        Ensemble prediction: average of multiple methods
        """
        linear_pred = self._predict_linear(query_point)
        knn_pred = self._predict_weighted_knn(query_point, k=10)
        
        # Average the predictions
        put_ensemble = (linear_pred['put_sensitivity'] + knn_pred['put_sensitivity']) / 2
        call_ensemble = (linear_pred['call_sensitivity'] + knn_pred['call_sensitivity']) / 2
        
        return {
            'put_sensitivity': float(put_ensemble),
            'call_sensitivity': float(call_ensemble),
            'method': 'ensemble',
            'individual_predictions': {
                'linear': {
                    'put': linear_pred['put_sensitivity'],
                    'call': linear_pred['call_sensitivity']
                },
                'knn': {
                    'put': knn_pred['put_sensitivity'],
                    'call': knn_pred['call_sensitivity'],
                    'std': {
                        'put': knn_pred.get('put_std', 0),
                        'call': knn_pred.get('call_std', 0)
                    }
                }
            },
            'confidence': knn_pred.get('confidence', 0.5),
            'query_point': {
                'distance': float(query_point[0]),
                'time_to_expiry': int(query_point[1]),
                'volatility': float(query_point[2])
            }
        }
    
    def _compute_confidence(self, avg_distance: float, n_neighbors: int) -> float:
        """
        Compute confidence score based on distance to neighbors
        Returns value between 0 and 1
        """
        # Distance normalization factors (tuned empirically)
        distance_factor = np.exp(-avg_distance / 100)
        neighbor_factor = min(n_neighbors / 10, 1.0)
        
        return float(distance_factor * neighbor_factor)
    
    def find_nearest_bin(self, distance: float, time_to_expiry: int, 
                        volatility: float) -> Dict:
        """Find the actual bin closest to the query point"""
        query_point = np.array([distance, time_to_expiry, volatility])
        distance_val, index = self.kdtree.query(query_point, k=1)
        
        nearest_point = self.points[index]
        
        # Find the bin label
        for bin_label, bin_def in self.bin_definitions.items():
            center = bin_def.get_center()
            if np.allclose(center, nearest_point):
                bin_data = self.bins_data[bin_label]
                return {
                    'bin_label': bin_label,
                    'bin_center': {
                        'distance': center[0],
                        'time_to_expiry': center[1],
                        'volatility': center[2]
                    },
                    'distance_to_query': float(distance_val),
                    'sample_count': bin_data['count'],
                    'put_sensitivity': bin_data['put_sensitivity'],
                    'call_sensitivity': bin_data['call_sensitivity']
                }
        
        return None
    
    def analyze_sensitivity_surface(self, time_to_expiry: int, volatility: float,
                                   distance_range: Tuple[float, float] = (0, 100),
                                   n_points: int = 50) -> Dict:
        """
        Analyze how sensitivity changes with distance at fixed time and volatility
        """
        distances = np.linspace(distance_range[0], distance_range[1], n_points)
        
        put_sensitivities = []
        call_sensitivities = []
        
        for dist in distances:
            pred = self.predict(dist, time_to_expiry, volatility, method='adaptive')
            put_sensitivities.append(pred['put_sensitivity'])
            call_sensitivities.append(pred['call_sensitivity'])
        
        return {
            'distances': distances.tolist(),
            'put_sensitivities': put_sensitivities,
            'call_sensitivities': call_sensitivities,
            'fixed_params': {
                'time_to_expiry': time_to_expiry,
                'volatility': volatility
            }
        }


def demo_predictor():
    """Demonstration of the predictor"""
    predictor = SensitivityPredictor('/mnt/user-data/uploads/sensitivity_transformed.json')
    
    # Test scenarios
    test_cases = [
        {"distance": 10, "time": 300, "volatility": 50, "desc": "Near strike, 5min, moderate vol"},
        {"distance": 50, "time": 600, "volatility": 30, "desc": "Mid distance, 10min, low vol"},
        {"distance": 100, "time": 120, "volatility": 100, "desc": "Far OTM, 2min, high vol"},
        {"distance": 2, "time": 30, "volatility": 40, "desc": "Very near strike, 30s, moderate vol"},
        {"distance": 200, "time": 800, "volatility": 200, "desc": "Very far, 13min, extreme vol"},
    ]
    
    print("\n" + "="*80)
    print("SENSITIVITY PREDICTIONS - VARIOUS METHODS")
    print("="*80)
    
    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"Scenario: {case['desc']}")
        print(f"Distance: ${case['distance']} | Time: {case['time']}s | Vol: {case['volatility']}%")
        print(f"{'='*80}")
        
        for method in ['adaptive', 'weighted_knn', 'ensemble']:
            pred = predictor.predict(case['distance'], case['time'], case['volatility'], method=method)
            
            print(f"\nMethod: {method.upper()}")
            print(f"  PUT sensitivity:  {pred['put_sensitivity']:+.6f}")
            print(f"  CALL sensitivity: {pred['call_sensitivity']:+.6f}")
            
            if 'confidence' in pred:
                print(f"  Confidence: {pred['confidence']:.3f}")
            if 'put_std' in pred:
                print(f"  Uncertainty: PUT ±{pred['put_std']:.6f}, CALL ±{pred['call_std']:.6f}")
            if 'nearest_distance' in pred:
                print(f"  Nearest data point: {pred['nearest_distance']:.2f} units away")
        
        # Show nearest actual bin
        nearest_bin = predictor.find_nearest_bin(case['distance'], case['time'], case['volatility'])
        if nearest_bin:
            print(f"\nNearest actual bin: {nearest_bin['bin_label']}")
            print(f"  Samples: {nearest_bin['sample_count']}")
            print(f"  PUT median: {nearest_bin['put_sensitivity']['median']:+.6f}")
            print(f"  CALL median: {nearest_bin['call_sensitivity']['median']:+.6f}")
    
    return predictor


if __name__ == "__main__":
    predictor = demo_predictor()
