import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize, brentq
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class BetaParameters:
    """Parameters for negative and positive class Beta distributions."""
    neg_alpha: float
    neg_beta: float
    pos_alpha: float
    pos_beta: float
    
    def validate(self) -> bool:
        """Ensure parameters create valid Beta distributions with desired skewness."""
        right_skew_valid = self.neg_alpha < 1 and self.neg_beta > 1
        left_skew_valid = self.pos_alpha > 1 and self.pos_beta < 1
        return right_skew_valid and left_skew_valid
    
    def __str__(self) -> str:
        return (f"Negative (right-skew): α={self.neg_alpha:.4f}, β={self.neg_beta:.4f}\n"
                f"Positive (left-skew): α={self.pos_alpha:.4f}, β={self.pos_beta:.4f}")

@dataclass
class PerformanceTargets:
    """Target performance metrics for the model."""
    ppv: float
    sensitivity: float
    
    def validate(self) -> bool:
        """Validate that targets are in valid range."""
        return 0 <= self.ppv <= 1 and 0 <= self.sensitivity <= 1

class AnalyticalBetaOptimizer:
    """
    Analytical optimizer for Beta distribution parameters given target performance metrics.
    
    Uses mathematical relationships between distribution parameters and performance
    metrics to solve directly rather than using expensive grid search.
    """
    
    def __init__(self, population_size: int, prevalence: float, top_k: int):
        """
        Initialize the optimizer.
        
        Args:
            population_size: Total population size
            prevalence: Prevalence of positive events (0 < prevalence < 1)
            top_k: Number of top-scoring individuals to select as positive predictions
        """
        self.population_size = population_size
        self.prevalence = prevalence
        self.top_k = top_k
        self.threshold_percentile = 1 - (top_k / population_size)
        
        # Validate inputs
        if not (0 < prevalence < 1):
            raise ValueError(f"Prevalence must be between 0 and 1, got {prevalence}")
        if not (0 < top_k < population_size):
            raise ValueError(f"top_k must be between 0 and population_size, got {top_k}")
    
    def calculate_mixture_threshold(self, neg_alpha: float, neg_beta: float, 
                                  pos_alpha: float, pos_beta: float) -> float:
        """
        Calculate the score threshold for top-k selection in mixture distribution.
        
        The threshold is the (1 - k/n) quantile of the mixture distribution:
        F_mixture(threshold) = prevalence * F_pos(threshold) + (1-prevalence) * F_neg(threshold)
        """
        def mixture_cdf(x):
            pos_cdf = beta_dist.cdf(x, pos_alpha, pos_beta)
            neg_cdf = beta_dist.cdf(x, neg_alpha, neg_beta)
            return self.prevalence * pos_cdf + (1 - self.prevalence) * neg_cdf
        
        def find_threshold(x):
            return mixture_cdf(x) - self.threshold_percentile
        
        try:
            # Find threshold where mixture CDF equals target percentile
            threshold = brentq(find_threshold, 0.001, 0.999)
            return threshold
        except ValueError:
            # Fallback to approximation if root finding fails
            return self._approximate_threshold(neg_alpha, neg_beta, pos_alpha, pos_beta)
    
    def _approximate_threshold(self, neg_alpha: float, neg_beta: float, 
                             pos_alpha: float, pos_beta: float) -> float:
        """Approximate threshold using weighted quantiles."""
        neg_quantile = beta_dist.ppf(self.threshold_percentile, neg_alpha, neg_beta)
        pos_quantile = beta_dist.ppf(self.threshold_percentile, pos_alpha, pos_beta)
        
        # Weighted average based on prevalence
        return ((1 - self.prevalence) * neg_quantile + self.prevalence * pos_quantile)
    
    def calculate_performance_metrics(self, neg_alpha: float, neg_beta: float,
                                    pos_alpha: float, pos_beta: float) -> Tuple[float, float]:
        """
        Calculate PPV and Sensitivity for given Beta parameters.
        
        Returns:
            Tuple of (PPV, Sensitivity)
        """
        # Calculate threshold
        threshold = self.calculate_mixture_threshold(neg_alpha, neg_beta, pos_alpha, pos_beta)
        
        # Calculate sensitivity: P(Score >= threshold | Positive)
        sensitivity = 1 - beta_dist.cdf(threshold, pos_alpha, pos_beta)
        
        # Calculate P(Score >= threshold) for entire population
        prob_above_threshold = (
            self.prevalence * sensitivity + 
            (1 - self.prevalence) * (1 - beta_dist.cdf(threshold, neg_alpha, neg_beta))
        )
        
        # Calculate PPV: P(Positive | Score >= threshold)
        if prob_above_threshold > 1e-10:  # Avoid division by zero
            ppv = (self.prevalence * sensitivity) / prob_above_threshold
        else:
            ppv = 0.0
        
        return ppv, sensitivity
    
    def solve_parameters(self, targets: PerformanceTargets) -> BetaParameters:
        """
        Solve for Beta distribution parameters that achieve target performance metrics.
        
        Uses constrained optimization with analytical performance metric calculations.
        
        Args:
            targets: Target PPV and Sensitivity values
            
        Returns:
            Optimized Beta parameters
        """
        if not targets.validate():
            raise ValueError(f"Invalid targets: {targets}")
        
        logging.info(f"Solving for targets: PPV={targets.ppv:.3f}, Sensitivity={targets.sensitivity:.3f}")
        
        def objective(params):
            """Objective function: minimize distance from target metrics."""
            neg_alpha, neg_beta, pos_alpha, pos_beta = params
            
            try:
                ppv, sensitivity = self.calculate_performance_metrics(
                    neg_alpha, neg_beta, pos_alpha, pos_beta
                )
                
                # L2 distance from targets with weights
                ppv_error = (ppv - targets.ppv) ** 2
                sens_error = (sensitivity - targets.sensitivity) ** 2
                
                return ppv_error + sens_error
                
            except Exception as e:
                # Return large penalty for invalid parameters
                return 1e6
        
        # Parameter bounds ensuring valid Beta distributions with correct skewness
        bounds = [
            (0.01, 0.99),   # neg_alpha < 1 for right skew
            (1.01, 10.0),   # neg_beta > 1 for right skew
            (1.01, 10.0),   # pos_alpha > 1 for left skew
            (0.01, 0.99),   # pos_beta < 1 for left skew
        ]
        
        # Multiple initial guesses to avoid local minima
        initial_guesses = [
            [0.1, 2.0, 2.0, 0.3],
            [0.3, 3.0, 3.0, 0.1],
            [0.5, 1.5, 1.5, 0.5],
            [0.2, 4.0, 4.0, 0.2],
        ]
        
        best_result = None
        best_objective = float('inf')
        
        for x0 in initial_guesses:
            try:
                result = minimize(
                    objective,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    
            except Exception as e:
                logging.warning(f"Optimization failed for initial guess {x0}: {e}")
                continue
        
        if best_result is None or not best_result.success:
            raise RuntimeError("Optimization failed to converge from all initial guesses")
        
        params = BetaParameters(*best_result.x)
        
        if not params.validate():
            raise RuntimeError("Optimized parameters do not satisfy Beta distribution constraints")
        
        # Verify achieved performance
        achieved_ppv, achieved_sens = self.calculate_performance_metrics(*best_result.x)
        
        logging.info(f"Optimization converged in {best_result.nit} iterations")
        logging.info(f"Achieved metrics: PPV={achieved_ppv:.4f}, Sensitivity={achieved_sens:.4f}")
        logging.info(f"Target error: {best_objective:.6f}")
        
        return params

class MLModelSimulator:
    """Complete ML model simulator using analytical Beta distribution optimization."""
    
    def __init__(self, population_size: int, prevalence: float, top_k: int):
        self.population_size = population_size
        self.prevalence = prevalence
        self.top_k = top_k
        self.optimizer = AnalyticalBetaOptimizer(population_size, prevalence, top_k)
        self.beta_params: Optional[BetaParameters] = None
    
    def calibrate(self, targets: PerformanceTargets) -> BetaParameters:
        """Calibrate the model to achieve target performance metrics."""
        self.beta_params = self.optimizer.solve_parameters(targets)
        return self.beta_params
    
    def generate_scores(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate risk scores using calibrated Beta distributions.
        
        Returns:
            Tuple of (scores, true_labels)
        """
        if self.beta_params is None:
            raise RuntimeError("Model must be calibrated before generating scores")
        
        if n_samples is None:
            n_samples = self.population_size
        
        # Generate true labels
        n_positive = int(n_samples * self.prevalence)
        n_negative = n_samples - n_positive
        
        true_labels = np.concatenate([
            np.ones(n_positive, dtype=int),
            np.zeros(n_negative, dtype=int)
        ])
        
        # Shuffle labels
        np.random.shuffle(true_labels)
        
        # Generate scores based on labels
        scores = np.zeros(n_samples)
        pos_mask = true_labels == 1
        neg_mask = true_labels == 0
        
        scores[pos_mask] = np.random.beta(
            self.beta_params.pos_alpha, 
            self.beta_params.pos_beta, 
            size=np.sum(pos_mask)
        )
        scores[neg_mask] = np.random.beta(
            self.beta_params.neg_alpha, 
            self.beta_params.neg_beta, 
            size=np.sum(neg_mask)
        )
        
        return scores, true_labels
    
    def validate_performance(self, scores: np.ndarray, true_labels: np.ndarray) -> dict:
        """Validate that generated scores achieve target performance."""
        # Sort by scores and take top k
        sorted_indices = np.argsort(scores)[::-1]
        top_predictions = sorted_indices[:self.top_k]
        
        # Calculate confusion matrix
        predictions = np.zeros_like(true_labels)
        predictions[top_predictions] = 1
        
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        
        # Calculate metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        return {
            'ppv': ppv,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    def plot_distributions(self):
        """Plot the calibrated Beta distributions."""
        if self.beta_params is None:
            raise RuntimeError("Model must be calibrated before plotting")
        
        x = np.linspace(0, 1, 1000)
        
        neg_pdf = beta_dist.pdf(x, self.beta_params.neg_alpha, self.beta_params.neg_beta)
        pos_pdf = beta_dist.pdf(x, self.beta_params.pos_alpha, self.beta_params.pos_beta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, neg_pdf, label='Negative Events (Right-skewed)', color='blue', linewidth=2)
        plt.plot(x, pos_pdf, label='Positive Events (Left-skewed)', color='red', linewidth=2)
        
        plt.xlabel('Risk Score')
        plt.ylabel('Probability Density')
        plt.title('Calibrated Beta Distributions for Risk Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of the analytical Beta optimizer."""
    # Configuration
    population_size = 10000
    prevalence = 0.05  # 5% positive events
    top_k = 250        # Select top 250 as high risk
    
    # Target performance metrics
    targets = PerformanceTargets(ppv=0.20, sensitivity=0.75)
    
    print("=== Analytical Beta Distribution Optimizer ===")
    print(f"Population size: {population_size}")
    print(f"Prevalence: {prevalence:.1%}")
    print(f"Top-k selection: {top_k}")
    print(f"Target PPV: {targets.ppv:.3f}")
    print(f"Target Sensitivity: {targets.sensitivity:.3f}")
    print()
    
    # Create and calibrate simulator
    simulator = MLModelSimulator(population_size, prevalence, top_k)
    
    print("Calibrating model...")
    beta_params = simulator.calibrate(targets)
    
    print("\nOptimal Beta Parameters:")
    print(beta_params)
    
    # Generate scores and validate
    print("\nGenerating scores and validating performance...")
    scores, true_labels = simulator.generate_scores()
    performance = simulator.validate_performance(scores, true_labels)
    
    print(f"\nAchieved Performance:")
    print(f"PPV: {performance['ppv']:.4f}")
    print(f"Sensitivity: {performance['sensitivity']:.4f}")
    print(f"Specificity: {performance['specificity']:.4f}")
    print(f"F1 Score: {performance['f1_score']:.4f}")
    
    # Plot distributions
    simulator.plot_distributions()

if __name__ == "__main__":
    main()