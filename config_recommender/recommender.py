"""GPU recommendation engine."""

from typing import List, Optional
from .estimator import BentoMLEstimator
from .models import PerformanceAnalysis, GPURecommendation


class GPURecommender:
    """
    GPU recommendation engine for model inference.
    
    Analyzes multiple models across different GPU types and provides
    recommendations for the best GPU for each model based on performance metrics.
    """
    
    def __init__(self, input_len: int = 2000, output_len: int = 256):
        """
        Initialize the recommender.
        
        Args:
            input_len: Input token length for benchmarking
            output_len: Output token length for benchmarking
        """
        self.estimator = BentoMLEstimator(input_len=input_len, output_len=output_len)
        self.analyses: List[PerformanceAnalysis] = []
    
    def analyze_model_gpu_pairs(
        self,
        models: List[str],
        gpus: List[str],
        num_gpus: int = 1
    ) -> List[PerformanceAnalysis]:
        """
        Analyze all combinations of models and GPUs.
        
        Args:
            models: List of model names
            gpus: List of GPU types
            num_gpus: Number of GPUs to use for each configuration
            
        Returns:
            List of PerformanceAnalysis objects
        """
        analyses = []
        total = len(models) * len(gpus)
        current = 0
        
        for model in models:
            for gpu in gpus:
                current += 1
                print(f"\nAnalyzing {model} on {gpu} ({current}/{total})...")
                
                analysis = self.estimator.estimate(model, gpu, num_gpus)
                if analysis:
                    analyses.append(analysis)
                    print(f"✓ Analysis complete for {model} on {gpu}")
                else:
                    print(f"✗ Failed to analyze {model} on {gpu}")
        
        self.analyses = analyses
        return analyses
    
    def recommend_gpus(
        self,
        models: List[str],
        gpus: List[str],
        num_gpus: int = 1,
        metric: str = "throughput"
    ) -> List[GPURecommendation]:
        """
        Recommend the best GPU for each model.
        
        Args:
            models: List of model names
            gpus: List of GPU types to consider
            num_gpus: Number of GPUs to use
            metric: Metric to optimize for ("throughput" or "latency")
            
        Returns:
            List of GPURecommendation objects
        """
        # Analyze all model/GPU pairs
        analyses = self.analyze_model_gpu_pairs(models, gpus, num_gpus)
        
        # Group analyses by model
        model_analyses = {}
        for analysis in analyses:
            model_name = analysis.config.model
            if model_name not in model_analyses:
                model_analyses[model_name] = []
            model_analyses[model_name].append(analysis)
        
        # Generate recommendations
        recommendations = []
        for model in models:
            if model not in model_analyses:
                print(f"Warning: No analyses found for {model}")
                continue
            
            best_gpu = self._select_best_gpu(model_analyses[model], metric)
            if best_gpu:
                recommendations.append(GPURecommendation(
                    model=model,
                    recommended_gpu=best_gpu,
                    all_analyses=model_analyses[model]
                ))
        
        return recommendations
    
    def _select_best_gpu(
        self,
        analyses: List[PerformanceAnalysis],
        metric: str = "throughput"
    ) -> Optional[str]:
        """
        Select the best GPU based on performance metrics.
        
        Args:
            analyses: List of analyses for different GPUs
            metric: Metric to optimize for
            
        Returns:
            GPU type name or None
        """
        if not analyses:
            return None
        
        best_analysis = None
        best_score = float('-inf') if metric == "throughput" else float('inf')
        
        for analysis in analyses:
            if not analysis.metrics:
                continue
            
            if metric == "throughput":
                # Higher throughput is better
                score = analysis.metrics.output_tokens_per_s or 0
                if score > best_score:
                    best_score = score
                    best_analysis = analysis
            elif metric == "latency":
                # Lower latency is better
                score = analysis.metrics.e2e_s or float('inf')
                if score < best_score:
                    best_score = score
                    best_analysis = analysis
        
        return best_analysis.config.gpu if best_analysis else None
    
    def print_recommendations(self, recommendations: List[GPURecommendation]) -> None:
        """Print recommendations in a formatted way."""
        print("\n" + "=" * 80)
        print("GPU RECOMMENDATIONS")
        print("=" * 80)
        
        for rec in recommendations:
            print(f"\n{rec}")
            print("-" * 80)
