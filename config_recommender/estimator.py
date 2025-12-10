"""BentoML llm-optimizer integration."""

import subprocess
from typing import Optional
from .models import PerformanceAnalysis
from .parser import parse_bentoml_output


class BentoMLEstimator:
    """Wrapper for BentoML llm-optimizer estimate command."""
    
    def __init__(self, input_len: int = 2000, output_len: int = 256):
        """
        Initialize the estimator.
        
        Args:
            input_len: Default input token length
            output_len: Default output token length
        """
        self.input_len = input_len
        self.output_len = output_len
    
    def estimate(self, model: str, gpu: str, num_gpus: int = 1) -> Optional[PerformanceAnalysis]:
        """
        Run llm-optimizer estimate for a model/GPU combination.
        
        Args:
            model: Model name (e.g., "Qwen/Qwen2.5-7B")
            gpu: GPU type (e.g., "H100", "H200", "L40")
            num_gpus: Number of GPUs to use
            
        Returns:
            PerformanceAnalysis object or None if estimation fails
        """
        cmd = [
            "llm-optimizer",
            "estimate",
            "--model", model,
            "--input-len", str(self.input_len),
            "--output-len", str(self.output_len),
            "--gpu", gpu,
            "--num-gpus", str(num_gpus),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Error running llm-optimizer for {model} on {gpu}:")
                print(result.stderr)
                return None
            
            # Parse the output
            return parse_bentoml_output(
                result.stdout,
                model=model,
                gpu=gpu,
                num_gpus=num_gpus,
                input_len=self.input_len,
                output_len=self.output_len
            )
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running llm-optimizer for {model} on {gpu}")
            return None
        except FileNotFoundError:
            print("Error: llm-optimizer command not found. Please install llm-optimizer:")
            print("  pip install llm-optimizer")
            return None
        except Exception as e:
            print(f"Unexpected error running llm-optimizer: {e}")
            return None
