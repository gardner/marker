# Model Quantization Strategy for Marker & Surya

This document outlines a comprehensive plan for implementing model quantization in both the marker PDF converter and the underlying Surya OCR/layout analysis library. Quantization can reduce memory usage by 50-75% and improve inference speed by 2-10x while maintaining acceptable accuracy.

## Current State Analysis

### Marker Repository Status
- **Models Used**: FoundationPredictor, LayoutPredictor, RecognitionPredictor, TableRecPredictor, DetectionPredictor, OCRErrorPredictor
- **Existing Optimization**: Mixed precision with `torch.bfloat16` on CUDA, `torch.float32` on CPU
- **Memory Usage**: 3.5-5GB VRAM per worker at default settings
- **Configuration**: Supports device/dtype parameters in `create_model_dict()`

### Surya Repository Status  
- **Model Architecture**: PyTorch-based vision transformers for OCR, layout analysis, table recognition
- **Memory Footprint**: 4-20GB VRAM at default batch sizes per component
- **Optimization**: Mixed precision (float16) enabled by default
- **Limitation**: No quantization support currently implemented

### Performance Baseline
- **Current VRAM**: 3.5-5GB per marker worker, 16GB+ per surya component at defaults
- **Target Reduction**: 50-75% memory usage through quantization
- **Expected Speedup**: 2-4x inference speed improvement

## Implementation Phases

### Phase 1: INT8 Post-Training Quantization (Immediate Impact)

**Timeline**: 2-4 weeks  
**Memory Reduction**: ~75%  
**Implementation Complexity**: Low

#### Core Quantization Module

```python
# marker/utils/quantization.py
import torch
from torch.quantization import quantize_dynamic
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """
    Quantization utilities for marker/surya models.
    
    AIDEV-NOTE: INT8 quantization reduces memory by ~75% with minimal accuracy loss
    for inference workloads. Focus on Linear layers which dominate model parameters.
    """
    
    @staticmethod
    def quantize_model(model: torch.nn.Module, quantization_type: str = "int8") -> torch.nn.Module:
        """Apply post-training quantization to reduce memory usage."""
        if quantization_type == "int8":
            logger.info(f"Applying INT8 quantization to {model.__class__.__name__}")
            return quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif quantization_type == "int4":
            # Placeholder for INT4 implementation
            logger.warning("INT4 quantization not yet implemented, falling back to INT8")
            return ModelQuantizer.quantize_model(model, "int8")
        
        logger.warning(f"Unknown quantization type {quantization_type}, returning original model")
        return model
    
    @staticmethod
    def estimate_memory_reduction(model: torch.nn.Module, quantization_type: str = "int8") -> Dict[str, float]:
        """Estimate memory reduction from quantization."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Rough estimates based on quantization type
        reductions = {
            "int8": 0.75,  # 32-bit -> 8-bit
            "int4": 0.875, # 32-bit -> 4-bit
            "bfloat16": 0.5 # 32-bit -> 16-bit
        }
        
        reduction = reductions.get(quantization_type, 0)
        original_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
        reduced_mb = original_mb * (1 - reduction)
        
        return {
            "original_size_mb": original_mb,
            "quantized_size_mb": reduced_mb,
            "reduction_percentage": reduction * 100,
            "memory_saved_mb": original_mb - reduced_mb
        }
```

#### Enhanced Model Loading

```python
# marker/models.py (enhanced)
import os
from typing import Optional
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from marker.utils.quantization import ModelQuantizer
from marker.logger import get_logger

logger = get_logger()

def create_model_dict(
    device=None, 
    dtype=None, 
    quantize: bool = False,
    quantization_type: str = "int8"
) -> dict:
    """
    Create model dictionary with optional quantization support.
    
    Args:
        device: Target device (cuda/cpu/mps)
        dtype: Model precision (float32/float16/bfloat16)
        quantize: Enable model quantization for memory efficiency
        quantization_type: Type of quantization (int8/int4)
        
    Returns:
        Dictionary of loaded models with optional quantization applied
        
    AIDEV-NOTE: Quantization is applied after model loading to preserve
    compatibility with existing Surya predictor initialization.
    """
    foundation_predictor = FoundationPredictor(device=device, dtype=dtype)
    
    models = {
        "foundation_model": foundation_predictor,
        "layout_model": LayoutPredictor(device=device, dtype=dtype),
        "recognition_model": RecognitionPredictor(foundation_predictor),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype)
    }
    
    if quantize:
        logger.info(f"Applying {quantization_type} quantization to all models")
        quantizer = ModelQuantizer()
        quantized_models = {}
        
        for name, model in models.items():
            # Log memory reduction estimate
            stats = quantizer.estimate_memory_reduction(model, quantization_type)
            logger.info(f"{name}: {stats['original_size_mb']:.1f}MB -> {stats['quantized_size_mb']:.1f}MB "
                       f"({stats['reduction_percentage']:.1f}% reduction)")
            
            quantized_models[name] = quantizer.quantize_model(model, quantization_type)
        
        models = quantized_models
    
    return models
```

#### Configuration Integration

```python
# marker/settings.py (additions)
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Quantization settings
    ENABLE_MODEL_QUANTIZATION: bool = False
    QUANTIZATION_TYPE: str = "int8"  # Options: int8, int4, bfloat16
    QUANTIZATION_BACKEND: str = "pytorch"  # Options: pytorch, torchao
    
    @computed_field
    @property 
    def QUANTIZATION_CONFIG(self) -> dict:
        """Quantization configuration based on device and settings."""
        return {
            "enabled": self.ENABLE_MODEL_QUANTIZATION,
            "type": self.QUANTIZATION_TYPE,
            "backend": self.QUANTIZATION_BACKEND,
            "device_optimized": self.TORCH_DEVICE_MODEL == "cuda",
            "memory_constrained": self.default_gpu_vram <= 12  # Auto-enable for low VRAM
        }
    
    @computed_field
    @property
    def AUTO_QUANTIZE(self) -> bool:
        """Auto-enable quantization on memory-constrained systems."""
        return self.default_gpu_vram <= 12 and not self.ENABLE_MODEL_QUANTIZATION
```

#### CLI Integration

```python
# marker/scripts/convert.py (additions)
@click.option(
    "--quantize", 
    is_flag=True, 
    help="Enable model quantization to reduce memory usage"
)
@click.option(
    "--quantization-type", 
    type=click.Choice(["int8", "int4", "bfloat16"]),
    default="int8",
    help="Type of quantization to apply (default: int8)"
)
def main(quantize, quantization_type, **kwargs):
    """Enhanced main function with quantization support."""
    # ... existing code ...
    
    if quantize or settings.AUTO_QUANTIZE:
        logger.info(f"Quantization enabled: {quantization_type}")
        models = create_model_dict(
            device=device,
            dtype=dtype,
            quantize=True,
            quantization_type=quantization_type
        )
    else:
        models = create_model_dict(device=device, dtype=dtype)
    
    # ... rest of conversion logic ...
```

### Phase 2: TorchAO Integration (Advanced Optimization)

**Timeline**: 4-8 weeks  
**Memory Reduction**: Up to 87.5% with INT4  
**Implementation Complexity**: Medium

#### TorchAO Quantization Module

```python
# marker/utils/torchao_quantization.py
"""
Advanced quantization using TorchAO for maximum performance.

AIDEV-NOTE: TorchAO provides state-of-the-art quantization with torch.compile()
support for optimal inference performance. Requires torch >= 2.4.0.
"""

try:
    from torchao.quantization import (
        quantize_, int8_weight_only, int4_weight_only,
        int8_dynamic_activation_int8_weight, smooth_fq_recipe
    )
    import torch._inductor.config
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TorchAOQuantizer:
    """Advanced quantization using PyTorch's official TorchAO library."""
    
    def __init__(self):
        if not TORCHAO_AVAILABLE:
            raise ImportError(
                "TorchAO not available. Install with: pip install torchao\n"
                "Requires PyTorch >= 2.4.0"
            )
    
    @staticmethod
    def apply_int8_quantization(model: torch.nn.Module) -> torch.nn.Module:
        """Apply INT8 weight-only quantization with TorchAO."""
        logger.info("Applying TorchAO INT8 weight-only quantization")
        quantize_(model, int8_weight_only())
        return model
    
    @staticmethod  
    def apply_int4_quantization(model: torch.nn.Module) -> torch.nn.Module:
        """Apply INT4 weight-only quantization with TorchAO."""
        logger.info("Applying TorchAO INT4 weight-only quantization")
        quantize_(model, int4_weight_only())
        return model
    
    @staticmethod
    def apply_smooth_quantization(model: torch.nn.Module) -> torch.nn.Module:
        """Apply smooth quantization recipe for better accuracy."""
        logger.info("Applying TorchAO smooth quantization recipe")
        quantize_(model, smooth_fq_recipe.int8_weight_only_recipe())
        return model
    
    @staticmethod
    def optimize_for_inference(model: torch.nn.Module, sample_input: torch.Tensor) -> torch.nn.Module:
        """Compile model with optimizations for inference."""
        logger.info("Compiling model for optimized inference")
        
        # Configure inductor for inference
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        
        # Compile with full optimizations
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True)
        
        # Warm up compilation with sample input
        with torch.no_grad():
            _ = compiled_model(sample_input)
            
        return compiled_model
```

### Phase 3: Surya-Specific Optimizations 

**Timeline**: 6-12 weeks (requires coordination with Surya maintainers)  
**Implementation Complexity**: High

#### Predictor Base Class Enhancement

```python
# Proposed changes for surya repository
# surya/base_predictor.py (new file)

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class QuantizedPredictor(ABC):
    """
    Base class for quantized predictors in Surya.
    
    AIDEV-NOTE: Provides unified interface for quantization across all
    Surya model components while maintaining backward compatibility.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        quantize: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.dtype = dtype
        self.quantize = quantize
        self.quantization_config = quantization_config or {}
        
        self.model = self.load_model()
        
        if self.quantize:
            self.model = self.apply_quantization(self.model)
    
    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        """Load the base model architecture."""
        pass
    
    def apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply quantization to the loaded model."""
        quantization_type = self.quantization_config.get("type", "int8")
        backend = self.quantization_config.get("backend", "pytorch")
        
        if backend == "torchao" and TORCHAO_AVAILABLE:
            return self._apply_torchao_quantization(model, quantization_type)
        else:
            return self._apply_pytorch_quantization(model, quantization_type)
    
    def _apply_pytorch_quantization(self, model: torch.nn.Module, qtype: str) -> torch.nn.Module:
        """Apply PyTorch native quantization."""
        from marker.utils.quantization import ModelQuantizer
        return ModelQuantizer.quantize_model(model, qtype)
    
    def _apply_torchao_quantization(self, model: torch.nn.Module, qtype: str) -> torch.nn.Module:
        """Apply TorchAO quantization."""
        from marker.utils.torchao_quantization import TorchAOQuantizer
        quantizer = TorchAOQuantizer()
        
        if qtype == "int8":
            return quantizer.apply_int8_quantization(model)
        elif qtype == "int4":
            return quantizer.apply_int4_quantization(model)
        else:
            return quantizer.apply_smooth_quantization(model)
```

## Performance Benchmarks & Testing

### Accuracy Validation Framework

```python
# marker/benchmarks/quantization.py
"""
Quantization accuracy and performance benchmarking.

AIDEV-NOTE: Essential for validating that quantization doesn't degrade
document processing quality beyond acceptable thresholds.
"""

import time
import torch
from typing import Dict, List, Tuple
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.benchmarks.overall.overall import run_single_benchmark

class QuantizationBenchmark:
    """Benchmark quantization impact on accuracy and performance."""
    
    def __init__(self, test_documents: List[str]):
        self.test_documents = test_documents
        self.results = {}
    
    def benchmark_configuration(
        self, 
        config_name: str,
        quantize: bool = False,
        quantization_type: str = "int8"
    ) -> Dict[str, float]:
        """Benchmark a specific quantization configuration."""
        
        # Load models with configuration
        models = create_model_dict(
            quantize=quantize,
            quantization_type=quantization_type
        )
        
        converter = PdfConverter(artifact_dict=models)
        
        # Measure memory usage
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        
        # Process test documents
        processing_times = []
        for doc_path in self.test_documents:
            start_time = time.time()
            rendered = converter(doc_path)
            processing_times.append(time.time() - start_time)
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            "config": config_name,
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "memory_baseline_mb": start_memory / (1024 * 1024),
            "quantization": quantization_type if quantize else "none"
        }
    
    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run comprehensive quantization benchmark."""
        configs = [
            ("baseline", False, "none"),
            ("int8_pytorch", True, "int8"),
            ("int4_pytorch", True, "int4"),
        ]
        
        results = {}
        for config_name, quantize, qtype in configs:
            try:
                results[config_name] = self.benchmark_configuration(
                    config_name, quantize, qtype
                )
            except Exception as e:
                print(f"Failed to benchmark {config_name}: {e}")
                results[config_name] = {"error": str(e)}
        
        return results
```

### Expected Performance Improvements

| Configuration | Memory Reduction | Speed Improvement | Accuracy Impact |
|---------------|------------------|-------------------|-----------------|
| Baseline (FP32) | 0% | 1x | 100% |
| Mixed Precision (FP16) | ~50% | 1.5-2x | 99.8% |
| INT8 Quantization | ~75% | 2-4x | 99.5% |
| INT4 Quantization | ~87.5% | 3-6x | 98-99% |
| TorchAO + Compile | ~75% | 5-10x | 99.5% |

## Deployment Considerations

### Hardware Compatibility

**Recommended for Quantization:**
- **GPU**: RTX 3060 12GB+, RTX 4060 8GB+, A10/A100
- **CPU**: Recent Intel/AMD with AVX2 support
- **Memory**: System RAM >= 2x VRAM for model swapping

**Not Recommended:**
- Very old GPUs without Tensor Core support
- Systems with <8GB system RAM
- Apple Silicon (MPS) - limited quantization support

### Configuration Examples

```bash
# Memory-constrained deployment (8-12GB VRAM)
export ENABLE_MODEL_QUANTIZATION=true
export QUANTIZATION_TYPE=int8
marker_single document.pdf --quantize

# High-performance deployment (16GB+ VRAM)  
export QUANTIZATION_BACKEND=torchao
marker_single document.pdf --quantize --quantization-type int4

# Auto-optimization for unknown hardware
marker_single document.pdf  # Auto-detects and applies optimal settings
```

## Migration Strategy

### Backwards Compatibility
- All quantization features are **opt-in** by default
- Existing configurations continue working unchanged
- Environment variables provide non-breaking activation
- CLI flags are additive, not replacing existing options

### Rollout Plan
1. **Phase 1**: Implement basic INT8 quantization in marker
2. **Phase 2**: Add TorchAO integration and advanced options
3. **Phase 3**: Coordinate with Surya maintainers for upstream integration
4. **Phase 4**: Optimize for specific model architectures and use cases

### Testing Strategy
- Maintain comprehensive benchmark suite comparing quantized vs. full precision
- Test on diverse document types (scientific papers, forms, tables, handwriting)
- Validate across different hardware configurations
- Monitor community feedback and adjust defaults based on real-world usage

## Future Enhancements

### Dynamic Quantization
- Runtime quantization based on document complexity
- Adaptive precision scaling for accuracy-critical sections
- Memory pressure detection with automatic quantization adjustment

### Model Compression
- Knowledge distillation for smaller specialized models
- Pruning combined with quantization for maximum efficiency
- Custom quantization schemes optimized for document processing workflows

### Integration Opportunities
- Hugging Face Transformers quantization compatibility
- ONNX Runtime quantized model export
- Integration with cloud deployment platforms (Modal, AWS Lambda)

---

**Implementation Priority**: Phase 1 (INT8 quantization) provides the highest impact with lowest complexity and should be implemented first. This alone will solve most memory constraint issues while maintaining high accuracy and providing substantial performance improvements.

**AIDEV-NOTE**: This quantization strategy balances immediate practical benefits with long-term optimization potential, making it suitable for incremental implementation across both repositories while maintaining backward compatibility and user adoption.