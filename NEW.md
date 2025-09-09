# RTX 3060 12GB Optimization Strategy

Maximize pages/second throughput on RTX 3060 12GB through strategic batch tuning and selective quantization.

## Executive Summary

**Best Strategy**: Two workers with tuned batches + selective INT8 quantization on transformers
- **Recognition is the bottleneck** (text-heavy pages) - invest VRAM budget there first
- **Two workers usually beat single worker** on RTX 3060 due to pipeline overlap between pages
- **Quantization's main value**: Frees VRAM to enable larger recognition batches + dual workers

## Optimized Configuration

### Baseline (Deploy Immediately)
```bash
# Two workers with conservative batch sizes (~11GB total)
export DETECTOR_BATCH_SIZE=6        # ~2.6GB (up from QUANT.md's 4)
export LAYOUT_BATCH_SIZE=8          # ~1.8GB  
export RECOGNITION_BATCH_SIZE=32    # ~1.3GB
export TABLE_REC_BATCH_SIZE=8       # ~1.2GB

# Memory management for dual workers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=2

marker /input/folder --workers 2
```

### Optimized with Quantization
```bash
# Enable transformer quantization first
pip install torchao  # Required for GPU quantization

# Reinvest saved VRAM into larger recognition batches
export DETECTOR_BATCH_SIZE=6        # Keep same (conv-heavy, leave FP16)
export LAYOUT_BATCH_SIZE=12         # +50% (transformer, benefits from quantization)
export RECOGNITION_BATCH_SIZE=64    # +100% (main bottleneck, biggest gain)
export TABLE_REC_BATCH_SIZE=8       # Keep same (modest size)

# Memory and performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=2

marker /input/folder --workers 2 --quantize-transformers
```

## Performance Optimizations

### GPU Optimizations (Add to marker codebase)

```python
# marker/utils/gpu_optimize.py
import torch

def configure_rtx3060_optimizations():
    """Configure RTX 3060-specific optimizations"""
    
    # Enable TF32 for FP32 fallbacks (Ampere feature)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Disable cudnn benchmark for variable input shapes (dual workers)
    torch.backends.cudnn.benchmark = False
    
    # Enable channels_last for conv models (detection)
    return {
        "use_channels_last_conv": True,
        "use_tf32": True,
        "stable_cudnn": True
    }

def optimize_model_for_rtx3060(model, model_type="transformer"):
    """Apply RTX 3060-specific model optimizations"""
    if model_type == "conv":
        # Detection model: use channels_last memory format
        model = model.to(memory_format=torch.channels_last)
    
    return model
```

### Integration with Model Loading

```python
# marker/models.py (enhanced)
def create_model_dict(device=None, dtype=None, quantize_transformers=False, optimize_rtx3060=True):
    foundation_predictor = FoundationPredictor(device=device, dtype=dtype)
    
    models = {
        "foundation_model": foundation_predictor,
        "layout_model": LayoutPredictor(device=device, dtype=dtype),
        "recognition_model": RecognitionPredictor(foundation_predictor),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype)
    }
    
    if optimize_rtx3060 and device and device.startswith("cuda"):
        from marker.utils.gpu_optimize import configure_rtx3060_optimizations, optimize_model_for_rtx3060
        
        # Apply GPU optimizations
        config = configure_rtx3060_optimizations()
        
        # Optimize conv models (detection)
        models["detection_model"] = optimize_model_for_rtx3060(models["detection_model"], "conv")
    
    if quantize_transformers:
        from marker.quantize import quantize_transformers
        # Quantize transformer-heavy components only
        models["layout_model"] = quantize_transformers(models["layout_model"], device)
        models["recognition_model"] = quantize_transformers(models["recognition_model"], device)
    
    return models
```

## Expected Performance

| Configuration | VRAM Usage | Speed | Pages/Sec | Stability |
|--------------|------------|-------|-----------|-----------|
| Default (1 worker) | ~5GB | 1.0x | Baseline | ✓ |
| Tuned (2 workers) | ~11GB | 1.6x | +60% | ✓ |
| + Quantization | ~9GB | 1.8x | +80% | ✓ |
| + GPU opts | ~9GB | 2.0x | +100% | ✓ |

## Implementation Steps

### Step 1: Batch Tuning (Immediate)
```bash
# Test baseline configuration
DETECTOR_BATCH_SIZE=6 LAYOUT_BATCH_SIZE=8 RECOGNITION_BATCH_SIZE=32 TABLE_REC_BATCH_SIZE=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
marker /test/docs --workers 2
```

### Step 2: Add Quantization Support
```python
# Add marker/quantize.py from QUANT.md
# Add quantization flags to CLI
# Test quantized configuration
```

### Step 3: GPU-Specific Optimizations
```python
# Add marker/utils/gpu_optimize.py
# Integrate with model loading
# Enable channels_last for conv models
```

### Step 4: Validation
```bash
# Test accuracy on regression suite
# Compare FP16 baseline vs INT8 quantized
# Ensure WER/Table F1 delta <0.5%
```

## Calibration Guide

### Finding Optimal Recognition Batch Size
The key tuning parameter for throughput:

```bash
# Start with baseline
export RECOGNITION_BATCH_SIZE=32

# With quantization, test scaling up:
# - 64 (target for RTX 3060)
# - 80 (if stable)
# - 96 (max before diminishing returns)

# Monitor VRAM usage:
watch nvidia-smi

# If OOM on complex PDFs, reduce by 16:
export RECOGNITION_BATCH_SIZE=48
```

### Memory Pressure Indicators
- **OOM errors**: Reduce recognition batch by 16-32
- **Fragmentation warnings**: Add `torch.cuda.empty_cache()` between docs
- **Slow startup**: Reduce detector batch (cuDNN workspace bloat)

## Alternative: Single Worker Configuration

If dual workers prove unstable, maximize single worker:

```bash
# Single worker with larger batches
export DETECTOR_BATCH_SIZE=8
export LAYOUT_BATCH_SIZE=16  
export RECOGNITION_BATCH_SIZE=128  # Quantized transformer can handle more
export TABLE_REC_BATCH_SIZE=8

marker /input/folder --workers 1 --quantize-transformers
```

**Trade-off**: Better per-batch GPU utilization but less pipeline parallelism. Usually slower than dual workers on RTX 3060.

## Production Deployment

### Environment Setup
```bash
# Install dependencies
pip install torchao

# Set environment variables
export DETECTOR_BATCH_SIZE=6
export LAYOUT_BATCH_SIZE=12
export RECOGNITION_BATCH_SIZE=64
export TABLE_REC_BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=2

# Optional: periodic cleanup for long runs
export MARKER_CLEANUP_INTERVAL=50  # restart workers every 50 docs
```

### Monitoring
```bash
# GPU utilization timeline
nsys profile --trace=cuda,cudnn marker /docs --workers 2 --quantize-transformers

# Memory usage over time  
while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 5; done

# Throughput measurement
time marker /test_docs --workers 2 --quantize-transformers
```

## Implementation Priority

1. **Phase 1 (Week 1)**: Batch tuning - immediate 60% throughput gain
2. **Phase 2 (Week 2)**: Add quantization support - additional 20% gain 
3. **Phase 3 (Week 3)**: GPU optimizations - final 20% gain
4. **Phase 4 (Week 4)**: Validation and fine-tuning

## Technical Notes

**Why Two Workers Beat Single Worker on RTX 3060:**
- RTX 3060 has modest SM count - small batches underutilize GPU
- Sequential pipeline (detection→layout→recognition→table) has idle gaps
- Two workers interleave stages, keeping GPU busier
- Pipeline overlap compensates for smaller per-worker batches

**Why Target Recognition First:**
- Recognition processes many text crops per page (bottleneck)
- Detection is one backbone pass per page (usually not limiting)
- Transformer quantization more effective than conv quantization
- Larger recognition batches = better GEMM efficiency on Tensor Cores

**Quantization ROI on RTX 3060:**
- Weight-only INT8: ~2x memory reduction for model weights
- Frees VRAM for larger activation batches (key benefit)
- Kernel speed-ups modest but batching gains significant
- Essential for fitting two workers + large recognition batches

## Fallback Plans

**If Quantization Causes Accuracy Issues:**
```bash
# Disable quantization, reduce batch sizes
export RECOGNITION_BATCH_SIZE=32  # Back to baseline
marker /docs --workers 2  # No quantization flag
```

**If Memory Fragmentation Causes OOMs:**
```bash
# Add periodic cleanup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# Reduce batch sizes slightly
export RECOGNITION_BATCH_SIZE=48
```

**If Single Worker Proves Faster:**
```bash
# Switch to single worker with large batches
export RECOGNITION_BATCH_SIZE=128
marker /docs --workers 1 --quantize-transformers
```

---

**Bottom Line**: Start with tuned dual workers, add quantization to reinvest VRAM into recognition batches, then apply RTX 3060-specific GPU optimizations. This should deliver ~2x throughput improvement while staying within 12GB VRAM constraints.