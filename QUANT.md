# Practical Memory Optimization for Marker & Surya

Fit two workers on 12GB GPU through batch tuning + selective quantization.

## Problem

- Marker uses 3.5-5GB VRAM per worker
- Two workers = 7-10GB + Surya batches = OOM on 12GB GPU
- Need reliable way to run dual workers

## Solution

**Phase 1**: Batch size tuning (immediate fix)  
**Phase 2**: GPU-appropriate quantization where it helps

## Phase 1: Batch Size Tuning (Deploy Today)

First lever: reduce Surya batch sizes to fit two workers in 12GB.

```bash
# Conservative batch sizes for dual workers on 12GB
export DETECTOR_BATCH_SIZE=4      # ~1.8GB (down from 36)
export LAYOUT_BATCH_SIZE=8        # ~1.8GB (down from 32) 
export RECOGNITION_BATCH_SIZE=32  # ~1.3GB (down from 256)
export TABLE_REC_BATCH_SIZE=8     # ~1.2GB (down from default)

# Run two workers safely
marker /data/in --workers 2
```

**Total usage**: ~6GB (Surya) + 2×2.5GB (Marker) = ~11GB on 12GB GPU

## Phase 2: Selective GPU Quantization 

Target transformer-heavy components with TorchAO weight-only quantization.

### Step 1: Add TorchAO quantization

```python
# marker/quantize.py
import torch

try:
    from torchao.quantization import quantize_, int8_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

def quantize_transformers(model, device):
    """Apply INT8 weight-only quantization to Linear layers on GPU"""
    if device == "cpu":
        # CPU path: use PyTorch dynamic quantization
        from torch.quantization import quantize_dynamic
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    elif device.startswith("cuda") and TORCHAO_AVAILABLE:
        # GPU path: use TorchAO weight-only quantization
        quantize_(model, int8_weight_only())
        return model
    
    # Fallback: no quantization
    return model
```

### Step 2: Apply only to transformer components

```python
# marker/models.py  
def create_model_dict(device=None, dtype=None, quantize_transformers=False):
    foundation_predictor = FoundationPredictor(device=device, dtype=dtype)
    
    models = {
        "foundation_model": foundation_predictor,
        "layout_model": LayoutPredictor(device=device, dtype=dtype),
        "recognition_model": RecognitionPredictor(foundation_predictor),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype)
    }
    
    if quantize_transformers:
        from marker.quantize import quantize_transformers
        # Only quantize transformer-heavy components
        models["layout_model"] = quantize_transformers(models["layout_model"], device)
        models["recognition_model"] = quantize_transformers(models["recognition_model"], device)
        # Leave detection and table-rec in FP16 (conv-heavy, modest size)
    
    return models
```

### Step 3: Component-specific CLI flags

```python
# marker/scripts/convert.py
@click.option("--quantize-transformers", is_flag=True, help="Quantize layout/recognition models")
def main(quantize_transformers, **kwargs):
    models = create_model_dict(quantize_transformers=quantize_transformers)
    # ... rest unchanged
```

## Usage

```bash
# Phase 1: Immediate fix with batch tuning
DETECTOR_BATCH_SIZE=4 LAYOUT_BATCH_SIZE=8 RECOGNITION_BATCH_SIZE=32 \
marker /input/folder --workers 2

# Phase 2: Add transformer quantization (requires pip install torchao)
marker_single document.pdf --quantize-transformers
```

## Expected Results

| Approach | VRAM Usage | Speed | 12GB GPU Workers |
|----------|-----------|-------|-----------------|
| Default | 10-15GB total | 1x | 1 worker max |
| Batch tuning | ~11GB total | 0.8x | 2 workers ✓ |
| + TorchAO quantization | ~8GB total | 0.7x | 2 workers ✓ |

## Test Plan

Regression test on text-heavy PDFs, math, tables, scans:
- Metrics: page-level WER, table cell F1, LaTeX match rate
- Compare FP16 baseline vs TorchAO INT8 weight-only
- Abort if accuracy drops >0.5% on WER or table F1

## Implementation Notes

**Phase 1 (batch tuning)**:
- Uses existing Surya environment variables
- No code changes needed
- Deploy immediately

**Phase 2 (selective quantization)**:
- TorchAO for GPU Linear layers only
- PyTorch dynamic quantization for CPU fallback
- Only targets transformer components (layout/recognition)
- Leaves conv-heavy detection in FP16
- Requires `pip install torchao` for GPU path

## Risks

- CUDA quantization requires TorchAO (PyTorch eager INT8 is CPU-only)
- Accuracy may drop on tiny glyphs and low-contrast scans
- Detection gets no VRAM reduction without ONNX/TensorRT (future work)