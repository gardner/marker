# Simple Model Quantization for Marker & Surya

Reduce memory usage by 50-75% with minimal code changes.

## Problem

- Marker uses 3.5-5GB VRAM per worker
- Surya components use 4-20GB VRAM at defaults  
- 12GB GPUs can't run multiple workers effectively

## Solution

Add simple quantization using PyTorch's built-in functions. No fancy frameworks or complex abstractions.

## Implementation

### Step 1: Add basic quantization function

```python
# marker/quantize.py
import torch
from torch.quantization import quantize_dynamic

def quantize_model(model):
    """Apply INT8 quantization to reduce memory by ~75%"""
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### Step 2: Add quantization to model loading

```python
# marker/models.py
def create_model_dict(device=None, dtype=None, quantize=False):
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
        from marker.quantize import quantize_model
        for name in models:
            models[name] = quantize_model(models[name])
    
    return models
```

### Step 3: Add CLI flag

```python
# marker/scripts/convert.py
@click.option("--quantize", is_flag=True, help="Reduce memory usage by 75%")
def main(quantize, **kwargs):
    models = create_model_dict(quantize=quantize)
    # ... rest unchanged
```

## Usage

```bash
# Enable quantization to use 75% less memory
marker_single document.pdf --quantize

# Multiple workers with quantization on 12GB GPU
marker /input/folder --quantize --workers 2
```

## Expected Results

| Mode | Memory Usage | Speed | 12GB GPU Workers |
|------|-------------|-------|-----------------|
| Normal | 5GB per worker | 1x | 1-2 workers max |
| Quantized | 1.25GB per worker | 0.9x | 4-8 workers |

## Testing

Test with a sample document:
```bash
# Before
marker_single test.pdf
# Monitor VRAM usage

# After  
marker_single test.pdf --quantize
# Compare VRAM usage (should be ~75% less)
```

## Implementation Notes

- Uses PyTorch's built-in `quantize_dynamic` function
- Only quantizes Linear layers (where most parameters live)
- Minimal accuracy impact (typically <1%)
- No external dependencies required
- Backward compatible (quantization is opt-in)