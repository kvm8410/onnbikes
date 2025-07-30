# 📥 Download Instructions - Improved NSFW Censoring System

## Quick Start

### 1. Download the Files
Download these essential files to get started:

**Required Files:**
- `improved_nsfw_censor.py` - Main censoring system
- `requirements.txt` - Dependencies list
- `setup.py` - Automated setup script

**Optional Files:**
- `usage_examples.py` - Comprehensive usage examples
- `comparison_demo.py` - Shows improvements over original code
- `README_NSFW_CENSOR.md` - Detailed documentation

### 2. Installation Methods

#### Method A: Automatic Setup (Recommended)
```bash
# Run the setup script (will install all dependencies automatically)
python setup.py
```

#### Method B: Manual Installation
```bash
# Install dependencies manually
pip install -r requirements.txt

# Test the installation
python improved_nsfw_censor.py
```

### 3. Quick Usage Test

```python
from improved_nsfw_censor import NSFWCensor

# Initialize the censor
censor = NSFWCensor()

# Process an image
censor.process_image("your_image.jpg", "censored_output.jpg", method="blur")
```

## File Descriptions

| File | Purpose | Required |
|------|---------|----------|
| `improved_nsfw_censor.py` | Main censoring system with all fixes | ✅ Required |
| `requirements.txt` | Python dependencies | ✅ Required |
| `setup.py` | Automated installation script | 🔧 Recommended |
| `usage_examples.py` | Code examples for various scenarios | 📚 Optional |
| `README_NSFW_CENSOR.md` | Complete documentation | 📖 Optional |
| `comparison_demo.py` | Shows differences from original | 🔍 Optional |

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (8GB+ recommended)
- **Storage**: 2GB+ free space (for model downloads)
- **GPU**: Optional but recommended (CUDA or Apple Silicon)

## First Run Notes

⚠️ **Important**: The first run will download AI models (~1-2GB total):
- ViT NSFW classifier (~500MB)
- NudeNet detection model (~200MB)

This is normal and only happens once. Subsequent runs will be much faster.

## Available Censoring Methods

- `"blur"` - Gaussian blur (recommended)
- `"pixelate"` - Pixelation effect
- `"black_bar"` - Black rectangles
- `"crop"` - Smart cropping

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **Model download fails**: Check internet connection and disk space
3. **GPU issues**: System will automatically fallback to CPU
4. **Memory errors**: Close other applications or use smaller images

### Performance Tips:

- Use GPU when available for 5-10x speed improvement
- Process images in batches for efficiency
- Resize very large images before processing

## Support

If you encounter issues:
1. Check the `README_NSFW_CENSOR.md` for detailed documentation
2. Run `comparison_demo.py` to see the improvements made
3. Try the examples in `usage_examples.py`

---

🚀 **Ready to use!** The improved system provides much more accurate NSFW detection and censoring compared to the original code.