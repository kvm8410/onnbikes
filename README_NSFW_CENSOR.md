# Improved NSFW Content Censoring System

This is an improved version of the NSFW content detection and censoring system that fixes several critical issues in the original implementation and provides more accurate and robust censoring capabilities.

## Key Improvements Over Original Code

### 1. **Correct NudeNet Label Matching**
- **Original Issue**: Used incorrect labels like `"EXPOSED_GENITALS"`, `"GENITALIA_EXPOSED"` that don't match NudeNet's actual output
- **Fix**: Updated to use correct NudeNet labels:
  - `"EXPOSED_GENITALIA_F"`, `"EXPOSED_GENITALIA_M"`
  - `"EXPOSED_BUTTOCKS"`, `"EXPOSED_BREAST_F"`
  - `"EXPOSED_ANUS"`, `"COVERED_GENITALIA_F"`, `"COVERED_GENITALIA_M"`

### 2. **Enhanced Detection Logic**
- **Original Issue**: Poor error handling and unsafe unpacking of detection results
- **Fix**: Robust detection parsing with proper validation and confidence filtering
- **Improvement**: Added separate confidence threshold for detections

### 3. **Multiple Censoring Methods**
- **Original Issue**: Limited to poor cropping strategy that often removed too much content
- **Fix**: Added 4 different censoring methods:
  - **Blur**: Gaussian blur on detected regions (recommended)
  - **Pixelate**: Pixelation effect on detected regions
  - **Black Bar**: Solid black rectangles over detected regions
  - **Smart Crop**: Intelligent cropping that preserves more content

### 4. **Better Image Format Handling**
- **Original Issue**: No RGB/BGR conversion consistency
- **Fix**: Proper color space conversion for both NudeNet (RGB) and OpenCV (BGR)

### 5. **Improved Device Support**
- **Original Issue**: Hardcoded MPS device usage that might not be available
- **Fix**: Automatic device detection with fallbacks (CUDA → MPS → CPU)

### 6. **Enhanced Error Handling and Logging**
- **Original Issue**: Minimal error handling and debugging information
- **Fix**: Comprehensive logging and error handling throughout the pipeline

### 7. **Object-Oriented Design**
- **Original Issue**: Procedural code that was hard to extend
- **Fix**: Clean class-based design for better maintainability and reusability

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
# CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for Apple Silicon:
pip install torch torchvision
```

## Basic Usage

### Simple Image Censoring

```python
from improved_nsfw_censor import NSFWCensor

# Initialize the censor
censor = NSFWCensor()

# Process a single image
censor.process_image("input.jpg", "censored_output.jpg", method="blur")
```

### Different Censoring Methods

```python
# Available methods: "blur", "pixelate", "black_bar", "crop"
methods = ["blur", "pixelate", "black_bar", "crop"]

for method in methods:
    output_path = f"output_{method}.jpg"
    censor.process_image("input.jpg", output_path, method=method)
```

### Video Frame Processing

```python
import cv2

censor = NSFWCensor()

# For video processing
cap = cv2.VideoCapture("input_video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    censored_frame = censor.process_video_frame(frame, method="blur")
    
    # Save or display the censored frame
    cv2.imshow("Censored", censored_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Configuration Options

You can adjust various parameters at the top of `improved_nsfw_censor.py`:

```python
PADDING_RATIO = 0.15          # Padding around detected regions (0.0-1.0)
NSFW_THRESHOLD = 0.6          # ViT model threshold for NSFW classification (0.0-1.0)
DETECTION_CONFIDENCE = 0.3    # NudeNet detection confidence threshold (0.0-1.0)
BLUR_INTENSITY = 51           # Blur kernel size (must be odd)
PIXELATE_RATIO = 0.1          # Pixelation ratio (lower = more pixelated)
```

## Censoring Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Blur** | Natural looking, preserves context | May still be recognizable | General use, social media |
| **Pixelate** | Classic censoring effect | Can look artificial | Traditional censoring needs |
| **Black Bar** | Complete coverage | Obvious censoring, loses context | Strong censoring requirements |
| **Smart Crop** | Preserves most content | May lose important parts | When content preservation is crucial |

## Performance Considerations

- **GPU Usage**: Automatically uses CUDA/MPS when available for faster processing
- **Batch Processing**: Process multiple images efficiently using the batch processing example
- **Video Processing**: For real-time video, consider processing every Nth frame for better performance

## Example Scripts

Run the included examples:

```bash
# Basic examples
python usage_examples.py

# Test the improved censor
python improved_nsfw_censor.py
```

## Technical Details

### Detection Pipeline

1. **ViT Classification**: First determines if image is NSFW using transformer model
2. **Region Detection**: If NSFW, uses NudeNet to find specific regions
3. **Region Filtering**: Filters detections by confidence and target labels
4. **Censoring Application**: Applies chosen censoring method to detected regions

### Model Information

- **ViT Model**: `Falconsai/nsfw_image_detection` - Binary NSFW classifier
- **NudeNet**: Object detection model specifically for anatomical regions
- **Supported Formats**: JPG, PNG, BMP, TIFF

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Ensure you have internet connection and sufficient disk space
   # Models will be downloaded automatically on first run
   ```

2. **CUDA Out of Memory**
   ```python
   # Use CPU instead
   device = torch.device("cpu")
   ```

3. **Poor Detection Results**
   ```python
   # Adjust thresholds
   NSFW_THRESHOLD = 0.4        # Lower = more sensitive
   DETECTION_CONFIDENCE = 0.2  # Lower = more detections
   ```

### Performance Optimization

- Use GPU when available
- Process images in batches for better efficiency
- For video, consider processing every 2nd or 3rd frame
- Resize very large images before processing

## License and Disclaimer

This tool is provided for content filtering and safety purposes. Please ensure compliance with local laws and platform policies when using automated content moderation tools.

## Contributing

To improve the system:

1. Test with diverse image sets
2. Adjust detection thresholds based on your use case
3. Add new censoring methods as needed
4. Optimize for your specific hardware setup

The modular design makes it easy to extend and customize for specific requirements.