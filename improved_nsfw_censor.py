import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from nudenet import NudeDetector
from transformers import AutoModelForImageClassification, AutoImageProcessor
import logging
from typing import List, Tuple, Optional
import os

# ------------------------- CONFIG -------------------------
PADDING_RATIO = 0.15  # Reduced padding for more precise detection
NSFW_THRESHOLD = 0.6  # Increased threshold for better accuracy
DETECTION_CONFIDENCE = 0.3  # Minimum confidence for NudeNet detections

# Correct NudeNet labels (these are the actual labels used by NudeNet)
TARGET_LABELS = {
    "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M", 
    "EXPOSED_BUTTOCKS", "EXPOSED_BREAST_F",
    "EXPOSED_ANUS", "COVERED_GENITALIA_F", "COVERED_GENITALIA_M"
}

# Censoring methods
CENSOR_METHOD = "blur"  # Options: "blur", "pixelate", "black_bar", "crop"
BLUR_INTENSITY = 51  # Must be odd number
PIXELATE_RATIO = 0.1  # Lower = more pixelated
# ----------------------------------------------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection with fallback
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
logger.info(f"Using device: {device}")

class NSFWCensor:
    def __init__(self):
        self.load_models()
    
    def load_models(self):
        """Load ViT model and NudeNet detector with error handling"""
        try:
            logger.info("Loading ViT NSFW Classifier...")
            self.vit_model = AutoModelForImageClassification.from_pretrained(
                "Falconsai/nsfw_image_detection"
            ).to(device)
            self.vit_processor = AutoImageProcessor.from_pretrained(
                "Falconsai/nsfw_image_detection"
            )
            logger.info("ViT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            raise
        
        try:
            logger.info("Loading NudeNet Detector...")
            self.detector = NudeDetector()
            logger.info("NudeNet loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NudeNet: {e}")
            raise
    
    def get_nsfw_score(self, img: np.ndarray) -> float:
        """Get NSFW probability score using ViT model"""
        try:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Process image
            inputs = self.vit_processor(images=pil_img, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                nsfw_score = scores[0][1].item()  # Index 1 is NSFW class
            
            return nsfw_score
        except Exception as e:
            logger.error(f"Error in NSFW scoring: {e}")
            return 0.0
    
    def get_detection_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """Get bounding boxes for NSFW regions using NudeNet"""
        try:
            # NudeNet expects RGB format
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector.detect(rgb_img)
            
            valid_boxes = []
            for detection in results:
                if len(detection) >= 6:
                    x1, y1, x2, y2, confidence, label = detection[:6]
                    
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Filter by confidence and target labels
                    if confidence > DETECTION_CONFIDENCE and label in TARGET_LABELS:
                        valid_boxes.append((x1, y1, x2, y2, label, confidence))
                        logger.info(f"Detected {label} with confidence {confidence:.3f} at ({x1},{y1},{x2},{y2})")
            
            return valid_boxes
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def apply_blur_censoring(self, img: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """Apply Gaussian blur to detected regions"""
        censored = img.copy()
        
        for x1, y1, x2, y2, label, confidence in boxes:
            # Add padding
            h, w = img.shape[:2]
            pad_x = int((x2 - x1) * PADDING_RATIO)
            pad_y = int((y2 - y1) * PADDING_RATIO)
            
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            # Extract region and apply blur
            region = censored[y1_pad:y2_pad, x1_pad:x2_pad]
            if region.size > 0:
                blurred_region = cv2.GaussianBlur(region, (BLUR_INTENSITY, BLUR_INTENSITY), 0)
                censored[y1_pad:y2_pad, x1_pad:x2_pad] = blurred_region
        
        return censored
    
    def apply_pixelate_censoring(self, img: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """Apply pixelation to detected regions"""
        censored = img.copy()
        
        for x1, y1, x2, y2, label, confidence in boxes:
            # Add padding
            h, w = img.shape[:2]
            pad_x = int((x2 - x1) * PADDING_RATIO)
            pad_y = int((y2 - y1) * PADDING_RATIO)
            
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            # Extract region
            region = censored[y1_pad:y2_pad, x1_pad:x2_pad]
            if region.size > 0:
                # Pixelate by downsampling and upsampling
                region_h, region_w = region.shape[:2]
                small_h = max(1, int(region_h * PIXELATE_RATIO))
                small_w = max(1, int(region_w * PIXELATE_RATIO))
                
                # Downsample
                small_region = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                # Upsample back
                pixelated_region = cv2.resize(small_region, (region_w, region_h), interpolation=cv2.INTER_NEAREST)
                
                censored[y1_pad:y2_pad, x1_pad:x2_pad] = pixelated_region
        
        return censored
    
    def apply_black_bar_censoring(self, img: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """Apply black bars to detected regions"""
        censored = img.copy()
        
        for x1, y1, x2, y2, label, confidence in boxes:
            # Add padding
            h, w = img.shape[:2]
            pad_x = int((x2 - x1) * PADDING_RATIO)
            pad_y = int((y2 - y1) * PADDING_RATIO)
            
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            # Draw black rectangle
            cv2.rectangle(censored, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 0, 0), -1)
        
        return censored
    
    def smart_crop_censoring(self, img: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """Intelligently crop image to avoid NSFW regions"""
        if not boxes:
            return img
        
        h, w = img.shape[:2]
        
        # Find the region that covers all detections
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[2] for box in boxes)
        max_y = max(box[3] for box in boxes)
        
        # Add padding
        pad_x = int((max_x - min_x) * PADDING_RATIO)
        pad_y = int((max_y - min_y) * PADDING_RATIO)
        
        # Determine best crop strategy
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Try different cropping strategies
        if center_y > h * 0.6:  # NSFW region in bottom - crop top
            cropped = img[:min_y - pad_y, :]
        elif center_y < h * 0.4:  # NSFW region in top - crop bottom
            cropped = img[max_y + pad_y:, :]
        elif center_x > w * 0.6:  # NSFW region in right - crop left
            cropped = img[:, :min_x - pad_x]
        elif center_x < w * 0.4:  # NSFW region in left - crop right
            cropped = img[:, max_x + pad_x:]
        else:
            # Central region - use blur instead
            return self.apply_blur_censoring(img, boxes)
        
        # Resize back to original dimensions if crop is too small
        if cropped.shape[0] < h * 0.3 or cropped.shape[1] < w * 0.3:
            return self.apply_blur_censoring(img, boxes)
        
        return cv2.resize(cropped, (w, h))
    
    def censor_image(self, img: np.ndarray, method: str = CENSOR_METHOD) -> np.ndarray:
        """Apply censoring to image based on specified method"""
        # Get NSFW score
        nsfw_score = self.get_nsfw_score(img)
        logger.info(f"NSFW Score: {nsfw_score:.3f}")
        
        if nsfw_score < NSFW_THRESHOLD:
            logger.info("Image is safe. No censoring needed.")
            return img
        
        # Get detection boxes
        boxes = self.get_detection_boxes(img)
        
        if not boxes:
            logger.info("No specific NSFW regions detected. Returning original image.")
            return img
        
        # Apply censoring based on method
        if method == "blur":
            return self.apply_blur_censoring(img, boxes)
        elif method == "pixelate":
            return self.apply_pixelate_censoring(img, boxes)
        elif method == "black_bar":
            return self.apply_black_bar_censoring(img, boxes)
        elif method == "crop":
            return self.smart_crop_censoring(img, boxes)
        else:
            logger.warning(f"Unknown method {method}, using blur instead")
            return self.apply_blur_censoring(img, boxes)
    
    def process_image(self, input_path: str, output_path: str, method: str = CENSOR_METHOD):
        """Process a single image file"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        logger.info(f"Processing: {input_path}")
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Process image
        censored_img = self.censor_image(img, method)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, censored_img)
        logger.info(f"Censored image saved: {output_path}")
    
    def process_video_frame(self, frame: np.ndarray, method: str = CENSOR_METHOD) -> np.ndarray:
        """Process a single video frame"""
        return self.censor_image(frame, method)

def main():
    """Example usage"""
    censor = NSFWCensor()
    
    # Process image
    input_image = "input.jpg"
    output_image = "censored_output.jpg"
    
    if os.path.exists(input_image):
        censor.process_image(input_image, output_image, method="blur")
    else:
        logger.warning(f"Input image {input_image} not found. Please provide a valid image path.")
        
        # Create a simple test
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.imwrite("test_input.jpg", test_img)
        logger.info("Created test_input.jpg for testing")

if __name__ == "__main__":
    main()