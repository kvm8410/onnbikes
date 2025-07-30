#!/usr/bin/env python3
"""
Debug version of the NSFW Censoring System
This helps diagnose detection issues and test different configurations.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from nudenet import NudeDetector
from transformers import AutoModelForImageClassification, AutoImageProcessor
import logging
import os
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All possible NudeNet labels
ALL_NUDENET_LABELS = {
    "EXPOSED_ANUS", "EXPOSED_ARMPITS", "EXPOSED_BELLY", "EXPOSED_BUTTOCKS",
    "EXPOSED_BREAST_F", "EXPOSED_BREAST_M", "EXPOSED_FEET", "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M", "COVERED_GENITALIA_F", "COVERED_GENITALIA_M",
    "COVERED_BREAST_F", "COVERED_BREAST_M", "COVERED_BUTTOCKS", "FACE_F", "FACE_M"
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class NSFWDebugger:
    def __init__(self):
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        self.load_models()
    
    def load_models(self):
        """Load models with detailed logging"""
        try:
            logger.info("Loading ViT NSFW Classifier...")
            self.vit_model = AutoModelForImageClassification.from_pretrained(
                "Falconsai/nsfw_image_detection"
            ).to(self.device)
            self.vit_processor = AutoImageProcessor.from_pretrained(
                "Falconsai/nsfw_image_detection"
            )
            logger.info("✅ ViT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ViT model: {e}")
            raise
        
        try:
            logger.info("Loading NudeNet Detector...")
            self.detector = NudeDetector()
            logger.info("✅ NudeNet loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load NudeNet: {e}")
            raise
    
    def analyze_image(self, image_path: str):
        """Comprehensive analysis of an image"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 ANALYZING: {image_path}")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(image_path):
            logger.error(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"❌ Could not read image: {image_path}")
            return
        
        logger.info(f"📏 Image dimensions: {img.shape}")
        
        # Step 1: ViT NSFW Classification
        logger.info(f"\n🤖 STEP 1: ViT NSFW Classification")
        nsfw_score = self.get_nsfw_score(img)
        logger.info(f"🎯 NSFW Score: {nsfw_score:.3f}")
        if nsfw_score > 0.5:
            logger.info("🔥 Image classified as NSFW")
        else:
            logger.info("✅ Image classified as Safe")
        
        # Step 2: NudeNet Region Detection
        logger.info(f"\n🎯 STEP 2: NudeNet Region Detection")
        all_detections = self.get_all_detections(img)
        
        # Step 3: Analysis Summary
        logger.info(f"\n📊 ANALYSIS SUMMARY")
        logger.info(f"ViT NSFW Score: {nsfw_score:.3f}")
        logger.info(f"Total NudeNet Detections: {len(all_detections)}")
        
        if all_detections:
            logger.info("Detected regions:")
            for i, det in enumerate(all_detections):
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, label = det[:6]
                    logger.info(f"  {i+1}. {label} (confidence: {conf:.3f}) at ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        else:
            logger.info("No regions detected by NudeNet")
        
        return nsfw_score, all_detections
    
    def get_nsfw_score(self, img: np.ndarray) -> float:
        """Get NSFW probability score using ViT model"""
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            inputs = self.vit_processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                nsfw_score = scores[0][1].item()
            
            return nsfw_score
        except Exception as e:
            logger.error(f"Error in NSFW scoring: {e}")
            return 0.0
    
    def get_all_detections(self, img: np.ndarray):
        """Get ALL detections from NudeNet with detailed logging"""
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector.detect(rgb_img)
            
            logger.info(f"📍 NudeNet found {len(results)} detections")
            
            if not results:
                logger.info("❌ No detections found")
                return []
            
            for i, detection in enumerate(results):
                logger.info(f"\nDetection {i+1}: {detection}")
                
                if len(detection) >= 6:
                    x1, y1, x2, y2, confidence, label = detection[:6]
                    logger.info(f"  📍 Coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                    logger.info(f"  🎯 Confidence: {confidence:.3f}")
                    logger.info(f"  🏷️  Label: {label}")
                    logger.info(f"  ✅ Known Label: {label in ALL_NUDENET_LABELS}")
                else:
                    logger.warning(f"  ⚠️  Malformed detection (only {len(detection)} elements)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def test_different_targets(self, image_path: str):
        """Test different target label configurations"""
        logger.info(f"\n🧪 TESTING DIFFERENT TARGET CONFIGURATIONS")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read {image_path}")
            return
        
        all_detections = self.get_all_detections(img)
        
        if not all_detections:
            logger.info("❌ No detections to filter")
            return
        
        # Test different target configurations
        configs = {
            "Conservative (Genitalia only)": {
                "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M", "COVERED_GENITALIA_F", "COVERED_GENITALIA_M"
            },
            "Standard (Original targets)": {
                "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M", 
                "EXPOSED_BUTTOCKS", "EXPOSED_BREAST_F",
                "EXPOSED_ANUS", "COVERED_GENITALIA_F", "COVERED_GENITALIA_M"
            },
            "Inclusive (All exposed)": {
                "EXPOSED_ANUS", "EXPOSED_BUTTOCKS", "EXPOSED_BREAST_F", 
                "EXPOSED_BREAST_M", "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M"
            },
            "All body parts": ALL_NUDENET_LABELS - {"FACE_F", "FACE_M"}
        }
        
        for config_name, target_labels in configs.items():
            logger.info(f"\n📋 {config_name}:")
            matches = 0
            for det in all_detections:
                if len(det) >= 6:
                    _, _, _, _, conf, label = det[:6]
                    if label in target_labels and conf > 0.3:
                        matches += 1
                        logger.info(f"  ✅ Would detect: {label} (conf: {conf:.3f})")
            
            if matches == 0:
                logger.info(f"  ❌ No matches with this configuration")
            else:
                logger.info(f"  🎯 Total matches: {matches}")
    
    def visualize_detections(self, image_path: str, output_path: str = None):
        """Create a visualization of all detections"""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read {image_path}")
            return
        
        all_detections = self.get_all_detections(img)
        
        if not all_detections:
            logger.info("No detections to visualize")
            return
        
        # Draw all detections
        vis_img = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, det in enumerate(all_detections):
            if len(det) >= 6:
                x1, y1, x2, y2, conf, label = det[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                color = colors[i % len(colors)]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_img, f"{label} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if output_path is None:
            output_path = f"debug_visualization_{os.path.basename(image_path)}"
        
        cv2.imwrite(output_path, vis_img)
        logger.info(f"📸 Visualization saved: {output_path}")

def main():
    """Debug main function"""
    debugger = NSFWDebugger()
    
    # Test image
    test_image = "input.jpg"
    
    if not os.path.exists(test_image):
        logger.warning(f"Test image {test_image} not found")
        logger.info("Please provide an image file named 'input.jpg' or modify the script")
        return
    
    # Run comprehensive analysis
    debugger.analyze_image(test_image)
    debugger.test_different_targets(test_image)
    debugger.visualize_detections(test_image)
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Debug analysis complete!")
    logger.info("Check the logs above to understand what's being detected")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()