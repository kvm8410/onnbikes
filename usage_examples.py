#!/usr/bin/env python3
"""
Usage examples for the improved NSFW censoring system.
This file demonstrates how to use the NSFWCensor class for various scenarios.
"""

import cv2
import os
import glob
from improved_nsfw_censor import NSFWCensor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_single_image():
    """Example: Process a single image with different censoring methods"""
    censor = NSFWCensor()
    
    input_image = "test_image.jpg"
    
    # Test different censoring methods
    methods = ["blur", "pixelate", "black_bar", "crop"]
    
    for method in methods:
        output_path = f"output_{method}.jpg"
        try:
            censor.process_image(input_image, output_path, method=method)
            logger.info(f"Processed with {method} method -> {output_path}")
        except FileNotFoundError:
            logger.warning(f"Input image {input_image} not found. Skipping {method} method.")
        except Exception as e:
            logger.error(f"Error processing with {method}: {e}")

def example_batch_processing():
    """Example: Process multiple images in a directory"""
    censor = NSFWCensor()
    
    input_dir = "input_images"
    output_dir = "censored_images"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"censored_{filename}")
        
        try:
            censor.process_image(img_path, output_path, method="blur")
            logger.info(f"Processed: {filename}")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

def example_video_processing():
    """Example: Process video frames in real-time"""
    censor = NSFWCensor()
    
    input_video = "input_video.mp4"
    output_video = "censored_video.mp4"
    
    if not os.path.exists(input_video):
        logger.warning(f"Video file {input_video} not found")
        return
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    
    logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        try:
            censored_frame = censor.process_video_frame(frame, method="blur")
            out.write(censored_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Processed {frame_count}/{total_frames} frames")
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            out.write(frame)  # Write original frame if error
    
    # Release everything
    cap.release()
    out.release()
    logger.info(f"Video processing complete. Output saved: {output_video}")

def example_live_camera():
    """Example: Real-time censoring from camera feed"""
    censor = NSFWCensor()
    
    # Open camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Can't receive frame")
            break
        
        # Process every 5th frame for performance (adjust as needed)
        if frame_count % 5 == 0:
            try:
                censored_frame = censor.process_video_frame(frame, method="blur")
                cv2.imshow('Censored Live Feed', censored_frame)
            except Exception as e:
                logger.error(f"Error processing live frame: {e}")
                cv2.imshow('Censored Live Feed', frame)
        else:
            cv2.imshow('Censored Live Feed', frame)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'saved_frame_{frame_count}.jpg', frame)
            logger.info(f"Saved frame {frame_count}")
    
    cap.release()
    cv2.destroyAllWindows()

def example_custom_configuration():
    """Example: Custom configuration and advanced usage"""
    
    # You can modify the global configuration
    from improved_nsfw_censor import (
        NSFW_THRESHOLD, DETECTION_CONFIDENCE, 
        BLUR_INTENSITY, PIXELATE_RATIO, PADDING_RATIO
    )
    
    logger.info("Current configuration:")
    logger.info(f"NSFW_THRESHOLD: {NSFW_THRESHOLD}")
    logger.info(f"DETECTION_CONFIDENCE: {DETECTION_CONFIDENCE}")
    logger.info(f"BLUR_INTENSITY: {BLUR_INTENSITY}")
    logger.info(f"PIXELATE_RATIO: {PIXELATE_RATIO}")
    logger.info(f"PADDING_RATIO: {PADDING_RATIO}")
    
    # Create censor with custom parameters
    censor = NSFWCensor()
    
    # Test with a sample image
    test_image = "test_input.jpg"
    
    # Create a test image if it doesn't exist
    if not os.path.exists(test_image):
        import numpy as np
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.imwrite(test_image, test_img)
        logger.info(f"Created {test_image} for testing")
    
    # Process with different methods
    methods = ["blur", "pixelate", "black_bar"]
    for method in methods:
        try:
            output_path = f"custom_{method}_output.jpg"
            censor.process_image(test_image, output_path, method=method)
            logger.info(f"Created {output_path} with {method} method")
        except Exception as e:
            logger.error(f"Error with {method}: {e}")

def main():
    """Run all examples"""
    logger.info("Starting NSFW Censoring Examples")
    
    # Run examples
    logger.info("\n=== Example 1: Single Image Processing ===")
    example_single_image()
    
    logger.info("\n=== Example 2: Batch Processing ===")
    example_batch_processing()
    
    logger.info("\n=== Example 3: Custom Configuration ===")
    example_custom_configuration()
    
    # Uncomment these for video/camera examples
    # logger.info("\n=== Example 4: Video Processing ===")
    # example_video_processing()
    
    # logger.info("\n=== Example 5: Live Camera Feed ===")
    # example_live_camera()
    
    logger.info("\nAll examples completed!")

if __name__ == "__main__":
    main()