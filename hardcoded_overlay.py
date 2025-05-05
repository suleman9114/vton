#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

def apply_overlay_with_hardcoded_config(base_img_path, overlay_img_path, output_path):
    """Apply overlay using hardcoded configuration without the GUI or config file"""
    # Load images
    base_img = cv2.imread(base_img_path, cv2.IMREAD_UNCHANGED)
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    
    # Convert overlay to RGBA if it's not already
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    
    # Convert base image to RGBA if needed
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)
    
    # Hardcoded configuration values
    x_pos = 701
    y_pos = 84
    scale = 1.05
    rotation = 0
    
    # Resize overlay image based on scale
    h, w = overlay_img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized_overlay = cv2.resize(overlay_img, (new_w, new_h))
    
    # Apply rotation if needed
    if rotation > 0:
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        resized_overlay = cv2.warpAffine(resized_overlay, rotation_matrix, (new_w, new_h))
    
    # Calculate position
    y1 = max(0, y_pos)
    y2 = min(base_img.shape[0], y_pos + resized_overlay.shape[0])
    x1 = max(0, x_pos)
    x2 = min(base_img.shape[1], x_pos + resized_overlay.shape[1])
    
    # Calculate overlay image region that fits within base image
    overlay_y1 = max(0, -y_pos)
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x1 = max(0, -x_pos)
    overlay_x2 = overlay_x1 + (x2 - x1)
    
    # Create a copy of the base image to work on
    result = base_img.copy()
    
    # Ensure we're not out of bounds
    if (y1 >= y2 or x1 >= x2 or 
        overlay_y1 >= resized_overlay.shape[0] or 
        overlay_y2 > resized_overlay.shape[0] or
        overlay_x1 >= resized_overlay.shape[1] or 
        overlay_x2 > resized_overlay.shape[1]):
        # Return the base image if the overlay is entirely outside
        cv2.imwrite(output_path, result)
        return
    
    # Extract regions
    roi = result[y1:y2, x1:x2]
    overlay_roi = resized_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    
    # Make sure ROIs have the same shape
    if roi.shape[:2] != overlay_roi.shape[:2]:
        min_h = min(roi.shape[0], overlay_roi.shape[0])
        min_w = min(roi.shape[1], overlay_roi.shape[1])
        roi = roi[:min_h, :min_w]
        overlay_roi = overlay_roi[:min_h, :min_w]
    
    # Extract alpha channel
    if overlay_roi.shape[2] == 4:
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        # Blend images based on alpha
        for c in range(0, 3):  # For each color channel
            roi[:, :, c] = roi[:, :, c] * (1 - alpha[:, :, 0]) + overlay_roi[:, :, c] * alpha[:, :, 0]
        
        # Copy the blended region back
        result[y1:y1+roi.shape[0], x1:x1+roi.shape[1], :3] = roi[:, :, :3]
    else:
        # If no alpha channel, just copy the overlay
        result[y1:y2, x1:x2] = overlay_roi
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Generated overlay image at {output_path}")

def auto_crop_image(image_path, output_path=None):
    """
    Automatically crop an image to remove white/transparent space around the edges.
    
    Args:
        image_path: Path to the image to crop
        output_path: Path to save the cropped image (if None, will overwrite original)
    
    Returns:
        Path to the cropped image
    """
    if output_path is None:
        output_path = image_path
    
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        return image_path
    
    # Check if image has alpha channel (4 channels)
    if img.shape[2] == 4:
        # Use alpha channel to find non-transparent pixels
        mask = img[:, :, 3]
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    else:
        # Convert to grayscale and threshold to find non-white pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Make sure mask is 8-bit single-channel
    mask = mask.astype(np.uint8)
    
    # Find coordinates of non-transparent/non-white pixels
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("Warning: No non-transparent/non-white pixels found")
        return image_path
    
    # Get the bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add a small padding
    padding = 10
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)
    
    # Crop the image
    cropped = img[y_start:y_end, x_start:x_end]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Cropped image saved to {output_path}")
    
    return output_path

def apply_white_background(image_path, output_path=None):
    """
    Apply a white background to an image with transparency.
    
    Args:
        image_path: Path to the image
        output_path: Path to save the image with white background (if None, will overwrite original)
    
    Returns:
        Path to the output image
    """
    if output_path is None:
        output_path = image_path
    
    # Read the image with transparency
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        return image_path
    
    # Check if image has alpha channel
    if img.shape[2] == 4:
        # Create a white background
        white_background = np.ones(img.shape[:3], dtype=np.uint8) * 255
        
        # Extract alpha channel
        alpha = img[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Blend the image with the white background based on alpha
        for c in range(0, 3):
            white_background[:, :, c] = img[:, :, c] * alpha[:, :, 0] + white_background[:, :, c] * (1 - alpha[:, :, 0])
        
        # Save the image with white background
        cv2.imwrite(output_path, white_background)
        print(f"Image with white background saved to {output_path}")
    else:
        # If image doesn't have alpha channel, just copy it
        cv2.imwrite(output_path, img)
        print(f"Image already has no transparency, saved to {output_path}")
    
    return output_path

def main():
    if len(sys.argv) < 3:
        print("Usage: python hardcoded_overlay.py <base_image> <overlay_image> [output_path]")
        print("Example: python hardcoded_overlay.py coat-img.png shirt-design.png output.png")
        return

    base_img = sys.argv[1]
    overlay_img = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_overlay.png"
    
    # Check if input files exist
    if not os.path.exists(base_img):
        print(f"Error: Base image '{base_img}' not found")
        return

    if not os.path.exists(overlay_img):
        print(f"Error: Overlay image '{overlay_img}' not found")
        return

    # Apply the overlay using the hardcoded configuration
    apply_overlay_with_hardcoded_config(base_img, overlay_img, output_path)
    print(f"Successfully created overlaid image: {output_path}")
    
    final_output = output_path
    
    # Auto-crop the result
    cropped_output = output_path.rsplit('.', 1)[0] + "_cropped." + output_path.rsplit('.', 1)[1]
    auto_crop_image(output_path, cropped_output)
    print(f"Cropped image saved to: {cropped_output}")
    final_output = cropped_output
    
    # Apply white background
    white_bg_output = final_output.rsplit('.', 1)[0] + "_white_bg." + final_output.rsplit('.', 1)[1]
    apply_white_background(final_output, white_bg_output)
    print(f"Final image with white background saved to: {white_bg_output}")
    final_output = white_bg_output
    
    print(f"Process complete. Final image is: {final_output}")

if __name__ == "__main__":
    main() 
