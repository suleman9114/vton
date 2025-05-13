import argparse
import os
import sys
from PIL import Image
import numpy as np
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from utils_mask import get_mask_location
import torch
import matplotlib.pyplot as plt

def test_mask(image_path, category="upper_body", jacket=False, hip_ratio=0.15, output_dir="mask_test"):
    """Test the masking functionality on a person image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    human_img = Image.open(image_path).convert("RGB").resize((768, 1024))
    
    # Save resized input
    human_img.save(os.path.join(output_dir, "input_resized.png"))
    
    # Initialize models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    # Get keypoints and parsing
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    
    # Save the parsing result for visualization
    plt.figure(figsize=(10, 15))
    plt.imshow(model_parse)
    plt.title("Parsing Result")
    plt.savefig(os.path.join(output_dir, "parsing_result.png"))
    plt.close()
    
    # Get mask with and without jacket
    mask_no_jacket, mask_gray_no_jacket = get_mask_location('hd', category, model_parse, keypoints, jacket=False)
    mask_no_jacket = mask_no_jacket.resize((768, 1024))
    mask_no_jacket.save(os.path.join(output_dir, "mask_no_jacket.png"))
    
    # Test with the specified hip_ratio
    mask_jacket, mask_gray_jacket = get_mask_location('hd', category, model_parse, keypoints, jacket=True, hip_ratio=hip_ratio)
    mask_jacket = mask_jacket.resize((768, 1024))
    mask_jacket.save(os.path.join(output_dir, f"mask_jacket_ratio_{hip_ratio:.2f}.png"))
    
    # Create a comparison image
    comp_width = 768 * 3
    comp_height = 1024
    comparison = Image.new('RGB', (comp_width, comp_height))
    
    # Paste the images side by side
    comparison.paste(human_img, (0, 0))
    comparison.paste(mask_no_jacket.convert('RGB'), (768, 0))
    comparison.paste(mask_jacket.convert('RGB'), (768*2, 0))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("Arial", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((384, 50), "Original Image", fill=(255, 0, 0), font=font, anchor="mm")
    draw.text((768 + 384, 50), "No Jacket Mask", fill=(255, 0, 0), font=font, anchor="mm")
    draw.text((768*2 + 384, 50), f"Jacket Mask (ratio={hip_ratio:.2f})", fill=(255, 0, 0), font=font, anchor="mm")
    
    comparison.save(os.path.join(output_dir, f"comparison_ratio_{hip_ratio:.2f}.png"))
    
    print(f"Saved mask test results to {output_dir}/")
    print(f"Check comparison_ratio_{hip_ratio:.2f}.png to see results with hip_ratio={hip_ratio}")
    
    return mask_jacket, mask_no_jacket

def test_different_ratios(image_path, category="upper_body", output_dir="mask_test"):
    """Test multiple hip_ratio values to find the best one"""
    # Test a range of hip ratios from 0.05 to 0.25
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    for ratio in ratios:
        print(f"Testing with hip_ratio = {ratio:.2f}")
        test_mask(image_path, category, jacket=True, hip_ratio=ratio, output_dir=output_dir)
    
    # Create a grid comparison of all ratios
    grid_width = 768 * 2  # Original + mask
    grid_height = 1024 * len(ratios)
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Load the original image 
    human_img = Image.open(image_path).convert("RGB").resize((768, 1024))
    
    # Add each ratio's result to the grid
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid_image)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("Arial", 30)
    except:
        font = ImageFont.load_default()
    
    for i, ratio in enumerate(ratios):
        y_offset = i * 1024
        
        # Add the original image in the first column
        grid_image.paste(human_img, (0, y_offset))
        
        # Add the mask for this ratio in the second column
        mask_path = os.path.join(output_dir, f"mask_jacket_ratio_{ratio:.2f}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('RGB')
            grid_image.paste(mask, (768, y_offset))
        
        # Add label for this row
        draw.text((384, y_offset + 50), f"Original Image", fill=(255, 0, 0), font=font, anchor="mm")
        draw.text((768 + 384, y_offset + 50), f"Hip Ratio: {ratio:.2f}", fill=(255, 0, 0), font=font, anchor="mm")
    
    # Save the grid comparison
    grid_image.save(os.path.join(output_dir, "hip_ratio_comparison.png"))
    print(f"Created grid comparison of all hip ratios: {output_dir}/hip_ratio_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test masking functionality')
    parser.add_argument('--image', type=str, required=True, help='Path to the human image')
    parser.add_argument('--category', type=str, default='upper_body', choices=['upper_body', 'lower_body', 'dresses'], 
                        help='Garment category')
    parser.add_argument('--output', type=str, default='mask_test', help='Output directory')
    parser.add_argument('--hip-ratio', type=float, default=0.15, help='Ratio of hip area to include in mask (0.05-0.25)')
    parser.add_argument('--test-multiple', action='store_true', help='Test multiple hip ratios to find the best one')
    
    args = parser.parse_args()
    
    if args.test_multiple:
        test_different_ratios(args.image, args.category, args.output)
        print(f"Testing with multiple hip ratios completed. Check {args.output} directory for results.")
    else:
        test_mask(args.image, args.category, jacket=True, hip_ratio=args.hip_ratio, output_dir=args.output)
        print(f"Testing completed with hip_ratio={args.hip_ratio}. Check {args.output} directory for results.") 