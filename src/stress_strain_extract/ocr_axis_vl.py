import os
import dashscope
import base64
import re
from pathlib import Path

# If using Singapore region, uncomment the line below
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

def encode_image_to_base64(image_path):
    """Convert local image to base64 encoding"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_numbers_with_positions(image_path, output_path):
    """
    Extract all numbers and their positions from an image
    
    Args:
        image_path: Path to the image file
        output_path: Path to save the output file (axis.txt)
        
    Returns:
        list: List of dictionaries containing:
              {
                  "number": "123",
                  "center_x": 100.5,
                  "center_y": 200.3
              }
    """
    # Read and encode image
    image_base64 = encode_image_to_base64(image_path)
    
    # Build message
    messages = [{
        "role": "user",
        "content": [
            {
                "image": f"data:image/png;base64,{image_base64}",
                # Minimum pixel threshold for upscaling
                "min_pixels": 28 * 28 * 4,
                # Maximum pixel threshold for downscaling
                "max_pixels": 28 * 28 * 8192,
                # Enable image auto-rotation
                "enable_rotate": True
            },
            {
                "text": "Locate all numbers in the image and return only their center point coordinates [cx, cy]. Output format: number, cx, cy"
            }
        ]
    }]
    
    # Call API
    response = dashscope.MultiModalConversation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-vl-ocr-2025-08-28',
        messages=messages,
        ocr_options={"task": "advanced_recognition"}
    )
    
    # Get OCR results
    ocr_text = response["output"]["choices"][0]["message"].content[0]["text"]
    
    # Save raw LLM output to axis.txt in the same folder as input image
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    
    print(f"Results saved to: {output_path}")
    
    # Parse results to extract numbers and positions
    results = parse_ocr_results(ocr_text)
    
    return results

def find_and_process_images(base_path="."):
    """
    Find and process all valid in.png files in the directory structure
    
    Directory structure:
    - L0001 to L2594 folders
      - YS* subfolders (must exist)
        - split* subfolders (optional)
          - in.png (must exist)
          - split_check.csv (must exist)
        - OR directly in.png + split_check.csv (if no split* folders)
    
    Args:
        base_path: Base directory to search (default: current directory)
    """
    base_path = Path(base_path)
    processed_count = 0
    skipped_count = 0
    
    # Loop through L0001 to L2594 folders
    for l_folder in sorted(base_path.glob("L*")):
        if not l_folder.is_dir():
            continue
        
        print(f"\nProcessing folder: {l_folder.name}")
        
        # Check if there are any YS* subfolders
        ys_folders = list(l_folder.glob("YS*"))
        if not ys_folders:
            print(f"  No YS* subfolder found, skipping {l_folder.name}")
            skipped_count += 1
            continue
        
        # Loop through all YS* subfolders
        for ys_folder in ys_folders:
            if not ys_folder.is_dir():
                continue
            
            print(f"  Checking YS folder: {ys_folder.name}")
            
            # Check for split* subfolders
            split_folders = list(ys_folder.glob("split*"))
            
            if split_folders:
                # Case 1: YS folder has split* subfolders
                for split_folder in split_folders:
                    if not split_folder.is_dir():
                        continue
                    
                    in_png = split_folder / "in.png"
                    split_check = split_folder / "split_check.csv"
                    
                    if in_png.exists() and split_check.exists():
                        print(f"    Processing: {split_folder.relative_to(base_path)}")
                        output_path = split_folder / "axis.txt"
                        
                        try:
                            extract_numbers_with_positions(str(in_png), str(output_path))
                            processed_count += 1
                        except Exception as e:
                            print(f"    Error processing {in_png}: {e}")
                    else:
                        missing = []
                        if not in_png.exists():
                            missing.append("in.png")
                        if not split_check.exists():
                            missing.append("split_check.csv")
                        print(f"    Skipping {split_folder.name}: missing {', '.join(missing)}")
                        skipped_count += 1
            else:
                # Case 2: No split* folders, check directly in YS folder
                in_png = ys_folder / "in.png"
                split_check = ys_folder / "split_check.csv"
                
                if in_png.exists() and split_check.exists():
                    print(f"    Processing: {ys_folder.relative_to(base_path)}")
                    output_path = ys_folder / "axis.txt"
                    
                    try:
                        extract_numbers_with_positions(str(in_png), str(output_path))
                        processed_count += 1
                    except Exception as e:
                        print(f"    Error processing {in_png}: {e}")
                else:
                    missing = []
                    if not in_png.exists():
                        missing.append("in.png")
                    if not split_check.exists():
                        missing.append("split_check.csv")
                    print(f"    Skipping {ys_folder.name}: missing {', '.join(missing)}")
                    skipped_count += 1
    
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print("="*50)

def parse_ocr_results(ocr_text):
    """
    Parse OCR results to extract numbers and their positions
    
    Args:
        ocr_text: Text result returned by OCR
        
    Returns:
        list: Extracted numbers and position information
    """
    results = []
    
    # Try to parse JSON format results
    try:
        # OCR results may be in JSON format with text and coordinate info
        lines = ocr_text.strip().split('\n')
        
        for line in lines:
            # Try to match lines containing coordinates
            # Format might be like: "123" [100.5, 200.3, 50, 20, 0]
            match = re.search(r'["\']?([^"\']+)["\']?\s*[\[,]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', line)
            
            if match:
                text = match.group(1).strip()
                cx = float(match.group(2))
                cy = float(match.group(3))
                width = float(match.group(4))
                height = float(match.group(5))
                angle = float(match.group(6))
                
                # Only keep text containing digits
                if re.search(r'\d', text):
                    results.append({
                        "number": text,
                        "center_x": cx,
                        "center_y": cy
                    })
    except Exception as e:
        print(f"Parse error: {e}")
        print("Trying alternative parsing method...")
    
    return results

def main():
    # Process all valid in.png files in the directory structure
    print("Starting batch OCR processing...")
    print("="*50)
    find_and_process_images(".")

if __name__ == "__main__":
    main()