import cv2
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

# Color mapping dictionary
COLOR_MAP = {
    'red': 1,
    'blue': 2,
    'green': 3,
    'black': 4,
    'yellow': 5,
    'cyan': 6,
    'magenta': 7,
    'white': 8,
    'orange': 9,
    'purple': 10,
    'gray': 11,
    'grey': 11  # Alternative spelling
}

def get_color_mask(img, color_option):
    """
    Generate a mask for the specified color with EXPANDED ranges.
    
    Color options:
    1 - Red
    2 - Blue
    3 - Green
    4 - Black
    5 - Yellow
    6 - Cyan
    7 - Magenta
    8 - White
    9 - Orange
    10 - Purple
    11 - Gray
    """
    
    # Convert to HSV for most colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if color_option == 1:  # Red
        # Red wraps around in HSV, so we need two ranges
        # EXPANDED: More permissive ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        color_name = "red"
        
    elif color_option == 2:  # Blue
        # EXPANDED: Lower saturation/value thresholds
        lower_blue = np.array([85, 50, 50])
        upper_blue = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        color_name = "blue"
        
    elif color_option == 3:  # Green
        # EXPANDED: Wider hue range, lower saturation/value
        lower_green = np.array([30, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        color_name = "green"
        
    elif color_option == 4:  # Black
        # EXPANDED: Increased value threshold
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        color_name = "black"
        
    elif color_option == 5:  # Yellow
        # EXPANDED: Lower saturation/value thresholds
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        color_name = "yellow"
        
    elif color_option == 6:  # Cyan
        # EXPANDED: Wider range
        lower_cyan = np.array([75, 50, 50])
        upper_cyan = np.array([105, 255, 255])
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        color_name = "cyan"
        
    elif color_option == 7:  # Magenta
        # EXPANDED: Wider range
        lower_magenta = np.array([135, 50, 50])
        upper_magenta = np.array([175, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        color_name = "magenta"
        
    elif color_option == 8:  # White
        # EXPANDED: Higher saturation tolerance
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        color_name = "white"
        
    elif color_option == 9:  # Orange
        # EXPANDED: Wider range
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        color_name = "orange"
        
    elif color_option == 10:  # Purple
        # EXPANDED: Wider range, lower thresholds
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        color_name = "purple"
        
    elif color_option == 11:  # Gray
        # EXPANDED: Wider value range
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 70, 220])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        color_name = "gray"
        
    else:
        print(f"Invalid color option: {color_option}")
        return None, None
    
    return mask, color_name


def draw_text(img, text, pos, color, font, font_scale, font_thickness):
    """Helper function for drawing text with boundary checks"""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = pos[0] + 15, pos[1] - 10
    if text_x + text_width > img.shape[1]:
        text_x = pos[0] - text_width - 15
    if text_y - text_height < 0:
        text_y = pos[1] + 20
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)


def process_curve(img_path, color_option, output_prefix, num_points=20):
    """
    Extract all patterns matching the specified color (not just curves).
    
    Parameters:
    - img_path: Path to input image
    - color_option: Color to extract (1-11)
    - output_prefix: Prefix for output files (e.g., 'out_blue')
    - num_points: Number of evenly spaced points to extract (default: 20)
    
    Returns:
    - True if successful, False otherwise
    """
    
    # Read the input image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image '{img_path}'")
        return False
    
    # Create a copy for annotation
    output_img = img.copy()
    
    # Get color mask
    mask, color_name = get_color_mask(img, color_option)
    if mask is None:
        return False
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for better connectivity
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print(f"  Analyzing {color_name} segments...")
    
    # Filter contours: only keep significant ones (length > 20)
    valid_segments = []
    for contour in contours:
        contour_points = contour.reshape(-1, 2)
        
        if len(contour_points) < 20:
            continue
        
        valid_segments.append(contour_points)
    
    if not valid_segments:
        print(f"  Warning: No {color_name} segments found!")
        return False
    
    # Combine ALL valid segments into one set of points
    # This extracts all matching patterns, not just one curve
    all_points = np.vstack(valid_segments)
    
    print(f"  Found {len(valid_segments)} segment(s) with total {len(all_points)} points")
    
    # Sort points by x-coordinate for left-to-right ordering
    sorted_indices = np.argsort(all_points[:, 0])
    curve_points = all_points[sorted_indices]
    
    # Group points by x-coordinate and take median y for each x
    x_to_y = {}
    for x, y in curve_points:
        if x not in x_to_y:
            x_to_y[x] = []
        x_to_y[x].append(y)
    
    # Get representative points (one per x-coordinate, using median y)
    sorted_curve_points = []
    for x in sorted(x_to_y.keys()):
        y_values = x_to_y[x]
        median_y = int(np.median(y_values))
        sorted_curve_points.append((x, median_y))
    
    # Extract evenly spaced points
    total_points = len(sorted_curve_points)
    
    if total_points < num_points:
        selected_points = sorted_curve_points
    else:
        # Calculate indices for evenly spaced points
        indices = np.linspace(0, total_points - 1, num_points, dtype=int)
        selected_points = [sorted_curve_points[i] for i in indices]
    
    # --- ANNOTATION AND DATA EXPORT SECTION ---
    
    # Define colors and styles
    green_color = (0, 255, 0)      # Green for regular points
    cross_size = 10
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Prepare data for CSV - only evenly spaced points
    csv_data = []
    
    # Process and store evenly spaced points
    for i, (x, y) in enumerate(selected_points):
        csv_data.append([f"P{i+1}", x, y])
    
    # Annotate the image with green crosses only
    for (x, y) in selected_points:
        cv2.line(output_img, (x - cross_size, y), (x + cross_size, y), green_color, line_thickness)
        cv2.line(output_img, (x, y - cross_size), (x, y + cross_size), green_color, line_thickness)
        draw_text(output_img, f"({x},{y})", (x, y), green_color, font, font_scale, font_thickness)
    
    # Get output directory from input path
    output_dir = os.path.dirname(img_path)
    
    # Save the annotated image
    output_img_path = os.path.join(output_dir, f'{output_prefix}.png')
    cv2.imwrite(output_img_path, output_img)
    
    # Save coordinates to CSV file
    output_csv_path = os.path.join(output_dir, f'{output_prefix}.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Point', 'X', 'Y'])
        csv_writer.writerows(csv_data)
    
    print(f"  Saved: {output_prefix}.png and {output_prefix}.csv")
    
    return True


def log_unknown_color(debug_file_path, folder_path, color):
    """Log unknown color to debug_color.csv"""
    file_exists = os.path.exists(debug_file_path)
    
    with open(debug_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Path', 'Unknown_Color'])
        writer.writerow([folder_path, color])


def process_directory(dir_path, debug_file_path):
    """Process a single directory containing in.png and label.csv"""
    
    in_png_path = os.path.join(dir_path, 'in.png')
    label_csv_path = os.path.join(dir_path, 'label.csv')
    
    # Check if both files exist
    if not os.path.exists(in_png_path):
        return
    
    if not os.path.exists(label_csv_path):
        return
    
    print(f"\nProcessing: {dir_path}")
    
    # Read label.csv
    try:
        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                label = row[0].strip()
                color = row[1].strip().lower()
                
                # Check if color is supported
                if color not in COLOR_MAP:
                    print(f"  Unknown color '{color}' for label '{label}'")
                    log_unknown_color(debug_file_path, dir_path, color)
                    continue
                
                # Get color option
                color_option = COLOR_MAP[color]
                
                # Process the curve
                output_prefix = f'out_{color}'
                success = process_curve(in_png_path, color_option, output_prefix, num_points=20)
                
                if not success:
                    print(f"  Failed to extract {color} pattern for '{label}'")
    
    except Exception as e:
        print(f"  Error processing {dir_path}: {e}")


def main():
    """Main function to process all directories"""
    
    print("=" * 60)
    print("EXPANDED RANGE BATCH EXTRACTOR")
    print("=" * 60)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    debug_file_path = script_dir / f'debug_color_{datetime.now():%Y%m%d_%H%M%S}.csv'
    
    # Find all L#### directories
    l_dirs = sorted([d for d in script_dir.iterdir() if d.is_dir() and d.name.startswith('L')])
    
    print(f"Found {len(l_dirs)} directories starting with 'L'")
    
    processed_count = 0
    
    for l_dir in l_dirs:
        # Check if YS subfolder exists
        ys_dirs = [d for d in l_dir.iterdir() if d.is_dir() and d.name.startswith('YS')]
        
        if not ys_dirs:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing directory: {l_dir.name}")
        print(f"{'='*60}")
        
        # Process each YS directory
        for ys_dir in ys_dirs:
            # Check for split subdirectories
            split_dirs = [d for d in ys_dir.iterdir() if d.is_dir() and d.name.startswith('split')]
            
            if split_dirs:
                # Process each split directory
                for split_dir in split_dirs:
                    process_directory(str(split_dir), str(debug_file_path))
                    processed_count += 1
            else:
                # No split directory, process YS directory directly
                process_directory(str(ys_dir), str(debug_file_path))
                processed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Processing completed!")
    print(f"Total directories processed: {processed_count}")
    if debug_file_path.exists():
        print(f"Unknown colors logged to: {debug_file_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
