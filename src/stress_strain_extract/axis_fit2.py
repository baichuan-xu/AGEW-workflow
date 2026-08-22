import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, List
import warnings
import re
import json
warnings.filterwarnings('ignore')


def parse_axis_txt(axis_file_path: Path) -> Optional[pd.DataFrame]:
    """
    Parse axis.txt file and extract data.
    Skip first 2 and last 2 lines.
    Extract Pixel X, Pixel Y from rotate_rect and Identified Number from text.
    Only keep rows where text is purely numeric.
    """
    try:
        with open(axis_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip first 2 and last 2 lines
        if len(lines) <= 4:
            print(f"  ⚠ File has too few lines: {axis_file_path}")
            return None
        
        middle_lines = lines[2:-2]
        
        data = []
        
        for line in middle_lines:
            line = line.strip()
            
            # Extract rotate_rect and text using regex
            # Pattern: "rotate_rect": [x, y, w, h, angle], "text": "value"
            rotate_rect_match = re.search(r'"rotate_rect"\s*:\s*\[([^\]]+)\]', line)
            text_match = re.search(r'"text"\s*:\s*"([^"]*)"', line)
            
            if not rotate_rect_match or not text_match:
                continue
            
            # Parse rotate_rect values
            rotate_rect_str = rotate_rect_match.group(1)
            rect_values = [float(x.strip()) for x in rotate_rect_str.split(',')]
            
            if len(rect_values) < 2:
                continue
            
            pixel_x = rect_values[0]
            pixel_y = rect_values[1]
            
            # Parse text value
            text_value = text_match.group(1).strip()
            
            # Check if text is purely numeric (only digits, possibly with negative sign or decimal)
            if re.match(r'^-?\d+\.?\d*$', text_value):
                identified_number = float(text_value)
                data.append({
                    'Identified Number': identified_number,
                    'Pixel X': pixel_x,
                    'Pixel Y': pixel_y
                })
        
        if not data:
            print(f"  ⚠ No valid numeric data found in: {axis_file_path}")
            return None
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"  ✗ Error parsing axis.txt: {str(e)}")
        return None


def find_similar_values(values: np.ndarray, threshold: float = 20) -> List[List[int]]:
    """
    Group indices of values that are similar within a threshold.
    Returns a list of groups, where each group is a list of indices.
    """
    if len(values) == 0:
        return []
    
    sorted_indices = np.argsort(values)
    groups = []
    current_group = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        idx = sorted_indices[i]
        prev_idx = current_group[-1]
        
        if abs(values[idx] - values[prev_idx]) <= threshold:
            current_group.append(idx)
        else:
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = [idx]
    
    if len(current_group) >= 3:
        groups.append(current_group)
    
    return groups


def iterative_linear_fit(x: np.ndarray, y: np.ndarray, max_iterations: int = 10, 
                         r_squared_threshold: float = 0.95) -> Optional[Tuple[float, float, float]]:
    """
    Perform iterative linear regression, removing outliers until a good fit is found.
    Returns (slope, intercept, r_squared) or None if no good fit can be achieved.
    """
    if len(x) < 3:
        return None
    
    indices = np.arange(len(x))
    
    for iteration in range(max_iterations):
        if len(indices) < 3:
            return None
        
        x_fit = x[indices].reshape(-1, 1)
        y_fit = y[indices]
        
        model = LinearRegression()
        model.fit(x_fit, y_fit)
        
        y_pred = model.predict(x_fit)
        residuals = np.abs(y_fit - y_pred)
        
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # If good fit achieved, return results with R²
        if r_squared >= r_squared_threshold or len(indices) == 3:
            slope = model.coef_[0]
            intercept = model.intercept_
            return (slope, intercept, r_squared)
        
        # Remove the point with largest residual
        max_residual_idx = np.argmax(residuals)
        indices = np.delete(indices, max_residual_idx)
    
    return None


def process_first_group(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Process the first group: similar Pixel X values, Identified Number is 0 or multiple of 100.
    Returns (a1, b1) from the group with the best linear fit (highest R²).
    """
    # Filter rows where Identified Number is 0 or multiple of 100
    filtered_df = df[df['Identified Number'] % 100 == 0].copy()
    
    if len(filtered_df) < 3:
        return None
    
    # Find groups with similar Pixel X values
    pixel_x_groups = find_similar_values(filtered_df['Pixel X'].values, threshold=20)
    
    if not pixel_x_groups:
        return None
    
    # Try all groups and keep track of the best one
    best_result = None
    best_r_squared = -1
    
    for group_indices in pixel_x_groups:
        if len(group_indices) < 3:
            continue
        
        group_df = filtered_df.iloc[group_indices]
        
        # Fit: Pixel Y (x-axis) vs Identified Number (y-axis)
        x = group_df['Pixel Y'].values
        y = group_df['Identified Number'].values
        
        result = iterative_linear_fit(x, y)
        
        if result is not None:
            slope, intercept, r_squared = result
            
            # Keep the result with the highest R²
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_result = (slope, intercept)
    
    return best_result


def process_second_group(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Process the second group: similar Pixel Y values.
    Returns (a2, b2) from the group with the best linear fit (highest R²).
    No requirement on Identified Number similarity - only linear fit quality matters.
    """
    if len(df) < 3:
        return None
    
    # Find groups with similar Pixel Y values
    pixel_y_groups = find_similar_values(df['Pixel Y'].values, threshold=20)
    
    if not pixel_y_groups:
        return None
    
    # Try all groups and keep track of the best one
    best_result = None
    best_r_squared = -1
    
    for group_indices in pixel_y_groups:
        if len(group_indices) < 3:
            continue
        
        group_df = df.iloc[group_indices]
        
        # Fit: Pixel X (x-axis) vs Identified Number (y-axis)
        x = group_df['Pixel X'].values
        y = group_df['Identified Number'].values
        
        result = iterative_linear_fit(x, y)
        
        if result is not None:
            slope, intercept, r_squared = result
            
            # Keep the result with the highest R²
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_result = (slope, intercept)
    
    return best_result


def process_axis_file(axis_file_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Process a single axis.txt file and return (a1, b1, a2, b2) or None.
    """
    try:
        # Parse axis.txt file
        df = parse_axis_txt(axis_file_path)
        
        if df is None or len(df) < 3:
            print(f"  ⚠ Insufficient data in {axis_file_path}")
            return None
        
        print(f"    Parsed {len(df)} valid data points")
        
        # Process first group - returns the best fit
        first_result = process_first_group(df)
        if first_result is None:
            print(f"  ⚠ First group fitting failed for {axis_file_path}")
            return None
        
        a1, b1 = first_result
        
        # Process second group - returns the best fit
        second_result = process_second_group(df)
        if second_result is None:
            print(f"  ⚠ Second group fitting failed for {axis_file_path}")
            return None
        
        a2, b2 = second_result
        
        return (a1, b1, a2, b2)
    
    except Exception as e:
        print(f"  ✗ Error processing {axis_file_path}: {str(e)}")
        return None


def save_fit_results(output_path: Path, a1: float, b1: float, a2: float, b2: float):
    """
    Save fitting results to fit2.csv file.
    """
    fit_df = pd.DataFrame({
        'a1': [a1],
        'b1': [b1],
        'a2': [a2],
        'b2': [b2]
    })
    
    fit_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"    a1={a1:.6f}, b1={b1:.6f}, a2={a2:.6f}, b2={b2:.6f}")


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting batch processing of axis.txt files")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        print(f"\n[{folder_name}]")
        
        # Find all YS* subfolders
        ys_folders = sorted([f for f in folder_path.iterdir() 
                            if f.is_dir() and f.name.startswith('YS')])
        
        if not ys_folders:
            print(f"  ⊘ No YS* folders found, skipping")
            continue
        
        print(f"  Found {len(ys_folders)} YS* folder(s)")
        
        # Process each YS* folder
        for ys_folder in ys_folders:
            print(f"\n  Processing: {ys_folder.name}")
            
            # Check for split* subfolders
            split_folders = sorted([f for f in ys_folder.iterdir() 
                                   if f.is_dir() and f.name.startswith('split')])
            
            if split_folders:
                # Process each split* folder
                for split_folder in split_folders:
                    axis_file = split_folder / 'axis.txt'
                    
                    if not axis_file.exists():
                        print(f"    ⊘ No axis.txt in {split_folder.name}")
                        continue
                    
                    print(f"    Processing: {split_folder.name}/axis.txt")
                    processed_count += 1
                    
                    result = process_axis_file(axis_file)
                    
                    if result is not None:
                        a1, b1, a2, b2 = result
                        fit_output = split_folder / 'fit2.csv'
                        save_fit_results(fit_output, a1, b1, a2, b2)
                        success_count += 1
            else:
                # No split folders, check for axis.txt directly in YS folder
                axis_file = ys_folder / 'axis.txt'
                
                if axis_file.exists():
                    print(f"    Processing: axis.txt (direct)")
                    processed_count += 1
                    
                    result = process_axis_file(axis_file)
                    
                    if result is not None:
                        a1, b1, a2, b2 = result
                        fit_output = ys_folder / 'fit2.csv'
                        save_fit_results(fit_output, a1, b1, a2, b2)
                        success_count += 1
                else:
                    print(f"    ⊘ No split* folders or axis.txt found")
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total axis.txt files processed: {processed_count}")
    print(f"  Successfully fitted: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()