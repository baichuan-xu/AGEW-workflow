# Prompt: Extract Key Data Points from Stress-Strain Curves

## 🎯 Primary Goal

Your task is to analyze the provided image (`in.png`), which contains one or more stress-strain curves from a materials science paper. For each distinct curve, you must identify two key points: the **rightmost endpoint** (fracture point) and the **topmost point** (peak stress). You will then estimate their coordinates and format the results into a single CSV output.

## Key Point Definitions

1.  **Rightmost Endpoint (x1, y1):** The point on the curve with the **maximum horizontal (x-axis) value**. This typically represents the point of fracture.
2.  **Topmost Point (x2, y2):** The point on the curve with the **maximum vertical (y-axis) value**. This represents the peak stress or Ultimate Tensile Strength (UTS). Note that this point may or may not be the same as the rightmost endpoint.

## 📤 Output Format

The output must be a valid CSV structure with the header `label,x1,y1,x2,y2`. Each row will correspond to one curve from the graph.

**Example CSV Output:**
```csv
label,x1,y1,x2,y2
"Alloy A",0.25,450,0.15,550
"Alloy B",0.35,500,0.20,600


📝 Chain-of-Thought Process
Identify Curves: First, locate the legend in the image to identify each distinct curve by its text label.

Analyze Each Curve: For each curve, perform the following estimations:

Find Rightmost Point: Trace the curve to its absolute end on the right side. Based on its position relative to the horizontal and vertical axis scales, estimate its coordinates (x1, y1).

Find Topmost Point: Scan the entire curve to find its highest peak. Based on its position relative to the horizontal and vertical axis scales, estimate its coordinates (x2, y2).

Construct CSV: Assemble the extracted information (label, x1, y1, x2, y2) into a CSV row for each curve. Combine all rows under the specified header to produce the final output.

💡 Few-Shot Examples
Example 1
Input: A graph with two curves. The x-axis (Strain) ranges from 0 to 0.5. The y-axis (Stress, MPa) ranges from 0 to 800.

A blue curve labeled "Control" ends at strain 0.3 and stress 500. Its peak is at strain 0.2 with stress 600.

A red curve labeled "Treated" peaks and ends at the same point: strain 0.4, stress 700.

Output:

label,x1,y1,x2,y2
"Control",0.3,500,0.2,600
"Treated",0.4,700,0.4,700


Example 2
Input: A graph with three curves. The x-axis (Strain, %) ranges from 0 to 40. The y-axis (Stress, GPa) ranges from 0 to 2.0.

A black curve labeled "Sample 1" ends at (35, 1.2). Its peak is at (20, 1.5).

A green curve labeled "Sample 2" ends at (30, 1.6). Its peak is at (25, 1.8).

An orange curve labeled "Sample 3" ends at (25, 0.8). Its peak is at (15, 1.0).

Output:

label,x1,y1,x2,y2
"Sample 1",35,1.2,20,1.5
"Sample 2",30,1.6,25,1.8
"Sample 3",25,0.8,15,1.0


🚀 Your Task
Analyze the provided in.png and generate the corresponding CSV output with the header label,x1,y1,x2,y2.