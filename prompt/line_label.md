# Prompt: Extract Curve Labels and Colors from a Graph

##  Primary Goal

Your task is to analyze the provided image, `in.png`. This image is confirmed to be a single, complete stress-strain curve from a scientific paper. You must identify every data curve shown in the graph's legend, extract its textual label and corresponding color, and output the results in a strict CSV format.

##  Output Format

The output must be a valid CSV structure with a header row `label,color`. Each subsequent row should contain the exact label of a curve and its corresponding color name. Use double quotes for labels that contain special characters or commas.

**Example CSV Output:**
```csv
label,color
"Sample A",red
"Control Group",blue
"T = 100 °C",green

## Chain-of-Thought Process
Locate the Legend: First, find the legend within the image. The legend is the key that associates text labels with specific lines, markers, or colors.

Iterate Through Legend Entries: Go through each entry in the legend one by one.

Extract Label and Color: For each entry, perform two actions:

Extract the exact text label associated with the curve.

Identify the visual color of the corresponding line or marker. Use common, simple color names (e.g., red, blue, green, black, purple, orange, grey, brown).

Construct CSV: For each curve found, create a CSV row. Combine all rows, including the header, into a single block of text for the final output.

## Few-Shot Examples
#Example 1: Simple Line Chart
Input: An image with a legend showing:

A red solid line labeled "Alloy 1"

A blue dashed line labeled "Alloy 2"

A green dotted line labeled "Alloy 3"

Output:

label,color
"Alloy 1",red
"Alloy 2",blue
"Alloy 3",green


#Example 2: Scatter Plot with Markers
Input: An image with a legend showing:

Black circles for "As-cast"

Red squares for "Aged 4h"

Blue triangles for "Aged 8h"

Output:

label,color
"As-cast",black
"Aged 4h",red
"Aged 8h",blue


#Example 3: Complex Labels
Input: An image with a legend showing:

An orange line for "Sample A, T = 25 °C"

A purple line for "Sample B, T = 150 °C"

Output:

label,color
"Sample A, T = 25 °C",orange
"Sample B, T = 150 °C",purple

## Your Task
Analyze the provided in.png and generate the corresponding CSV output.