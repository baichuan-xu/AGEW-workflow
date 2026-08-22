# Prompt: Binary Classification of Stress-Strain Graphs

## 🎯 Primary Goal

Your task is to perform a strict binary analysis on the provided image, `in.png`. You must determine if the image represents a single, complete stress-strain graph. Based on your analysis, you must output only "yes" or "no".

## ✅ "Yes" Criteria (Must meet ALL)

An image qualifies for a "yes" output if and only if it meets **both** of the following conditions:

1.  **Structural Integrity:** The image must contain **exactly one** data graph. This graph must be **complete**, meaning it clearly shows a vertical axis, a horizontal axis, their corresponding scales/ticks, and one or more data curves.
2.  **Label Content:** The axis labels must be correct. The vertical axis (y-axis) label **must contain** the substring "stress", and the horizontal axis (x-axis) label **must contain** the substring "strain".

If **any** of these conditions are not met, the output must be "no".

## 📝 Chain-of-Thought Process

1.  **First, check the structure.** Does the image contain exactly one graph? Is that graph complete with axes, scales, and curves? If the answer to either of these questions is no, immediately conclude the output is "no".
2.  **If the structure is valid, check the labels.** Does the vertical axis label contain "stress"? Does the horizontal axis label contain "strain"? If the answer to either of these questions is no, conclude the output is "no".
3.  **Only if both the structure and the labels are valid**, conclude the output is "yes".

## 💡 Few-Shot Examples

### Example 1: Valid Graph

**Input:** An image showing a single, complete graph with the y-axis labeled "True Stress (MPa)" and the x-axis labeled "Engineering Strain".
**Output:** `yes`

### Example 2: Invalid (Multiple Graphs)

**Input:** An image containing two separate graph plots side-by-side.
**Output:** `no`

### Example 3: Invalid (Incomplete Graph)

**Input:** An image of a graph that has been cropped, cutting off the vertical axis and its scale.
**Output:** `no`

### Example 4: Invalid (Incorrect Labels)

**Input:** An image of a single, complete graph with the y-axis labeled "Flow Rate (m/s)" and the x-axis labeled "Strain (%)".
**Output:** `no`

### Example 5: Invalid (No Labels)

**Input:** An image of a single, complete graph that has no text labels on its axes.
**Output:** `no`

## 🚀 Your Task

Analyze the provided `in.png` and generate the corresponding output ("yes" or "no").