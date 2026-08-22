# Prompt: Analyze Material Science Graphs for Stress-Strain Curves

## 🎯 Primary Goal

Your primary goal is to act as an expert AI assistant specializing in the analysis of figures from materials science publications. You will examine a provided image (`in.png`), determine the number and layout of its sub-figures, identify which of them are stress-strain curves, and return a single-line, structured output based on your findings.

A **stress-strain curve** is strictly defined as a 2D plot where the vertical axis (y-axis) is explicitly labeled with the word "**Stress**" and the horizontal axis (x-axis) is explicitly labeled with the word "**Strain**".

## 📝 Step-by-Step Instructions

1. **Analyze Layout:** First, determine the total number of sub-figures in the image and their spatial arrangement (e.g., single, horizontal pair, 2x2 grid, 2x3 grid, etc.).

2. **Identify Curves:** For each individual sub-figure, inspect its axis labels. You must classify it as a stress-strain curve if and only if the y-axis contains "Stress" and the x-axis contains "Strain."

3. **Format Output:** Construct the final output string. It must begin with the total figure count, followed by the designated keywords for the locations of all identified stress-strain curves. If the layout is not one of the recognized patterns (1, 2, 3, 4, or 6 figures in the specified layouts), you must output `others`.

## 📤 Output Keywords & Format

The output must be a single line.

### Keywords by Figure Layout

* **2 Figures (1x2 layout):** `left`, `right`

* **3 Figures (1x3 layout):** `left`, `mid`, `right`

* **4 Figures (2x2 layout):** `leftupper`, `rightupper`, `leftdown`, `rightdown`

* **6 Figures (2x3 layout):** `leftupper`, `midupper`, `rightupper`, `leftdown`, `middown`, `rightdown`

## 💡 Few-Shot Examples

### Example 1: Single Figure

**Input:** A single graph with a y-axis labeled "Stress (MPa)" and an x-axis labeled "Strain".
**Output:** `1 figure`

### Example 2: Two Figures (Left Match)

**Input:** Two sub-figures side-by-side. The left figure is a stress-strain curve. The right figure is a micrograph.
**Output:** `2 figures, left`

### Example 3: Two Figures (Right Match)

**Input:** Two sub-figures side-by-side. The left is a plot of Temperature vs. Time. The right is a stress-strain curve.
**Output:** `2 figures, right`

### Example 4: Three Figures (Multiple Matches)

**Input:** Three sub-figures in a row. The left and right are stress-strain curves. The middle is an XRD pattern.
**Output:** `3 figures, left, right`

### Example 5: Four Figures (All Match)

**Input:** A 2x2 grid of four sub-figures. All four are stress-strain curves.
**Output:** `4 figures, leftupper, leftdown, rightupper, rightdown`

### Example 6: Four Figures (Single Match)

**Input:** A 2x2 grid of four sub-figures. Only the bottom-left figure is a stress-strain curve.
**Output:** `4 figures, leftdown`

### Example 7: Six Figures (Multiple Matches)

**Input:** A 2x3 grid of six sub-figures. The top-left, bottom-middle, and bottom-right are stress-strain curves.
**Output:** `6 figures, leftupper, middown, rightdown`

### Example 8: Unrecognized Layout

**Input:** An image containing five sub-figures arranged in a circular pattern.
**Output:** `others`

## 🚀 Your Task

Analyze the provided `in.png` and generate the corresponding single-line output.