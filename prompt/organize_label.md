# Prompt: Augment Stress-Strain Data with Material and Process Information

## 🎯 Primary Goal

You are to act as an expert in materials science, with deep knowledge of material names, fabrication processes, stress-strain curve analysis, and literature analysis. Your task is to analyze two input files (`Fig_out.csv` and `Fig_material.txt`) to create a new, augmented CSV file. This new file will map every data point to its correct material name **from the `Fig_material.txt` file** and, if available, its specific processing condition.

## 📂 Input Files

1.  **`Fig_out.csv`**: A CSV file containing raw stress-strain data points. It has three columns: `x_new`, `y_new`, and `label`. The `label` column is a critical clue that identifies the curve each point belongs to (e.g., "As-cast", "Alloy A", "Treated").
2.  **`Fig_material.txt`**: A text file containing the **ground truth material name(s)** for the entire dataset (e.g., "Mg-Zn-Y alloy" or "Ti-6Al-4V,Ti-5553"). This file is the *only* valid source for the `material_name` column.

## 📤 Output Format

The output must be a single, valid CSV file with exactly four columns and a header row: `x_new,y_new,material_name,material_process`. The number of data rows must match the input `Fig_out.csv`.

**Column Rules:**
* `x_new`: Copied directly from the input.
* `y_new`: Copied directly from the input.
* `material_name`: **This column cannot be empty.** The value **must be one of the exact material names** listed in the `Fig_material.txt` file. Each data point (row) can only belong to **one** material.
* `material_process`: **This column can be empty.** You must infer the process (e.g., "As-cast", "Annealed", "Forged") from the `label` column of the input file. If no process is mentioned in the label, this field must be left blank.

## 📝 Chain-of-Thought Process

1.  **Read Material(s):** First, read `Fig_material.txt` to create a list of all valid, "ground truth" material names.
2.  **Process CSV Row-by-Row:** Read the `Fig_out.csv` file, starting from the first data row.
3.  **Copy Data:** For each row, copy the `x_new` and `y_new` values directly to the output.
4.  **Assign `material_name`:**
    * Read the `label` for the current row (e.g., "Ti-6Al-4V - Annealed").
    * If the list from `Fig_material.txt` contains only **one** material (e.g., "Ti-6Al-4V"), assign that single name to the `material_name` column.
    * If the list contains **multiple** materials (e.g., "Ti-6Al-4V", "Ti-5553"), you must find which material from the list best matches the `label`. For a `label` of "Ti-6Al-4V - Annealed", the correct match is "Ti-6Al-4V". Assign this **exact name from `Fig_material.txt`** to the `material_name` column.
5.  **Extract `material_process`:**
    * Analyze the `label` (e.g., "Ti-6Al-4V - Annealed") again.
    * Identify any keywords related to material processing (e.g., "Annealed", "As-cast", "Homogenized", "Forged", "Aged", "Rolled", "Extruded").
    * If a process keyword is found (and it's not the material name itself), extract it and place it in the `material_process` column.
    * If no process keyword is found in the `label` (e.g., the label is just "Alloy A"), leave the `material_process` column empty.
6.  **Construct Output Row:** Assemble the new 4-column row. Repeat for all data rows.
7.  **Finalize CSV:** Combine all generated rows under the header `x_new,y_new,material_name,material_process`.

## 💡 Few-Shot Examples

### Example 1: Single Material, Multiple Processes

* **Input `Fig_material.txt`:** `Mg-Zn-Y alloy`
* **Input `Fig_out.csv` (partial):**
    ```csv
    x_new,y_new,label
    0,0,"As-cast"
    0.01,50,"As-cast"
    0,0,"Homogenized"
    0.01,60,"Homogenized"
    ```
* **Expected Output (partial):**
    ```csv
    x_new,y_new,material_name,material_process
    0,0,"Mg-Zn-Y alloy","As-cast"
    0.01,50,"Mg-Zn-Y alloy","As-cast"
    0,0,"Mg-Zn-Y alloy","Homogenized"
    0.01,60,"Mg-Zn-Y alloy","Homogenized"
    ```
    *(Reasoning: The `material_name` is always "Mg-Zn-Y alloy" as it's the only one in the .txt file. The process is extracted from the label.)*

### Example 2: Multiple Materials, No Process

* **Input `Fig_material.txt`:** `Alloy A, Alloy B`
* **Input `Fig_out.csv` (partial):**
    ```csv
    x_new,y_new,label
    0,0,"Alloy A"
    0.02,100,"Alloy A"
    0,0,"Alloy B"
    0.02,120,"Alloy B"
    ```
* **Expected Output (partial):**
    ```csv
    x_new,y_new,material_name,material_process
    0,0,"Alloy A",
    0.02,100,"Alloy A",
    0,0,"Alloy B",
    0.02,120,"Alloy B",
    ```
    *(Reasoning: The `material_name` is matched from the label to the list in the .txt file. The label contains no separate process info, so the process column is blank.)*

### Example 3: Multiple Materials, Explicit Processes (Crucial Example)

* **Input `Fig_material.txt`:** `Ti-6Al-4V, Ti-5553`
* **Input `Fig_out.csv` (partial):**
    ```csv
    x_new,y_new,label
    0,0,"Ti-6Al-4V - Annealed"
    0.01,800,"Ti-6Al-4V - Annealed"
    0,0,"Ti-5553 - Forged"
    0.01,1000,"Ti-5553 - Forged"
    ```
* **Expected Output (partial):**
    ```csv
    x_new,y_new,material_name,material_process
    0,0,"Ti-6Al-4V","Annealed"
    0.01,800,"Ti-6Al-4V","Annealed"
    0,0,"Ti-5553","Forged"
    0.01,1000,"Ti-5553","Forged"
    ```
    *(Reasoning: The `label` "Ti-6Al-4V - Annealed" is correctly mapped to the `material_name` "Ti-6Al-4V" (from Fig_material.txt) and the `material_process` "Annealed".)*

## 🚀 Your Task

Analyze the provided `Fig_out.csv` and `Fig_material.txt` files. Generate a new CSV file with the header `x_new,y_new,material_name,material_process`, populating the columns according to the logic defined above.