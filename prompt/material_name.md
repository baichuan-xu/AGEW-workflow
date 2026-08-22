# Prompt: Extract Material Names from Experimental Data and Text

## 🎯 Primary Goal

Your primary task is to act as a materials science expert. You will analyze two files: `out_filter.csv` (containing data labels) and `fig.txt` (containing descriptive text). By correlating the information in both files, you must identify and extract the specific names of the materials whose stress-strain curves are depicted.

## 📂 Input Files

1.  **`out_filter.csv`**: A CSV file containing data points. Critically, its `label` column lists the identifiers for each curve shown in a graph (e.g., "As-cast", "0.1% Sc", "Alloy A").
2.  **`fig.txt`**: A text file containing the figure caption or related description from the original scientific paper. This text provides the necessary context to understand what the labels in the CSV file represent.

## 📤 Output Format

The output must be a single line of text containing a comma-separated list of all identified material names.

**Example Output:**
`Mg-4Zn-1Y alloy, Al-6061 alloy`

## 📝 Chain-of-Thought Process

1.  **Analyze Labels:** First, identify all the unique values from the `label` column in `out_filter.csv`. These labels are your primary clues.
2.  **Read Context:** Next, carefully read the text in `fig.txt` to understand the experiment and the materials involved.
3.  **Correlate and Identify:** Look for connections between the labels from the CSV and the description in the text. The text will explain what material(s) were tested under the conditions specified by the labels.
4.  **Extract Material Name(s):** Pinpoint the specific names of the alloys, composites, or materials being discussed. Distinguish the material's name (e.g., "Ti-6Al-4V") from its processing condition (e.g., "annealed").
5.  **Compile and Format:** Create a unique, comma-separated list of all the material names you have extracted.

## 💡 Few-Shot Examples

### Example 1: Single Material, Multiple Conditions

* **`out_filter.csv` content includes labels:** `As-cast`, `Extruded`
* **`fig.txt` content:** "Figure 3 displays the tensile stress-strain curves of the novel Mg-4Zn-1Y alloy in as-cast and extruded states."
* **Expected Output:**
    `Mg-4Zn-1Y alloy`

### Example 2: Comparing Two Different Materials

* **`out_filter.csv` content includes labels:** `Ti-6Al-4V`, `Ti-5553`
* **`fig.txt` content:** "A comparison of the mechanical properties of Ti-6Al-4V and Ti-5553 is presented. The stress-strain curves show superior strength for the Ti-5553."
* **Expected Output:**
    `Ti-6Al-4V,Ti-5553`

### Example 3: Base Material with Additives

* **`out_filter.csv` content includes labels:** `0.1wt% Zr`, `0.3wt% Zr`
* **`fig.txt` content:** "The engineering stress-strain curves of the Al-Mg-Si alloy with different Zirconium additions are shown. The sample with 0.3wt% Zr shows the highest strength."
* **Expected Output:**
    `Al-Mg-Si alloy, Al-Mg-Si-Zr alloy`

## 🚀 Your Task

Analyze the provided `out_filter.csv` and `fig.txt` files and generate the corresponding comma-separated list of material names.