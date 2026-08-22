# Prompt: Determine Material Compositions from Text and Tables

## 🎯 Primary Goal

You are to act as an expert in materials science, with deep knowledge of alloy naming conventions, fabrication processes, and literature analysis. Your task is to determine the chemical composition (elements and their proportions) for **each material** listed in the `Fig.txt` file.

There are two scenarios for this task, based on the files you receive:

1.  **Scenario 1 (Knowledge-Only):** You receive **only `Fig.txt`**. You must use your expert knowledge of material naming conventions to deduce the composition (e.g., "Ti-6Al-4V" is Ti-based with 6% Al and 4% V).
2.  **Scenario 2 (Context-Provided):** You receive **`Fig.txt` AND `table_composition.txt`**. You **must** use the `table_composition.txt` file as the primary source of truth. This file contains tables and text describing the *exact* compositions used in the paper, which **overrides** your general knowledge.

In all cases, the sum of element proportions for each material **must equal 100**. The output must be a single JSON object.

## 📂 Input Files

* **`Fig.txt` (Required):** A text file containing a comma-separated list of target material names (e.g., "Al-Mg-Si, Ti-6Al-4V").
* **`table_composition.txt` (Optional):** An optional file containing a data table and/or descriptive text from the paper, detailing the chemical compositions (e.g., "Alloy 1 (wt.%): Mg 1.0, Si 0.5, Al Bal.").

## 📤 Output Format

The output must be a single, valid JSON object. Each key in the object is a material name from `Fig.txt`. The value for each key is another object, where keys are the chemical element symbols and values are their numerical proportions (totaling 100).

**JSON Output Example:**
```json
{
  "Material A": {
    "Al": 90,
    "Mg": 5,
    "Si": 5
  },
  "Material B": {
    "Ti": 90,
    "Al": 6,
    "V": 4
  }
}
📝 Chain-of-Thought Process
Read Material List: First, parse the list of material names from Fig.txt.

Check for Context: Determine if table_composition.txt was provided.

Analyze (Scenario 1: Knowledge-Only):

If no table is provided, iterate through each material name.

Use your materials science expertise to interpret the name.

For "Ti-6Al-4V", deduce 6% Al, 4% V.

Calculate the balance element: 100 - 6 - 4 = 90. Assign 90% to the base element, Ti.

Assign to the JSON object.

Analyze (Scenario 2: Context-Provided):

If table_composition.txt is provided, read and parse it thoroughly.

For each material in Fig.txt, find its corresponding entry in the table or text.

The table is the ground truth. If the table says "Ti-6-4 alloy (wt.%): Al 5.8, V 4.1", you must use 5.8 and 4.1, not 6 and 4.

If the table provides a "Balance" or "Bal." element (e.g., "Al: Bal."), calculate its value by subtracting all other elements from 100.

Assign these explicit values to the JSON object.

Format JSON: Construct the final JSON object with all materials as keys.

💡 Few-Shot Examples
Example 1: Scenario 1 (Knowledge-Only)
Input Fig.txt: Ti-6Al-4V, Al-Mg-Si

Input table_composition.txt: (Not provided)

Reasoning: LLM must use internal knowledge.

"Ti-6Al-4V" is a standard alloy: 6% Al, 4% V, Balance Ti.

"Al-Mg-Si" is a 6000-series alloy. A common composition is ~1.0% Mg, ~0.6% Si, Balance Al. (An expert estimation is required).

Output:

JSON

{
  "Ti-6Al-4V": {
    "Ti": 90.0,
    "Al": 6.0,
    "V": 4.0
  },
  "Al-Mg-Si": {
    "Al": 98.4,
    "Mg": 1.0,
    "Si": 0.6
  }
}
Example 2: Scenario 2 (Context-Provided)
Input Fig.txt: Alloy 1, Alloy 2

Input table_composition.txt: "The nominal compositions of the alloys are shown in Table 1. Table 1. Chemical composition of alloys (wt.%) | Alloy | Mg | Zn | Sc | Al | |---|---|---|---|---| | Alloy 1 | 2.0 | 5.0 | - | Bal. | | Alloy 2 | 2.0 | 5.0 | 0.2 | Bal. | "

Reasoning: LLM must use the table.

Alloy 1: 2.0% Mg, 5.0% Zn. Balance = 100 - 2.0 - 5.0 = 93.0% Al.

Alloy 2: 2.0% Mg, 5.0% Zn, 0.2% Sc. Balance = 100 - 2.0 - 5.0 - 0.2 = 92.8% Al.

Output:

JSON

{
  "Alloy 1": {
    "Mg": 2.0,
    "Zn": 5.0,
    "Al": 93.0
  },
  "Alloy 2": {
    "Mg": 2.0,
    "Zn": 5.0,
    "Sc": 0.2,
    "Al": 92.8
  }
}
🚀 Your Task
Analyze the provided file(s) (Fig.txt and optional table_composition.txt) and generate a single JSON object detailing the chemical composition (in 100%) for each material.