# Prompt: Identify Material Preparation Sentences from Scientific Text

## 🎯 Primary Goal

You are to act as an expert in materials science, with deep knowledge of material names, fabrication processes, and literature analysis. Your task is to analyze two files: `Fig.txt`, which lists target materials, and `sentence_split.csv`, which contains sentences from a scientific article. You must identify which sentences describe the **preparation process** for the target materials and tag them accordingly in a CSV output.

## 📂 Input Files

1.  **`Fig.txt`**: A text file containing a comma-separated list of target material names (e.g., "Fe-Mn-Si alloy", "Ti-6Al-4V").
2.  **`sentence_split.csv`**: A CSV file with two columns: `serial_number` and `sentence`.

## ✅ "Preparation Process" Definition

A sentence describes a "preparation process" if it details the synthesis, fabrication, or treatment of a material. Look for keywords and concepts related to:
* **Synthesis/Alloying:** `arc melting`, `casting`, `induction melting`, `powder metallurgy`, `alloyed`.
* **Mechanical Processing:** `hot rolling`, `cold rolling`, `forging`, `extrusion`, `drawing`.
* **Heat Treatment:** `annealing`, `quenching`, `tempering`, `aging`, `solution treatment`.
* **Fabrication:** `sintering`, `milling`, `deposition`, `3D printing`, `additive manufacturing`.

Sentences describing material characterization (e.g., `tested with SEM`, `XRD patterns were collected`), results, or conclusions are **NOT** preparation processes.

## 📤 Output Format

The output must be a CSV file with the header `serial_number,materials`. Only include rows for sentences that describe a preparation process for one or more target materials. The `materials` column should be a comma-separated list of the relevant material names from `Fig.txt`.

**Example Output:**
```csv
serial_number,materials
15,Fe
20,"Fe,Cu"
35,Ti-6Al-4V

📝 Chain-of-Thought Process
Identify Targets: First, read Fig.txt to create a definitive list of target material names.

Iterate Sentences: Process sentence_split.csv row by row.

Analyze Each Sentence: For each sentence: a. Determine if it describes a material preparation process based on the definition provided. b. If it is a preparation process, check if it refers to any of the target materials from your list. The connection might be direct (naming the material) or contextual. c. If both conditions are met, identify all target materials the process applies to.

Construct Output: If a sentence is a match, create a row for the output CSV with the serial_number and a comma-separated list of the identified material(s). If a sentence is not a match, ignore it completely.

Finalize CSV: Combine all generated rows into a single CSV block with the correct header.

💡 Few-Shot Examples
Example 1
Fig.txt content: Al-Mg-Si alloy, Al-Zn alloy

sentence_split.csv relevant rows:

12, The initial ingots of the Al-Mg-Si alloy were produced by induction melting.

18, Both alloys were then subjected to homogenization annealing at 500 °C for 24 hours.

25, Tensile tests were conducted at room temperature.

Expected Output:

serial_number,materials
12,Al-Mg-Si alloy
18,"Al-Mg-Si alloy,Al-Zn alloy"
(Note: Sentence 25 is ignored as it describes testing, not preparation.)

Example 2
Fig.txt content: HEA, CoCrFeNi

sentence_split.csv relevant rows:

5, The CoCrFeNi high-entropy alloy (HEA) was prepared by arc melting in a titanium-gettered argon atmosphere.

30, The microstructure of the HEA was observed using scanning electron microscopy.

Expected Output:

serial_number,materials
5,"HEA,CoCrFeNi"
(Note: Sentence 5 refers to both target materials, and sentence 30 is ignored.)

🚀 Your Task
Analyze the provided Fig.txt and sentence_split.csv files. Generate a CSV output listing the serial numbers of sentences that describe the preparation of the target materials, along with the names of those materials.