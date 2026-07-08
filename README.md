# REAL-ErgoSSVEP: Code and Pipeline

This repository contains the source code, data exporting pipelines, and technical validation notebooks for the paper:  
*"REAL-ErgoSSVEP: EEG and ergonomic dataset of steady-state visual evoked potentials under realistic backgrounds"* (
Submitted to *Scientific Data*).

The repository provides tools to:

1. Generate the multi-factor stimulus presentation timelines.
2. Standardize raw EEG data and subjective behavioral metadata into the international **BIDS (Brain Imaging Data
   Structure)** standard.
3. Perform technical validation, including signal quality assessments (PSD/SNR), subjective fatigue cross-validation (
   KSS vs. Alpha Band), and baseline machine learning (CCA/FBCCA) and deep learning (EEGNet) benchmarks.

---

## 📂 Repository Structure

```text
ROOT_DIR/
├── data/                              # Metadata and trial sequence specifications
│   ├── events1.tsv                     # Sequence mapping for Video Version 1 (Blocks 1-6)
│   ├── events2.tsv                    # Sequence mapping for Video Version 2 (Blocks 1-6, counterbalanced)
│   └── events3.tsv                    # Sequence mapping for Video Version 3 (Blocks 1-6, counterbalanced)
│
├── scripts/                           # Automation and data processing scripts
│   ├── generate_ssvep_video.py        # Generates the multi-factor stimulus frame coordinates and video timelines
│   ├── export_to_bids.py              # Converts raw EEG and questionnaire metadata to BIDS-compliant format
│   └── export_preprocessed_to_bids.py # Converts preprocessed EEG data to BIDS derivatives
│
├── notebooks/                         # Interactive technical validation & analysis notebooks
│   ├── group_analysis_ssvep_data_quality.ipynb  # Group-level validation of signal quality and baseline BCI classification
│   ├── analyze_ssvep.ipynb                      # Validation of signal quality
│   ├── fatigue_cross_validation.ipynb           # Statistical analysis correlating subjective KSS with EEG Alpha power
│   └── eegnet_classification.ipynb              # Deep learning benchmark using EEGNet
│
└── requirements.txt                   # List of Python dependencies
```

---

## ⚙️ Installation and Environment Setup

This project is implemented in Python. To ensure reproducibility, please install the required libraries within a virtual
environment.

### Prerequisites

* Python 3.12 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ssvep-ergonomic-dataset.git
cd ssvep-ergonomic-dataset
```

### Step 2: Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

*Key dependencies include:*

* `mne` & `mne-bids` for EEG data structure and standard BIDS formatting.
* `numpy`, `scipy`, and `pandas` for scientific computation and signal analysis.
* `scikit-learn` & `torch` for baseline classification (CCA and EEGNet).
* `matplotlib` & `seaborn` for generating validation visualizations.

---

## 🛠️ Usage Guide

This section describes how to run the data generation and BIDS standardization pipelines using the provided Python
scripts. All scripts are designed with standard CLI interfaces for ease of integration into batch-processing workflows.

---

### Generating Stimulus Demonstration Videos

The `generate_ssvep_video.py` script renders the periodic visual stimulus paradigms using the zero-jitter frame-division
math discussed in Section 3.2.2. By default, executing this script generates two exemplary paradigms used in our
experimental design and saves them to the `./media/videos/` directory:

1. **Standard Paradigm**: Black-White sine wave flicker at 10 Hz (Opaque, $512 \times 512$ resolution, 60 FPS).
2. **Ergonomic Paradigm**: Comfortable Green-Black sine wave flicker at 12 Hz (Opaque, $512 \times 512$ resolution, 60
   FPS).

To generate the demonstration videos, run:

```bash
python scripts/generate_ssvep_video.py
```

*Expected Output:*

```text
media/
└── videos/
    ├── ssvep_bw_10hz.mp4    # Standard Black-White stimulus
    └── ssvep_gb_12hz.mp4    # Ergonomic Green-Black stimulus
```

---

### Exporting Raw EEG and Metadata to BIDS

The `export_to_bids.py` script standardizes raw continuous Biosemi `.bdf` recordings and pairs them with their
randomized event marker mappings (`data/events*.tsv`) and phenotypic metadata into a validated BIDS dataset structure.

#### CLI Arguments:

* `--sub`: Subject identifier (string, e.g., `01` to `30`). Default is `'01'`.
* `--bdf_input`: Absolute path to the raw continuous input `.bdf` file.
* `--events_tsv`: Path to the corresponding trial sequence file (`data/events.tsv`, `data/events2.tsv`, or
  `data/events3.tsv`).
* `--bids_out`: Path to the output directory where the BIDS structure will be written.

#### Execution Example:

```bash
python scripts/export_to_bids.py \
    --sub 01 \
    --bdf_input /absolute/path/to/sub-01_raw.bdf \
    --events_tsv data/events.tsv \
    --bids_out /absolute/path/to/bids_dataset
```

---

### Exporting Preprocessed EEG to BIDS Derivatives

The `export_preprocessed_to_bids.py` script processes the continuous raw signals (applying standard bandpass filtering,
notch filtering, and downsampling) and exports the processed results into BIDS derivatives for benchmarking and
statistical validation.

#### CLI Arguments:

* `--sub`: Subject identifier (string, e.g., `01` to `30`). Default is `'01'`.
* `--bdf_input`: Absolute path to the input `.bdf` file.
* `--events_tsv`: Path to the corresponding trial sequence file (`data/events.tsv`, `data/events2.tsv`, or
  `data/events3.tsv`).
* `--bids_out`: Path to the output directory where the BIDS derivatives structure will be written.

#### Execution Example:

```bash
python scripts/export_preprocessed_to_bids.py \
    --sub 01 \
    --bdf_input /absolute/path/to/sub-01_preprocessed.bdf \
    --events_tsv data/events.tsv \
    --bids_out /absolute/path/to/bids_derivatives
```

---

## 📋 Data Formats & Metadata

* **Events Specifications (`data/events*.tsv`)**:
  These files specify the chronological events within each video version. Each trial is cataloged with its `onset`,
  `duration`, `frequency` (10/12/15 Hz), `transparency` (0/50), and `color` (BW/GB).

---

## 📝 License

This repository is licensed under the **MIT License**.  
The raw EEG dataset and associated metadata are shared under the **Creative Commons Attribution 4.0 International (CC BY
4.0)** license.

[//]: # (---)

[//]: # (## ✉️ Citation)

[//]: # ()

[//]: # (If you use this dataset or code in your research, please cite our paper:)

[//]: # ()

[//]: # (```bibtex)

[//]: # (@article{yourcitation2026,)

[//]: # (  title={A comprehensive EEG and subjective ergonomic evaluation dataset for steady-state visual evoked potentials under realistic backgrounds},)

[//]: # (  author={Tairan Liang, Jinke Fan, Lingfeng Zhang, Ye Tian, and Tao Hu},)

[//]: # (  journal={Scientific Data},)

[//]: # (  volume={xx},)

[//]: # (  pages={xx},)

[//]: # (  year={2026},)

[//]: # (  publisher={Nature Publishing Group})

[//]: # (})

[//]: # (```)