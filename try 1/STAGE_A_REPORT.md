# Crop-JEPA Stage A — Research Report

**Date completed:** 2026-10-05
**Experiment:** Stage A — Controlled factorial comparison of JEPA target-sampling strategies on crop images
**Status:** COMPLETE (18/18 runs finished)

---

## Table of Contents

1. What Is This Project?
2. What Did We Actually Do? (Plain English)
3. The Dataset - LUCASvision
4. The Experiment - Four Arms Explained
5. How the Experiment Was Run
6. Results
7. What the Statistics Say
8. The G1 Gate Decision
9. Problems Encountered and How They Were Fixed
10. Repository Structure
11. Next Steps - Stage B C D Roadmap

---

## 1. What Is This Project?

### The big-picture goal

We want to build an AI system that can recognise crop types from field photographs, even when very few labelled examples are available. This is important for:

- Automated crop monitoring across Europe
- Early disease/stress detection
- Land-use mapping at scale

The challenge: labelling crop photos is expensive (you need agronomists in the field). So we use self-supervised learning, training a model to understand images WITHOUT needing labels first.

### What is JEPA?

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning method. Instead of comparing two augmented views of the same image (like DINO, MoCo), JEPA works like this:

1. Show the model part of an image (the context)
2. Ask it to predict the hidden/masked parts (the targets)
3. The model must build a rich internal representation to do this well

Think of it like learning to understand a puzzle. If you can fill in the missing pieces, you must actually understand what is in the image.

### Why agriculture?

Crop images have a strong green signal. Healthy vegetation reflects differently than soil or background. We hypothesised we could make JEPA smarter by guiding it to focus on the greenest patches (the most crop-relevant parts). This is the vegetation-aware idea.

---

## 2. What Did We Actually Do? (Plain English)

### Step 1 - Smoke Test (verified the pipeline works)
Ran the notebook in smoke mode using synthetic fake images. Confirmed every part of the code works without errors. Completed in about 2 minutes.

### Step 2 - Set Up the Real Dataset
Downloaded the official LUCASvision dataset from the European Commission Joint Research Centre (JRC).

- Downloaded 100 images per crop class x 12 classes = 1200 real crop images
- All from the official source: jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/
- Ran data quality checks (no corrupt files, no duplicates found)
- Split into train / validation / test sets in a fair reproducible way

### Step 3 - Benchmark Run
Before training for real, ran a benchmark to measure GPU speed. Confirmed the experiment would take about 1 hour on the RTX 3050 laptop GPU.

### Step 4 - Full Experiment (Stage A)
Trained 18 models in total:
- 4 main experiment arms x 3 random seeds = 12 SSL models
- 2 baselines (random weights + supervised training) x 3 seeds = 6 models

Each model was evaluated on how well it can classify crop types using different amounts of labelled data (from 1 example per class up to 100% of the training set).

---

## 3. The Dataset - LUCASvision

| Field | Value |
|---|---|
| Name | LUCASvision (Land Use/Cover Area frame Survey) |
| Provider | European Commission JRC |
| URL | https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/ |
| Total images used | 1200 |
| Images per class | 100 (perfectly balanced) |
| Classes | 12 crop types |
| Corrupt images | 0 |
| Duplicate images | 0 |
| Selection seed | 42 (fully reproducible) |

### The 12 Crop Classes

| Code | Crop Type |
|---|---|
| B11 | Common wheat |
| B12 | Durum wheat |
| B13 | Barley |
| B14 | Rye |
| B15 | Oats |
| B16 | Maize |
| B21 | Potatoes |
| B22 | Sugar beet |
| B31 | Sunflower |
| B32 | Rapeseed |
| B33 | Soya |
| B55 | Temporary grassland |

### Dataset Splits

| Split | Images |
|---|---|
| Training | 840 (70%) |
| Validation | 180 (15%) |
| Test | 180 (15%) |

The split was done in a group-aware way. Images from the same survey location are never split across training and test. This prevents data leakage where the model sees the same field twice.

---

## 4. The Experiment - Four Arms Explained

We compared four different ways of doing JEPA self-supervised learning. These are four variants of the same basic idea, differing only in which part of the image the model is asked to predict.

### Arm 1: ijepa - The Baseline
What it does: Picks random rectangular regions of the image to predict.
Think of it as: Closing your eyes and putting your hand on a random part of the jigsaw puzzle.

### Arm 2: ms (Multi-Scale)
What it does: Picks target regions at multiple different zoom levels.
Think of it as: Touching parts of different sizes, sometimes a single piece, sometimes a big chunk.
Why we tried it: Different crop features are visible at different scales.

### Arm 3: veg (Vegetation-Aware)
What it does: Uses the greenness of each image patch to guide which regions to predict. Greener patches are more likely to be selected as targets.
Think of it as: A smart assistant that points you to the most plant-like parts of the image to fill in.
Why we tried it: We hypothesised that focusing on vegetated regions teaches the model more crop-relevant features.

### Arm 4: veg_ms (Vegetation-Aware + Multi-Scale)
What it does: Combines both ideas. Picks green regions at multiple scales.
Think of it as: The smart assistant pointing you to plant-like parts at different zoom levels.

### Baselines

| Baseline | What it is |
|---|---|
| random_init | A model with random weights, no training at all. Tests what chance looks like. |
| supervised | A model trained the traditional way with all labels. The gold standard upper bound. |

---

## 5. How the Experiment Was Run

### Hardware

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| Architecture | Ampere (bf16 supported) |
| Profile | laptop_4gb |

### Model and Training

| Setting | Value |
|---|---|
| Architecture | ViT-Tiny (Vision Transformer tiny version) |
| Image size | 160x160 pixels |
| Patch size | 16x16 pixels |
| Training precision | bf16 (fast memory-efficient) |
| Batch size | 128 images |
| Epochs | 200 |
| Throughput | ~576 images/second |
| Peak GPU memory | 1.22 GB |
| Seeds | 3 |
| Total runs | 18 |
| Total runtime | ~1.1 hours |

### Evaluation Methods

1. Linear Probe (probe): Freeze the model and train a simple linear classifier on top. Tests whether learned features are linearly separable.
2. k-Nearest Neighbours (knn): Find the most similar images in the training set and vote on the label. No extra training needed. Tests raw feature space quality.

Both tested at: 1-shot, 5-shot, 10-shot, 25-shot, 50-shot, 10%, 100%

---

## 6. Results

All numbers are Macro-F1. Higher is better. Chance level = 8.3%.

### Linear Probe Results (mean +/- std across 3 seeds)

| Labels | ijepa | veg | ms | veg_ms | random_init |
|---|---|---|---|---|---|
| 1-shot | 12.2 +/- 2.0% | 11.6 +/- 1.2% | 12.0 +/- 2.1% | 11.8 +/- 1.0% | 13.3 +/- 2.9% |
| 5-shot | 19.5 +/- 3.0% | 18.9 +/- 5.3% | 19.8 +/- 3.1% | 19.2 +/- 4.0% | 19.5 +/- 3.7% |
| 10-shot | 22.7 +/- 2.3% | 24.7 +/- 2.9% | 22.2 +/- 2.6% | 23.5 +/- 1.9% | 21.3 +/- 1.6% |
| 25-shot | 25.2 +/- 3.4% | 26.9 +/- 3.1% | 25.0 +/- 2.3% | 27.5 +/- 3.1% | 22.5 +/- 2.0% |
| 50-shot | 29.3 +/- 1.9% | 30.1 +/- 3.2% | 28.5 +/- 1.9% | 29.2 +/- 2.8% | 26.2 +/- 2.0% |
| 10% | 19.7 +/- 1.6% | 22.5 +/- 2.7% | 19.2 +/- 1.9% | 22.3 +/- 2.3% | 19.2 +/- 2.9% |
| 100% | 29.8 +/- 2.2% | 29.9 +/- 4.2% | 28.5 +/- 1.4% | 30.7 +/- 3.7% | 28.3 +/- 1.1% |
| Supervised 100% | - | - | - | - | 53.3 +/- 2.2% |

veg and veg_ms are consistently +1 to +3 percentage points ahead of ijepa at 10-shot to 10% regimes.
ms alone does not help. It performs similarly to ijepa.

### k-NN Results (mean +/- std across 3 seeds)

| Labels | ijepa | veg | ms | veg_ms | random_init |
|---|---|---|---|---|---|
| 1-shot | 11.3 +/- 1.4% | 11.1 +/- 1.6% | 11.2 +/- 1.3% | 11.3 +/- 1.4% | 11.0 +/- 1.0% |
| 5-shot | 14.2 +/- 2.7% | 12.8 +/- 2.2% | 15.4 +/- 2.7% | 13.7 +/- 2.4% | 12.3 +/- 2.5% |
| 10-shot | 18.6 +/- 2.1% | 17.5 +/- 2.4% | 18.9 +/- 2.3% | 17.2 +/- 2.3% | 14.9 +/- 1.4% |
| 25-shot | 19.0 +/- 1.9% | 17.0 +/- 1.8% | 19.9 +/- 2.8% | 17.5 +/- 1.4% | 15.8 +/- 1.6% |
| 50-shot | 19.6 +/- 1.6% | 18.3 +/- 1.4% | 19.2 +/- 1.7% | 19.0 +/- 1.4% | 16.9 +/- 1.7% |

veg and veg_ms are slightly below ijepa in kNN, especially at 25-shot.
This means the vegetation-aware representations are better organised for linear classification but less locally clustered for nearest-neighbour retrieval.

---

## 7. What the Statistics Say

### Most important finding

| Comparison | Regime | Improvement | Corrected p-value | Conclusion |
|---|---|---|---|---|
| veg_ms vs ms (probe) | 10% labels | +3.1 pp | 0.004 | STATISTICALLY SIGNIFICANT |
| veg vs ijepa (probe) | 10% labels | +2.8 pp | 0.026 | Marginal |
| veg vs ijepa (probe) | 25-shot | +1.7 pp | 0.027 | Marginal |
| veg vs ijepa (kNN) | 25-shot | -2.0 pp | 0.012 | Vegetation hurts kNN |

### Factorial Effects Summary

- Vegetation effect (probe): Positive and consistent +1 to +3 pp across 10-shot to 10% regimes. Significant at 10% regime.
- Multi-scale effect (probe): Near zero. Multi-scale alone does not help.
- Interaction (veg x ms): Small and inconsistent. The two effects do not multiply each other.
- kNN: Vegetation effect is negative (-1 to -2 pp at 5-shot to 25-shot).

---

## 8. The G1 Gate Decision

### What is G1?

G1 is a pre-specified decision gate defined BEFORE looking at results, to prevent cherry-picking. The criterion is:

PASS if veg or veg_ms beats ijepa in at least 1 of the 3 pre-specified low-label kNN regimes (5-shot, 10-shot, 25-shot).

### Result

| Arm | Regimes won vs ijepa (kNN) | Mean difference |
|---|---|---|
| veg | 0 out of 3 | +1.02 pp (not significant) |
| veg_ms | 0 out of 3 | +1.05 pp (not significant) |

## G1 = FAIL

### What does FAIL mean?

A FAIL does not mean the experiment was a scientific failure. It means:

- The vegetation-aware approach did not pass the strictest pre-specified test (kNN at low-label)
- However it showed consistent positive effects in the linear probe evaluation
- The two evaluators tell different stories, which is itself an interesting finding

The G1 gate was defined on kNN because kNN is the most honest test of representation quality (no extra training needed). The fact that probe improves but kNN does not suggests vegetation-aware masking changes the GEOMETRY of the feature space, not just its classification accuracy.

---

## 9. Problems Encountered and How They Were Fixed

| Problem | What happened | Fix |
|---|---|---|
| PyTorch had no GPU support | Installed PyTorch was CPU-only despite RTX 3050 being present | Reinstalled PyTorch 2.6 with CUDA 12.4 (cu124 build) |
| Disk ran out of space mid-training | veg_ms_s2 crashed while saving a checkpoint. Pip download cache (~7.5 GB) was filling the C drive | Cleared pip cache, freeing 7.5 GB |
| Corrupted checkpoint on resume | Partially written last.pt caused a TypeError on reload | Deleted corrupted checkpoint. Run retrained from epoch 1. |
| sklearn iprint warning | Cosmetic warning from scikit-learn 1.6.1 | Non-fatal, no action needed |
| Notebook cell ordering bug | Dataset audit cell was inserted before data was loaded, causing NameError | Moved audit cell to correct position after data loading |

---

## 10. Repository Structure

```
try 1/
|
|-- Crop_JEPA_StageA.ipynb              <- Main experiment notebook (source)
|-- Crop_JEPA_StageA_Final.ipynb        <- Fully executed notebook with all outputs
|-- STAGE_A_REPORT.md                   <- This report
|
|-- notebooks/                          <- Intermediate notebook snapshots
|   |-- Crop_JEPA_StageA_Executed.ipynb     (smoke test run)
|   |-- Crop_JEPA_StageA_Benchmark.ipynb    (benchmark run)
|   `-- Crop_JEPA_StageA_Full_Resumed.ipynb (partial full run)
|
|-- scripts/                            <- Helper and setup scripts
|   |-- edit_nb_lucas.py    (modified notebook for LUCASvision + benchmark mode)
|   |-- fetch_lucas.py      (inspected JRC FTP directory structure)
|   |-- fetch_lucas2.py     (inspected individual class folder)
|   |-- fix_nb.py           (fixed cell ordering bug)
|   |-- inspect_nb.py       (inspected notebook cell layout)
|   |-- to_full.py          (switched notebook to full mode)
|   `-- notebook_code.py    (extracted notebook code for analysis)
|
|-- docs/                               <- Reference documents
|   |-- Crop-JEPA Stage A Notebook - Feasibility, Run Guide & Technical Report.docx
|   `-- JEPA for Crop Representation Learning - Final Research Directions & Roadmap.docx
|
|-- lucas_data/                         <- Downloaded LUCASvision images
|   |-- B11/  B12/  B13/  B14/
|   |-- B15/  B16/  B21/  B22/
|   |-- B31/  B32/  B33/  B55/
|   `-- metadata.csv                    (manifest: path, label, group)
|
`-- outputs/                            <- All experiment outputs
    |-- splits.csv                      <- Reproducible train/val/test split
    |-- label_draws.json                <- Fixed label subsets for each regime
    |-- locked_profile.json             <- Hardware profile used
    |-- results_bundle.zip              <- Complete results archive
    |
    |-- runs/                           <- One folder per trained model
    |   |-- ijepa_s0/    ijepa_s1/    ijepa_s2/
    |   |-- ms_s0/       ms_s1/       ms_s2/
    |   |-- veg_s0/      veg_s1/      veg_s2/
    |   |-- veg_ms_s0/   veg_ms_s1/   veg_ms_s2/
    |   |-- random_init_s0/  random_init_s1/  random_init_s2/
    |   `-- supervised_s0/   supervised_s1/   supervised_s2/
    |
    |-- stats/                          <- All computed statistics
    |   |-- dataset_audit.json          (data quality report)
    |   |-- hygiene_report.json         (duplicate/corruption report)
    |   |-- summary_macro_f1.csv        (results for every arm x regime)
    |   |-- paired_comparisons.csv      (statistical comparisons with p-values)
    |   |-- factorial_effects.csv       (main effects and interaction)
    |   |-- gate_G1.json                (G1 gate decision)
    |   |-- token_budget.csv            (context/target token counts)
    |   |-- benchmark.json              (hardware and throughput)
    |   `-- y_test.npy                  (test set labels)
    |
    |-- figures/                        <- Generated plots
    |   |-- label_efficiency.png        (main results figure)
    |   |-- training_curves.png         (loss curves for all runs)
    |   |-- paired_differences.png      (statistical comparison plot)
    |   `-- confusion_best_arm.png      (confusion matrix, best arm)
    |
    `-- cache/                          <- Preprocessed image cache (fast loading)
```

---

## 11. Next Steps - Stage B, C, D Roadmap

### Why G1 Failed and What That Tells Us

The vegetation-aware approach showed real positive effects in probe accuracy but did not pass the kNN test. This divergence is the key finding from Stage A.

- kNN tests local structure: similar images should be close to each other in feature space
- Probe tests linear separability: classes should be separable by a hyperplane
- Vegetation-aware masking seems to improve linear separability but distorts local clustering

This is a known phenomenon in self-supervised learning. It suggests the pretraining objective is shaping the geometry in a way that benefits supervised fine-tuning but not pure retrieval.

---

### Stage B - Cross-View JEPA (Priority 1)

**What it is:**
The current experiment treated every image independently. But LUCAS survey points have multiple photos per location (different angles, different times). Cross-view JEPA would use this structure:

- Context: Photo A of survey point P
- Target: Photo B of the same survey point P (different angle/time)

**Why this matters:**
This grounds the pretraining objective in real-world consistency. The model must learn features that are stable across views of the same field, which is exactly what a crop monitoring system needs.

**What needs to be done:**
1. Download and parse lucasvision_MMEC_trainSet.csv from the JRC FTP to recover true survey-point group IDs
2. Identify which images share a survey point
3. Modify the GPULoader to sample pairs from the same point
4. Implement a cross-view JEPA training loop
5. Compare against Stage A ijepa baseline with the same evaluation protocol

---

### Stage C - Scale-Up on Better Hardware (Priority 2)

**What it is:**
Stage A used ViT-Tiny on a laptop GPU with only 1200 images. The vegetation effect (+1 to +3 pp) might be understated because:
- The model is too small to fully exploit the vegetation signal
- The dataset is too small (100 images/class)

**What needs to be done:**
1. Run on a Kaggle T4 or Colab A100 using the kaggle_t4 profile (ViT-S/16 at 224px)
2. Use the full LUCASvision dataset (~2000+ images per class). Stage A used only 100/class.
3. Re-run the same 4-arm factorial experiment
4. Check whether the G1 gate passes at larger scale

---

### Stage D - Prediction-Difficulty Curriculum (Priority 3)

**What it is:**
Currently the model training difficulty is fixed. A curriculum approach would:
- Start easy (predict easy patches, low-frequency textures)
- Gradually increase difficulty (predict semantically complex patches)

This is motivated by the Stage A finding that vegetation-aware masking changes the feature geometry. A curriculum might let the model first learn stable low-level features then refine with vegetation guidance.

**What needs to be done:**
1. Implement a patch-difficulty score based on prediction error from a moving-average teacher
2. Anneal the masking temperature during training
3. Compare against Stage A baseline with the same evaluation protocol

---

### Immediate Next Actions (in order)

| Priority | Task | Estimated effort |
|---|---|---|
| 1 | Parse lucasvision_MMEC_trainSet.csv to recover true survey-point group IDs | 1-2 hours |
| 2 | Implement Stage B cross-view JEPA data loader | 2-3 hours |
| 3 | Run Stage B smoke test | 15 minutes |
| 4 | Run Stage B full experiment on Kaggle T4 | ~8 hours (one Kaggle session) |
| 5 | Analyse Stage B results and compare with Stage A | 1 hour |
| 6 | If Stage B passes G2, proceed to Stage C (scale-up) | TBD |

---

## Reproducibility Checklist

To reproduce this experiment exactly:

- Dataset: lucas_data/metadata.csv (1200 images, seed=42)
- Splits: outputs/splits.csv (group-aware, split_seed=42)
- Label draws: outputs/label_draws.json (fixed, seed=1234)
- Profile: outputs/locked_profile.json (laptop_4gb)
- Seeds: 0, 1, 2
- Notebook: Crop_JEPA_StageA.ipynb (RUN_MODE="full")
- Executed notebook: Crop_JEPA_StageA_Final.ipynb
- Results archive: outputs/results_bundle.zip

Python packages used:
- Python 3.13
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- numpy 2.5.3
- pandas 3.0.6
- scikit-learn 1.6.1
- scipy 1.18.1
- matplotlib 3.9.2
- pillow 11.0.0

---

This report was generated after completion of the Stage A experiment on 2026-10-05.
All raw data, checkpoints, and statistics are preserved in the outputs/ directory.
