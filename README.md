# hERG Blockade Prediction from Molecular SMILES

**Molecular machine learning | Scaffold-aware validation | Random Forest | ONNX | Gradio**

A reproducible research and deployment project that predicts the probability of hERG potassium-channel blockade from a compound's SMILES string. The final deployed classifier uses **Morgan fingerprints (radius 2, 2,048 bits) + 10 RDKit molecular descriptors** with a Random Forest. Its decision threshold was fixed at **0.49** before the one-time internal locked-set evaluation.

**Live application:** [hERG Blockade Predictor (Hugging Face Space)](https://huggingface.co/spaces/arminelalani/herg-blockade-predictor)  
*Check that this link opens the running App before sharing; the public Space URL has not been independently verified in this repository package.*

> **Educational/research use only.** The predicted probability is a model output, not an experimental hERG assay result or a clinical cardiotoxicity assessment.

## Overview

The project explored several representations and machine-learning families before selecting the deployed model:

1. Data-quality audit and chemistry-driven exploratory analysis.
2. Morgan fingerprints, RDKit descriptors, and alternative molecular feature encodings.
3. Logistic Regression, Random Forest (RF), Extra Trees, and RBF-SVM comparisons.
4. Scaffold-aware cross-validation, hyperparameter search, threshold screening, development-set checks, and CPU inference benchmarking.
5. Graph convolutional network (GCN) variants and frozen ChemBERTa molecular embeddings as research alternatives.
6. Final RF-Hybrid configuration freeze, one-time locked test, all-data refit, ONNX export, and a hosted Gradio interface.

## Data and validation design

The course-provided data comprise **10,000 labeled compounds**, with **5,236 blockers (52.36%)** and **4,764 non-blockers**. An RDKit data audit found no missing or invalid SMILES, no duplicated canonical SMILES, and **5,185 distinct Bemis–Murcko scaffolds**.

For research, the dataset was partitioned using scaffold groups into **TRAIN 6,841**, **DEV 1,074**, and **LOCKED TEST 2,085** molecules, with no scaffold overlap across partitions. Scaffold-aware three-fold CV on TRAIN was used to compare candidates. The **final submission refit** used all 10,000 course-supplied molecules, **after** the configuration was chosen and internal locked-set evaluation completed.

**Data availability:** `hw1data.tab` is course-provided and intentionally not included in this public repository. To reproduce training, supply an authorized copy in `data/hw1data.tab` or alongside the training script. Do not redistribute the original course data unless permission has been granted.

## Features and final model

`featurize(smiles)` returns a `float32` matrix of shape `(n_molecules, 2058)`:

- **2048 Morgan fingerprint bits**, radius 2 (RDKit Morgan generator).
- **10 descriptors:** molecular weight, MolLogP, TPSA, hydrogen-bond donors, hydrogen-bond acceptors, rotatable bonds, aromatic rings, fraction Csp3, formal charge, and nitrogen-atom count.

Selected Random Forest settings: `n_estimators=150`, `max_depth=30`, `min_samples_leaf=1`, `max_features=0.2`, `class_weight='balanced'`, `random_state=42`. The **classification threshold is 0.49**, not 0.5.

## Experimental results

### Classical models — scaffold-aware CV on original TRAIN

| Model | Mean AUROC |
| --- | ---: |
| **RF + Morgan + 10 descriptors** | **0.8564** |
| SVM-RBF + Morgan + 10 descriptors | 0.8553 |
| Extra Trees + Morgan + 10 descriptors | 0.8497 |

These are results of a prior fixed-hyperparameter three-fold comparison, **not** fully nested unbiased model-selection estimates. A later matched RF baseline in the ChemBERTa experiment yielded **0.8549** mean AUROC; numbers from distinct experiments should not be conflated.

### GCN experiments — three scaffold-aware folds

| Architecture | Mean AUROC |
| --- | ---: |
| Two-layer GCN alone (width 64) | 0.7046 |
| GCN + 10 descriptors | 0.7447 |
| GCN + Morgan + 10 descriptors (width 64) | 0.8070 |
| GCN + Morgan + 10 descriptors (width 128) | **0.8118** |

These experiments do **not** establish that the graph branch contributed complementary information: a graph-branch ablation of the fusion model was considered but not executed. GCN was not deployed.

### Frozen pretrained ChemBERTa — matched scaffold folds

| Representation and downstream model | Mean AUROC | Mean AUPRC |
| --- | ---: | ---: |
| **RF-Hybrid (matched baseline)** | **0.8549** | **0.8670** |
| Frozen ChemBERTa embeddings + Morgan + descriptors + RF | 0.8385 | 0.8551 |
| Frozen ChemBERTa embeddings + Logistic Regression | 0.7875 | 0.7954 |

This was a **frozen-encoder** comparison, not transformer fine-tuning. Transformer embedding extraction time is **not** included in the downstream estimator fit times. The pretrained model files are not included in the repository.

### Selected RF — one-time internal locked-set test

The frozen RF-Hybrid was trained on **TRAIN + DEV = 7,915 samples**, then evaluated on the previously untouched **2,085-molecule locked test**.

| Metric | Score |
| --- | ---: |
| **AUROC** | **0.8806** |
| AUPRC | 0.8990 |
| Sensitivity at 0.49 | 0.7915 |
| Specificity at 0.49 | 0.7922 |
| Precision at 0.49 | 0.8049 |
| Accuracy | 0.7918 |
| Balanced accuracy | 0.7919 |

**Important:** These metrics describe the internal **7,915-training-molecule model**, not the final refit on all 10,000 molecules, and not the instructor's independent hidden test.

### Speed and final artifact

In an **sklearn end-to-end CPU inference benchmark** (including featurization), RF scored approximately **1,521 molecules/second**, versus **190 molecules/second** for SVM. This was **not an ONNX benchmark** and is not an official competition speed measurement.

The final RF refit on 10,000 compounds took **14.68 s** in the tested UCSF environment and exported an approximately **12.67 MB ONNX** model (assignment cap: 50 MB). The instructor's `TestModel.py` passed the model-related checks, including finite 2-D features, a single dynamic-batch ONNX input, valid probability outputs, and single-versus-batch prediction consistency. The course checker also confirmed that the `Part 2.zip` archive contained an identical copy of the ONNX model. The final README/link checker should be rerun after placing this README in the submission directory.

## Repository structure

```text
hERG-blockade-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Part1_RF_HYBRID_FINAL.py     # Training, features, ONNX export
├── app.py                       # Gradio UI and ONNX inference
├── Part1_Armineh-Rezagholilalani.onnx  # ADD manually; not in starter archive
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_hERG_Research_Pipeline.ipynb
│   ├── 03_Experimental_Design.ipynb
│   ├── 04_RF_SVM_Selection_and_Locked_Test.ipynb
│   ├── 05a_GCN_Baseline.ipynb
│   ├── 05b_GCN_Loss_Diagnostics.ipynb
│   ├── 05c_GCN_Descriptors.ipynb
│   ├── 05d_GCN_Morgan_Descriptors.ipynb
│   └── 06_ChemBERTa_Frozen_Embeddings.ipynb
├── docs/
│   └── research_history_prior_to_chemberta.md
└── data/                        # local only; ignored by Git
    └── hw1data.tab              # supplied by authorized user
```

The notebooks document **research development**. They are not required for deployed inference, and some require local data, saved intermediate outputs, or downloaded pretrained weights. This starter repository preserves their original code/results rather than claiming that every notebook runs end-to-end in a fresh environment.

## Installation and reproduction

Use **Python 3.10** and the instructor-approved pinned dependencies. For a clean local environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Add the authorized dataset, then retrain/export the final model:

```bash
python Part1_RF_HYBRID_FINAL.py
```

The generated ONNX file will be named `Part1_Armineh-Rezagholilalani.onnx`. If you only want to run the application, place the **already exported identical ONNX file** next to `app.py` and the training/featurization module. No retraining is needed for inference.

Launch locally:

```bash
python app.py
```

The Gradio UI accepts SMILES, computes the 2,058 RDKit features, runs the ONNX classifier using ONNX Runtime on CPU, then flags a compound as a predicted blocker when `P(blocker) >= 0.49`. Test input `CCO` produced approximately **0.101647**, i.e. predicted **non-blocker**, in the reported final deployment check.

To run the **instructor-provided** checker, obtain `TestModel.py` from the course (not redistributed here) and run:

```bash
python TestModel.py Part1_RF_HYBRID_FINAL.py
```

`TestModel.py` also checks for a course-specific `Part 2.zip` and a README linking to the live application; those files may be part of your **private course submission** rather than the public GitHub repository.

## ONNX and hosted app

- `skl2onnx` converts the fitted RF to ONNX (`zipmap=False`, `target_opset=12`).
- ONNX is a portable serialized model **format**, not a programming language; it does not itself parse SMILES.
- `onnxruntime` executes the classifier. `RDKit` continues to provide the exact same SMILES featurization as in training.
- `Gradio` provides the web interface; `Hugging Face Spaces` hosts and launches it. The Space should use the exact same ONNX bytes and matching Python featurizer.

## Limitations

This is not a validated cardiotoxicity risk tool. Molecular-scaffold validation reduces some structural similarity leakage but does not guarantee prospective generalization. Exploratory comparisons involved several tuning decisions, so their metrics are not fully independent model-selection estimates. Only one internal locked-set evaluation of the selected RF was reported. The GCN experiments were not converted to ONNX, and a larger neural classification head and graph-only ablation were not completed. Other pretrained models and end-to-end ONNX speed rankings remain untested.

## Attribution and sharing

Developed for UCSF CDD203 Homework 1. The dataset, assignment handout, and instructor's checker are course materials; share only with appropriate permission. Model weights and code may also be subject to course disclosure and team collaboration rules. No external licensing grant is asserted for these materials.
