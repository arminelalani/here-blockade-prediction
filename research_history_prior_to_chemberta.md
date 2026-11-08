# hERG Blockade Prediction — Complete Research and Development History

**Project:** CDD203 Homework 1  
**Final selected model:** Random Forest on a 2,058-dimensional hybrid molecular representation  
**Decision threshold:** 0.49  
**Evidence sources:** eight uploaded Jupyter notebooks (01, 02, 03, 04, four 05 variants), their saved notebook outputs where present, and the observed terminal/Space results from the project conversation.

> **Evidence boundary:** A numeric result is reported as a completed experiment only when it appears in a saved notebook output or in a previously shared execution screenshot. Some notebooks contain code without saved execution outputs; their cells establish intended methods, but not independently recoverable results. Results from different splits and CV designs are not directly interchangeable.

## Executive summary

We investigated the classification of compounds as hERG blockers (Y=1) or non-blockers (Y=0) from SMILES strings. Starting from 10,000 labeled molecules, we audited data quality, tested molecular fingerprints and chemical descriptors, explored classical classifiers with scaffold-aware validation, tuned RF and SVM candidates, selected a single decision threshold, and compared graph convolutional networks. RF-Hybrid was frozen before the one-time locked-set evaluation. It achieved **AUROC 0.8806, AUPRC 0.8990, sensitivity 0.7915, specificity 0.7922 and precision 0.8049** on 2,085 locked molecules at threshold 0.49. The final submission script was then retrained on all 10,000 supplied molecules, exported to a 12.67 MB ONNX model, and passed all model-specific checks in `TestModel.py`. The Gradio application was deployed to a Hugging Face Space and was reported working; its public URL should be copied directly from the running Space before submission.

## 1. Problem and evaluation constraints

The assignment requires `featurize(smiles)` to return a fixed-width two-dimensional numeric array and a fitted model exported to ONNX, with the last model output representing P(blocker). The submitted classifier must use a threshold other than 0.5. The official checker inspects class probabilities, batch independence, ONNX size <=50 MB, requirements, web app packaging, and a README linking to the live application. Model training is subject to a 30-minute CPU budget. Competitions emphasize sensitivity, specificity, precision under minimum counterpart constraints, ROC-AUC, and inference speed above AUROC gates. One submission uses one model and one threshold; it is not possible to submit a different threshold to each competition within one file.

## 2. Data examination and quality audit — Notebook 01

The table contains **10,000 rows**, the columns `Drug_ID`, `Drug` (SMILES), and `Y`, and **5,236 positives (52.36%) / 4,764 negatives (47.64%)**. Notebook outputs show zero missing values, **10,000 valid RDKit-parsed SMILES**, and **zero duplicate canonical SMILES**. There are **5,185 unique Bemis–Murcko scaffolds**. A two-stage group-based split yields **TRAIN 6,841**, **DEV 1,074**, **LOCKED TEST 2,085** samples, with **3,629 / 519 / 1,037 distinct scaffolds** and zero scaffold overlap across the three partitions. Positive rates are approximately 0.526/0.518/0.520.

We created a 2,048-bit Morgan fingerprint and ten interpretable physicochemical descriptors: `MolWt`, `MolLogP`, `TPSA`, `HBD`, `HBA`, `RotatableBonds`, `AromaticRings`, `FractionCSP3`, `FormalCharge`, and `NitrogenCount`. The hybrid vector concatenates both into **2,058 input features**. The descriptor audit showed blockers had higher mean MolLogP (about 4.24 vs 3.32 for non-blockers) and lower mean TPSA (about 65.6 vs 83.3). These are sample associations, not proof of mechanism or causation.

## 3. Early baselines and exploratory feature/model experiments — Notebook 01

Initial validation experiments with logistic regression yielded **Morgan AUROC ~0.791, descriptors-only ~0.743, hybrid L2 ~0.809** at a 0.5 threshold. Additional regularization and feature-selection experiments were conducted: an L1 combined model reached about **0.8345 validation AUROC** in one saved run (659 selected features), and a later validation experiment reported **0.8761 AUROC / 0.8898 AUPRC**. These were exploratory results and should not be presented as the final independent generalization estimate.

Other exploratory comparisons in Notebook 01 included RF, Extra Trees, HistGradientBoosting; Morgan radius 2 versus 3 and 1,024 versus 2,048 bits; binary versus count and feature-based fingerprints; adding a much larger RDKit descriptor panel; and L1-based feature selection. Representative saved CV results include **RF hybrid mean AUROC ~0.8609** in an early comparison, **Feature_Count mean AUROC ~0.8688** under a different evaluated design, and RF versus Extra Trees exploratory scores in the ~0.86–0.87 range. The extended 210-descriptor matrix contained **24 non-finite values and 9 constant descriptors**, highlighting additional preprocessing complexity. These explorations did not replace the final fixed Morgan-2048 + ten-descriptor pipeline.

## 4. Research pipeline and experimental design — Notebooks 02 and 03

Notebook 02 defined a standardized comparison of three representations (**Morgan**, **10 Descriptors**, **Hybrid**) with **Logistic Regression, Random Forest, Extra Trees, and RBF SVM**, using scaffold-based cross-validation and ROC-AUC selection. Earlier shared coarse screening output identified Hybrid as strongest among tested representations; the strongest reported coarse result was SVM-Hybrid mean AUROC **~0.8672** (SD ~0.0156), followed by RF-Hybrid ~0.8614 and Extra Trees-Hybrid ~0.8556.

Notebook 03 formalized a comparison between ordinary stratified CV and scaffold-aware stratified-group CV on TRAIN. Previously reported 3-fold scaffold-aware scores were **RF-Hybrid 0.8564 (SD 0.0142)**, **SVM-Hybrid 0.8553 (SD 0.0097)**, and **Extra Trees-Hybrid 0.8497 (SD 0.0126)**. Their ordinary stratified-CV scores were higher (**RF 0.8982; SVM 0.8998; Extra Trees 0.8932**), illustrating the value of scaffold-aware estimation. These are fold means with fixed hyperparameters from prior experimentation, not strictly unbiased nested-CV estimates. The copies of notebooks 02/03 uploaded here do not preserve all console outputs, so these numbers are identified as results shared during execution rather than re-extracted from notebook cell outputs.

## 5. Focused tuning and final RF/SVM comparison

A focused randomized hyperparameter search on TRAIN with scaffold-aware folds selected **RF: `n_estimators=150`, `max_depth=30`, `min_samples_leaf=1`, `max_features=0.2`, `class_weight='balanced'`**, with shared reported mean tuning AUROC ~**0.8549**. SVM selected **RBF, `C=3`, `gamma='scale'`, no class weighting**, with mean tuning AUROC ~**0.8608**. This search was part of model development and not the official final training step.

Notebook 04 explicitly reconstructs the selected sklearn estimators. On out-of-fold TRAIN predictions, pooled AUROC was **RF 0.8544**, **SVM 0.8599**. Robust threshold screening required pooled sensitivity/specificity margins and minimal fold performance rather than selecting only barely feasible thresholds. The chosen robust candidates were **RF: sensitivity objective t=0.49; specificity/precision objective t=0.51**; **SVM: sensitivity t=0.48; specificity/precision t=0.57**. These thresholds were selected using development data and are not hidden-test guarantees.

On the previously established DEV set (1,074 molecules), RF scored **AUROC 0.8663 / AUPRC 0.8742**; SVM scored **0.8860 / 0.8913**. At their sensitivity-oriented thresholds, RF (t=0.49) had sensitivity **0.7842**, specificity **0.7683**, and SVM (t=0.48) had **0.8201 / 0.7664**. The DEV set was not a pristine final test because previous exploration could have influenced decisions, so it was used for a limited check only.

The median CPU sklearn benchmark on 1,074 molecules gave **RF 0.6733 s featurization + 0.0333 s prediction = 0.7061 s (1,521 molecules/s)**, versus **SVM 0.6744 + 4.9707 = 5.6452 s (190 molecules/s)**. RF therefore had ~8× higher overall throughput *in this sklearn benchmark*. These measurements are not an ONNX benchmark and do not establish a competition speed ranking.

## 6. GCN investigation — Notebook 05 variants

We also tested learned graph representations, with two GCN message-passing layers, 16 atom features, hidden width 64 at baseline, normalized adjacency/self-loops, mean pooling, a dense classification head, Adam optimization, and scaffold-separated inner early stopping. **Two graph convolution steps enable atom representations to aggregate information from up to two graph hops**. Each outer fold used the same original TRAIN scaffold-aware partitions, keeping DEV and LOCKED aside.

| Completed GCN variant | Fold AUROCs | Mean AUROC | Notes |
|---|---|---:|---|
| GCN only, width 64 | 0.6953 / 0.6925 / 0.7259 | **0.7046** | Best epochs 12 / 13 / 7 |
| GCN + 10 descriptors | 0.7212 / 0.7387 / 0.7742 | **0.7447** | Explicitly standardized descriptors within fold training |
| GCN + Morgan 2048 + 10 descriptors, width 64 | 0.7881 / 0.8031 / 0.8297 | **0.8070** | Best epochs 2 / 2 / 3 |
| Same three-branch GCN, width 128 | 0.7947 / 0.8078 / 0.8329 | **0.8118** | Best epochs 2 / 3 / 3 |

The descriptor addition improved mean AUROC by **0.0401** over graph-only; the Morgan branch brought another **0.0623**; increasing width to 128 added **0.0048**. Importantly, this sequential comparison does **not** establish that the graph branch supplies independent predictive value. An ablation of the neural fusion architecture without the graph branch was discussed but **not executed**. Likewise, a deeper head (128→64) and three-layer GCN were ideas, **not completed experiments**.

Loss diagnostics were performed for the two-layer GCN. Its inner validation stopping losses reached minima around epochs 12, 13 and 7; fold 3 showed an increase afterward. Training loss was not always below early-stopping loss (dropout and differing calculation conditions affect comparisons). These curves did not justify simply increasing all epoch budgets. GCN was retained as a documented exploratory comparison rather than an ONNX submission candidate; its compatibility with the teacher's fixed 2D featurization interface was not established.

## 7. Precommitted model choice, one-time locked test, and limitations

After considering quality, throughput, simplicity, and deployment, we selected **RF-Hybrid** and fixed **threshold 0.49**, with a freeze configuration stored under `research_outputs/final_decision/final_model_freeze.json` in the user's workspace. The training set was expanded to **TRAIN+DEV = 7,915 molecules**; the **2,085** locked samples were then evaluated once. The notebook screenshot and recorded output are:

| Locked-set metric | Value |
|---|---:|
| AUROC | **0.8806** |
| AUPRC | **0.8990** |
| Sensitivity | **0.7915** |
| Specificity | **0.7922** |
| Precision | **0.8049** |
| Accuracy / Balanced accuracy | **0.7918 / 0.7919** |
| TP / TN / FP / FN | **858 / 793 / 208 / 226** |
| Training time for 7,915 molecules | **9.47 seconds** |

No further hyperparameter or threshold changes were made in response to the locked-test score. It does **not** guarantee performance on the professor's unknown independent evaluation set. Furthermore, repeated exploratory decisions before the freeze make the development process less rigorous than a fully nested, prespecified experimental design.

## 8. Final artifact, ONNX and web app

The final standalone sklearn script `Part1_RF_HYBRID_FINAL.py` was executed in the class environment and trained on **all 10,000 provided molecules** in **14.68 seconds**. It exported `Part1_Armineh-Rezagholilalani.onnx` (about **12.67 MB**, under the 50 MB limit). The official `TestModel.py` passed NAME, threshold (0.49, not 0.5), `(6, 2058)` finite feature matrix, ONNX size, single input, valid probabilities, and consistent batch-versus-single predictions. ONNX is a portable *model representation*, not a SMILES parser or programming language: `app.py` still uses the identical RDKit featurization before calling ONNX Runtime.

The Gradio app's local server was initially inaccessible through the UCSF Jupyter proxy; `127.0.0.1` referred to the remote server. Local functional testing nevertheless matched the official test: **`CCO` → P(blocker) 0.101647, predicted non-blocker** at 0.49. The submission archive `Part 2.zip` was created with the app code, training/featurization script, ONNX model, and `requirements.txt`; `TestModel.py` verified that the ONNX bytes in the archive are identical to the submission model.

The Hugging Face Gradio Space `herg-blockade-predictor` was subsequently deployed. During deployment, runtime errors from absent `rdkit` and from placeholder `NAME` (`GROUP-MEMBER-NAMES-HERE`) were corrected by installing the exact allowed RDKit dependency and setting the published script's model name to `Armineh-Rezagholilalani`. The user reported that the web app then worked. The actual final Space URL and availability should be verified by opening its public App page before submitting the README.

**Important packaging check:** `requirements.txt` used for the *class checker* must contain only the instructor-approved exact pins. The dependency configuration for a hosted Space may need to be maintained separately if its runtime has different constraints; any deployed changes to app code after making `Part 2.zip` should be reflected in that archive before final delivery.

## 9. What was done versus not done

**Completed:** data audit; scaffold-disjoint splits; Morgan and descriptor engineering; multiple baseline and feature/algorithm comparisons; scaffold-aware CV; RF/SVM tuning and threshold screening; DEV check; sklearn speed benchmark; four reported GCN variants and loss diagnostics; model+threshold freeze; one-time locked evaluation; all-data final RF training; ONNX export; official model checker; app functional test; ZIP identity check; published Space reported working.

**Not completed / not demonstrated:** fully nested unbiased model-selection benchmarking; graph-branch ablation; deeper GCN head and 3-layer graph tests; GCN conversion to the professor's ONNX interface; ONNX-based competitive inference speed test; proof of final deployed app accessibility by an unauthenticated visitor; true external hidden-test metrics; final README checker PASS (as of last screenshot, README was the remaining FAIL). None of these should be reported as performed.

## 10. Final submission checklist

- [x] Frozen RF architecture and threshold
- [x] Final all-data ONNX exported and tested
- [x] Exact ONNX copied into `Part 2.zip`
- [x] App inference checked against the official test
- [x] Hosted Hugging Face app reported functional
- [ ] Copy verified *live Space URL* into submission `README.md`
- [ ] Ensure hosted Space remains public and serves an example in an external browser
- [ ] If needed, update ZIP with the latest deployed `app.py` and approved dependencies
- [ ] Run `python TestModel.py Part1_RF_HYBRID_FINAL.py` again and confirm **every line PASS**
- [ ] Rename the Python submission file to the precise naming format required by the assignment, if different from the working name

## Reproducibility and interpretive notes

All numbers should be labeled by evaluation protocol. Scaffold-aware CV fold means, pooled out-of-fold AUROC, reused DEV performance, one-time LOCKED performance, and the professor's inaccessible hidden test are different quantities. The 0.8806 locked AUROC specifically evaluates the model trained on 7,915 samples; it is **not** a fresh held-out score for the final 10,000-sample ONNX. Similarly, 14.68-second all-data training time does not include every experimental search or production infrastructure step. The 12.67 MB ONNX size is the relevant official artifact-size measurement. In a biomedical setting, hERG classification is a research screening predictor, not a substitute for electrophysiological measurement or drug safety assessment.
