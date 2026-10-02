# NeuroPred Review

Code and processed benchmark data accompanying the review and comparative
evaluation of sequence-based neuropeptide predictors.

This repository supports the empirical analyses reported in the manuscript:

1. Dataset A/B overlap and normalized Levenshtein-similarity analysis;
2. verification of selected author-released implementations;
3. unified retraining of six representative predictors; and
4. bidirectional transfer of four source-trained predictors between the two
   benchmark lineages.

Trained weights, raw source repositories, prediction outputs, and third-party
implementations are not redistributed. Obtain these resources from their
original providers and follow the applicable licences and terms of use.

## Repository structure

```text
code/
├── data_preparation/              Dataset preparation and leakage removal
├── overlap_similarity/            Exact-overlap and sequence-similarity analyses
├── implementation_verification/   Checks of released implementations
├── unified_comparison/            Shared-data retraining of six predictors
└── cross_dataset_evaluation/       Bidirectional source-to-target evaluation
data/processed/
├── dataset_a_predneurop/                  Dataset A
├── dataset_b_neuropred_plm_original/      Original Dataset B
└── dataset_b_neuropred_plm_standard20/    Dataset B restricted to AA20
requirements.txt                    Shared Python package families
```

Generated checkpoints and results are written to user-selected directories or
to ignored output directories and are not committed to the repository.

## Benchmark files and counts

The processed metadata files use one row per benchmark record and include the
normalized sequence, binary label, original train/test assignment, source, and
sequence-level audit fields.

| Dataset | Training records | Test records | Purpose |
| --- | ---: | ---: | --- |
| Dataset A | 3,880 | 970 | PredNeuroP benchmark lineage |
| Dataset B, original | 8,038 | 888 | Verification of the released NeuroPred-PLM model |
| Dataset B-standard20 | 7,867 | 871 | Dataset B after removing non-AA20 records |
| Unified development/test collection | 7,855 | 871 | Dataset B-standard20 after removing 12 training records whose four unique sequences also occurred in the fixed test set |

Dataset A and Dataset B refer to the PredNeuroP and NeuroPred-PLM benchmark
lineages, respectively. Neuropeptides are encoded as the positive class.

## Experimental protocol

### Unified comparison

- The same corrected Dataset B-standard20 development collection was used for
  all six methods.
- Five sequence-grouped, class-stratified train/validation partitions used the
  prespecified seeds `2026`, `2027`, `2028`, `2029`, and `2030`.
- Each partition contained 7,069 training and 786 validation records.
- All five trained models were evaluated on the same fixed 871-record
  independent test set. The five values are training-run results, not
  independent test-set replicates.
- Model and checkpoint selection used validation data only.
- MCC at a decision threshold of 0.5 was the prespecified primary metric.
  ACC, sensitivity, specificity, precision, F1, AUROC, and AUPRC were also
  reported.
- Model-specific training procedures were retained; a common hyperparameter
  grid or matched optimization budget was not imposed.

The reported configurations were:

| Method | Representation and implementation | Optimization | Selection |
| --- | --- | --- | --- |
| PredNeuroP | Released handcrafted-feature stacking pipeline with tenfold out-of-fold base predictions | Eight base learners and logistic-regression meta-classifier; released/default settings | Fixed configuration |
| NeuroPred-ResSE | Reconstructed terminal one-hot, DDE, and natural-vector features; training-set standardization | Adam; learning rate 1e-3; batch size 32; maximum 50 epochs | Validation MCC; patience 8 |
| NeuroPred-PLM | ESM-1 85M; upper three encoder layers fine-tuned | Adam; head/backbone learning rates 5e-4/2e-6; batch size 16; maximum 50 epochs | Validation MCC; patience 8 |
| MSKDNP | Reconstructed ESM2-650M teacher and ESM2-8M student | AdamW; teacher/student learning rates 1e-5/2e-5; maximum 15 epochs | Validation MCC; patience 3 |
| NeuroScale | Reconstructed ESM2-650M with full-backbone fine-tuning | SGD; learning rate 1e-3; batch size 1; maximum 60 epochs | Validation MCC; patience 8 |
| NeuroPred-MTCL | Frozen ESM-1 residue embeddings with the released BiLSTM-attention architecture | Adam; learning rate 1e-3; batch size 64; maximum 50 epochs | Validation accuracy; patience 8 |

### Cross-dataset evaluation

The transfer analysis was restricted to four source-trained implementations:
PredNeuroP and NeuroPred-CLQ for Dataset A to Dataset B-standard20, and
NeuroPred-PLM and MSKDNP for Dataset B to Dataset A. Source-trained models and
their original preprocessing pipelines were applied without retraining,
adaptation, calibration, or threshold adjustment.

Three target-test subsets were evaluated:

- `all_target`: the complete fixed target test set;
- `exact_novel`: excludes target sequences identical to a source-development
  sequence; and
- `strict_0_9_novel`: additionally excludes target sequences with normalized
  Levenshtein similarity of at least 0.9 to the source-development collection.

## Environment

Create a dedicated environment and install the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The benchmark spans legacy scikit-learn/TensorFlow implementations and recent
PyTorch/ESM models, so one environment may not reproduce every author release.
Use method-specific environments from the original repositories where needed.
GPU memory requirements are dominated by ESM2-650M fine-tuning. The included
scripts expose their principal paths and runtime settings through `--help`.

## Reproduction workflow

Run commands from the repository root.

### 1. Use or rebuild the processed benchmark files

The processed files used in the analyses are already included. To rebuild them,
place the original PredNeuroP and NeuroPred-PLM repositories at
`data/raw/source_repos/PredNeuroP/` and
`data/raw/source_repos/NeuroPred-PLM/`, then run:

```bash
python code/data_preparation/prepare_dataset_ab.py
```

The raw source repositories are intentionally ignored and are not redistributed.

### 2. Construct the leakage-corrected unified collection

```bash
python code/data_preparation/prepare_unified_test_deduplicated.py \
  --input data/processed/dataset_b_neuropred_plm_standard20/metadata.tsv \
  --output-dir data/processed/dataset_b_standard20_unified

python code/unified_comparison/neuroscale/make_grouped_splits.py \
  --metadata data/processed/dataset_b_standard20_unified/metadata.tsv \
  --output-dir data/processed/unified_train_validation_splits \
  --seeds 2026 2027 2028 2029 2030
```

The first command should report 7,855 development and 871 test records after
removing 12 duplicated training records. The second command writes the five
fixed split manifests and a summary of their class counts.

### 3. Reproduce the Dataset A/B overlap analysis

```bash
python code/overlap_similarity/analyze_dataset_ab_overlap_similarity.py
```

This command writes exact-overlap tables, directional similarity summaries,
and figures to `results/dataset_ab_overlap_similarity_final/`.

### 4. Run implementation-verification checks

Verification entry points are grouped by method:

```text
code/implementation_verification/mskdnp/
code/implementation_verification/neuropred_clq/
code/implementation_verification/neuropred_mtcl/
code/implementation_verification/neuropred_plm/
```

Each script requires the corresponding author-released repository or model
assets. Run an entry point with `--help` to view the required paths, batch size,
device, and output directory. NeuroPred-MTCL verification uses its originally
reported seed of 37.

### 5. Run the unified comparison

Training entry points are located under `code/unified_comparison/`:

| Method | Entry point |
| --- | --- |
| PredNeuroP | `predneurop/train_predneurop.py` |
| NeuroPred-ResSE | `neuropred_resse/train_resse.py` |
| NeuroPred-PLM | `neuropred_plm/train_neuropred_plm.py` |
| MSKDNP | `mskdnp/prepare_unified_inputs.py`, `train_teacher.py`, `export_teacher_targets.py`, and `train_distilled_student.py` |
| NeuroScale | `neuroscale/train_neuroscale.py` |
| NeuroPred-MTCL | `neuropred_mtcl/extract_esm1_t6.py` and `train_mtcl.py` |

For each method, repeat training with seeds 2026-2030 and the matching split
manifest. Pass the reported hyperparameters explicitly when they differ from a
script default. The training pipelines record test probabilities, metrics,
seeds, and, where applicable, the selected checkpoint in their output
directories.

Example for one NeuroScale run:

```bash
python code/unified_comparison/neuroscale/train_neuroscale.py \
  --metadata data/processed/dataset_b_standard20_unified/metadata.tsv \
  --split-manifest data/processed/unified_train_validation_splits/seed_2026.tsv \
  --seed 2026 \
  --epochs 60 \
  --patience 8 \
  --batch-size 1 \
  --learning-rate 1e-3 \
  --output-dir results/unified/neuroscale/seed_2026
```

### 6. Run the cross-dataset evaluation

First generate leakage-annotated target manifests:

```bash
python code/cross_dataset_evaluation/prepare_cross_dataset_manifests.py \
  --out-dir data/processed/cross_dataset_generalization
```

Then run the relevant released-model wrapper:

```text
run_predneurop.py
run_neuropred_clq.py
run_neuropred_plm.py
run_mskdnp.py
```

Each wrapper writes a TSV containing the target labels, predicted
probabilities, and novelty annotations. Score that file with the shared metric
implementation:

```bash
python code/cross_dataset_evaluation/evaluate_predictions.py \
  --predictions results/cross_dataset/predictions.tsv \
  --out-dir results/cross_dataset/metrics \
  --method METHOD_NAME \
  --direction SOURCE_TO_TARGET \
  --threshold 0.5
```

## Output and audit conventions

- Prediction tables retain `seq_id` so all metric calculations can be audited
  at the sequence level.
- Probabilities and thresholded predictions are stored separately.
- JSON files record seeds, selected epochs, validation criteria, and test
  metrics; TSV files support direct inspection and reanalysis.
- Test labels are not used for hyperparameter selection, early stopping,
  calibration, or threshold adjustment.
- The repository compares the evaluated implementations under the reported
  protocols; it does not claim exact reproduction of incompletely specified
  original training workflows.

## External resources

The included scripts wrap or reconstruct methods described in the following
author releases:

- [PredNeuroP](https://github.com/xialab-ahu/PredNeuroP)
- [NeuroPred-PLM](https://github.com/ISYSLAB-HUST/NeuroPred-PLM)
- [NeuroPred-ResSE](https://github.com/yunyunliang88/NeuroPred-ResSE)
- [NeuroScale](https://github.com/ZhangHongqi215/NeuroScale)
- [NeuroPred-MTCL](https://github.com/JinArry/NeuroPred-MTCL)

Third-party datasets, pretrained models, weights, and software remain subject
to their original licences.

## Licence

Original code in this repository is released under the MIT License. This
licence does not supersede the terms of the source databases or repositories
from which the processed benchmark records were derived.

## Citation

Citation information will be added after publication of the review.
