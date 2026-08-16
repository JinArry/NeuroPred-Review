# NeuroPred-Review

Code accompanying the review and comparative evaluation of sequence-based
neuropeptide predictors.

The repository currently contains the analysis and model-evaluation code only.
Benchmark sequences, trained weights, prediction outputs, and third-party
implementations are not redistributed. Obtain those resources from their
original providers and follow the applicable licences and terms of use.

## Repository structure

```text
code/
├── data_preparation/              # Canonical Dataset A/B and unified-test preparation
├── overlap_similarity/            # Exact-overlap and normalized edit-similarity analyses
├── implementation_verification/   # Checks of released predictor implementations
├── unified_comparison/            # Retraining under the shared experimental protocol
└── cross_dataset_evaluation/      # Bidirectional source-to-target evaluation
```

The code covers four empirical components reported in the review:

1. characterization of overlap and sequence similarity between Dataset A
   (PredNeuroP lineage) and Dataset B (NeuroPred-PLM lineage);
2. verification of selected author-released inference or training pipelines;
3. comparison of representative predictors under a common training–validation–test
   protocol; and
4. bidirectional cross-dataset evaluation of fixed source-trained predictors.

## External resources

Several scripts are wrappers around, or reimplementations informed by, the
original method releases. The relevant repositories include:

- [PredNeuroP](https://github.com/xialab-ahu/PredNeuroP)
- [NeuroPred-PLM](https://github.com/ISYSLAB-HUST/NeuroPred-PLM)
- [NeuroPred-ResSE](https://github.com/yunyunliang88/NeuroPred-ResSE)
- [NeuroScale](https://github.com/ZhangHongqi215/NeuroScale)
- [NeuroPred-MTCL](https://github.com/JinArry/NeuroPred-MTCL)

PredNeuroP and NeuroPred-PLM source datasets are expected, by default, under
`data/raw/source_repos/PredNeuroP/` and
`data/raw/source_repos/NeuroPred-PLM/`. Most model scripts instead accept the
location of an external repository, input manifest, and output directory as
command-line arguments; run a script with `--help` to inspect its interface.

## Environment

The methods span legacy machine-learning code, TensorFlow models, and recent
PyTorch/ESM models, so a single environment may not exactly reproduce every
author release. `requirements.txt` lists the shared package families used by
the included scripts. Method-specific environments should follow the original
repositories where possible.

No manuscript results are generated automatically on installation. Outputs
are written to user-specified locations or to the ignored `results/` directory.

## Reproducibility notes

- Neuropeptides are treated as the positive class.
- The main decision threshold used in the comparative analyses is 0.5.
- Unified-comparison scripts use predefined split manifests and keep the fixed
  independent test set outside model selection.
- Cross-dataset scripts apply source-trained models without target-set
  retraining, calibration, or threshold adjustment.

## Licence

Original code in this repository is released under the MIT License. Third-party
datasets, models, and software remain subject to their original licences.

## Citation

Citation information will be added after publication of the review.
