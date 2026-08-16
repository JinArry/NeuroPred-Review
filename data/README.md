# Processed benchmark datasets

This directory contains the three sequence collections used in the review's
empirical analyses. The files were normalized into a common representation by
`code/data_preparation/prepare_dataset_ab.py`.

| Dataset | Development | Test | Positive | Negative | Total |
|---|---:|---:|---:|---:|---:|
| Dataset A (`dataset_a_predneurop`) | 3,880 | 970 | 2,425 | 2,425 | 4,850 |
| Dataset B, original (`dataset_b_neuropred_plm_original`) | 8,038 | 888 | 4,463 | 4,463 | 8,926 |
| Dataset B-standard20 (`dataset_b_neuropred_plm_standard20`) | 7,867 | 871 | 4,432 | 4,306 | 8,738 |

Dataset B-standard20 was produced by removing records containing residues
outside the 20 standard amino-acid alphabet. Its original development/test
assignment is otherwise retained.

## Files

Each dataset directory contains:

- `metadata.tsv`: one row per record, including normalized sequence, binary
  label, original split, source description, length, and alphabet status;
- `all.fasta`: all records;
- `train.fasta` and `test.fasta`: records in the published development and
  independent-test partitions; and
- `positives.fasta` and `negatives.fasta`: class-specific sequence files.

FASTA headers contain the normalized record identifier, label, and split.
Neuropeptides are encoded as label `1`, and non-neuropeptide background records
as label `0`.

## Provenance

- Dataset A was reconstructed from the files released with
  [PredNeuroP](https://github.com/xialab-ahu/PredNeuroP). Its reported positive
  and negative sources are NeuroPep and Swiss-Prot, respectively.
- Dataset B was reconstructed from the files released with
  [NeuroPred-PLM](https://github.com/ISYSLAB-HUST/NeuroPred-PLM). Its reported
  positive and negative sources are NeuroPep 2.0 and UniProt, respectively.
- Dataset B-standard20 is a deterministic filtered derivative of Dataset B.

The original train/test assignments are retained in the `split` column.
Additional training–validation partitions used during model development are
generated separately by the experiment scripts and are not embedded in these
canonical dataset directories.

## Integrity checks

SHA-256 checksums of the metadata tables:

```text
d8efa7d420e53aa195180efa5d55ea99ccd47e4af5da707c4d5234ccbd3309de  dataset_a_predneurop/metadata.tsv
e3a832abab5eba107964b49651c7e31272e850de788d1c8a0c97742244a20bbf  dataset_b_neuropred_plm_original/metadata.tsv
92d539e25f94a4ae43572c42de501fbdebc660aa1ef4de70cc1e58b1aea78b43  dataset_b_neuropred_plm_standard20/metadata.tsv
```

## Data notice

The repository's MIT License applies to original code, not automatically to
underlying sequence records obtained from external databases and method
repositories. These processed files are provided to support scholarly
reproducibility. Users should cite the original benchmark publications and
comply with the terms of the corresponding source databases and repositories.
