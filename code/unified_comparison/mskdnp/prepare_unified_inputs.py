#!/usr/bin/env python3
"""Materialize the fixed leakage-corrected unified splits as MSKDNP CSV inputs."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--splits",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_csv(a.metadata,sep="\t")
    test=metadata.loc[metadata.split.eq("test"),["seq_id","sequence","label"]].rename(columns={"sequence":"seq"}).reset_index(drop=True)
    assert len(test)==871 and test.seq_id.nunique()==871
    test.to_csv(a.output/"test.csv",index=False)
    for split_file in sorted(a.splits.glob("seed_*.tsv")):
        split=pd.read_csv(split_file,sep="\t"); seed=int(split.seed.iloc[0]); out=a.output/f"seed_{seed}"; out.mkdir(exist_ok=True)
        for partition, expected in (("train",7069),("validation",786)):
            frame=split.loc[split.partition.eq(partition),["seq_id","sequence","label"]].rename(columns={"sequence":"seq"}).reset_index(drop=True)
            assert len(frame)==expected and frame.seq_id.nunique()==expected
            frame.to_csv(out/f"{partition}.csv",index=False)
        train=set(pd.read_csv(out/"train.csv").seq); validation=set(pd.read_csv(out/"validation.csv").seq)
        assert not train & validation and not train & set(test.seq)
    print(f"prepared five fixed seeds and {len(test)} test records")


if __name__=="__main__": main()
