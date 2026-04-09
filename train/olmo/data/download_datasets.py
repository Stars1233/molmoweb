#!/usr/bin/env python3
"""Download all MolmoWeb datasets from HuggingFace."""

from olmo.data.web_datasets import (
    MolmoWebSyntheticGround,
    MolmoWebSyntheticQA,
    MolmoWebSyntheticTrajs,
    MolmoWebHumanTrajs,
    MolmoWebSyntheticSkills,
    MolmoWebHumanSkills,
)

if __name__ == "__main__":
    MolmoWebSyntheticGround.download()
    MolmoWebSyntheticQA.download()
    MolmoWebSyntheticTrajs.download()
    MolmoWebHumanTrajs.download()
    MolmoWebSyntheticSkills.download()
    MolmoWebHumanSkills.download()
