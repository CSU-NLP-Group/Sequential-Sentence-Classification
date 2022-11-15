import imp
import json
import shutil
import sys

from allennlp.commands import main
import torch

config_file = "config/biobert_biaffine_pubmed-rct.jsonnet"
serialization_dir = "saved_models/pubmed-rct"
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "codes",
    "-f"
]
main()
