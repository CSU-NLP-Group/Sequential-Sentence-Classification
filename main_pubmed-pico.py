import imp
import json
import shutil
import sys

from allennlp.commands import main
import torch


config_file = "config/biobert_biaffine_pubmed-pico.jsonnet"
for i in range(10):

    serialization_dir = "saved_models/pubmed-pico_10fold/"+str(i+1)+'-th-fold'
    data_dir='datasets/pubmed-pico/10_folds/'
    overrides = json.dumps({"train_data_path":data_dir+str(i+1)+'/train.txt',
                            "validation_data_path":data_dir+str(i+1)+'/dev.txt',
                            "test_data_path":data_dir+str(i+1)+'/test.txt'
                            })
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "codes",
        "-f",
        "-o", overrides,
    ]
    main()

