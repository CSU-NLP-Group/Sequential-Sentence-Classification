# Sequential-Sentence-Classification
code for our paper Sequential Sentence Classification with a Boundary-aware Dual Biaffine Model for Biomedical Documents

# Requirement
All the requirements are listed in the requirement.txt

# Dataset
## Pubmed-pico
This dataset is introduced by [Jin, Di, and Peter Szolovits. "PICO Element Detection in Medical Text via Long Short-Term Memory Neural Networks." Proceedings of the BioNLP 2018 workshop. 2018.](http://www.aclweb.org/anthology/W18-2308)
And it can be find in github from ["PubMed PICO Element Detection Dataset"](https://github.com/jind11/PubMed-PICO-Detection) 
## Pubmed-rct
This dataset is introduced by [Franck Dernoncourt, Ji Young Lee. PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. International Joint Conference on Natural Language Processing (IJCNLP). 2017.]([http://www.aclweb.org/anthology/W18-2308](https://arxiv.org/abs/1710.06071))
And it can be find in github from ["PubMed 200k RCT dataset"](https://github.com/Franck-Dernoncourt/pubmed-rct)

# Run
The command for runing the model on pubmed-rct dataset `allennlp train config/biobert_biaffine_pubmed-rct.jsonnet -s saved_models/pubmed-rct/ --include-package codes --force` .

The command for runing the model on pubmed-pico dataset `allennlp train config/biobert_biaffine_pubmed-pico.jsonnet -s saved_models/pubmed-pico/ --include-package codes --force` .

After training, the results can be found in folder 'saved_models/pubmed-rct/' and 'saved_models/pubmed-pico/' respectivly.
