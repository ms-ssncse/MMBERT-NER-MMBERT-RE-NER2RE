STEPS TO RUN head_tail.py:
* Run mnre_reformatting.py to extract the sentences from the dataset
* Give output text file of mnre_reformatting.py as input to head_tail.py

STEPS TO RUN re_validation.py:
* head_tail outputs for both train and test files of dataset obtained by running head_tail.py
* ner_labels output for both train and test files of dataset obtained by running through MMBERT-NER model
* MNRE dataset for training with the relations.

REQUIREMENTS:
* Python
* Pandas
* Transformers
* Pillow
* Torch
