MMBERT-NER:
DATASETS: 
* Download Twitter-2015 dataset from https://github.com/jefferyYu/UMT?tab=readme-ov-file to resources/datasets/twitter2015

* Download text-image relationship dataset from https://github.com/thecharm/MNRE to resources/datasets/relationship

MODELS:
* Download ResNet-101 weights from https://download.pytorch.org/models/resnet101-63fe2227.pth to resources/models/cnn/resnet101.pth

* Download BERT-Base weights from https://huggingface.co/bert-base-uncased/tree/main to resources/models/transformers/bert-base-uncased

* Download word embeddings from https://flair.informatik.hu-berlin.de/resources/embeddings/token/ to resources/models/embeddings

REQUIREMENTS:
* flair
* numpy
* Pillow
* pytorch-crf
* torch
* torchvision
* tqdm
* transformers

INSTRUCTIONS:
* Run loader.py and check the statistics. 

* Run main.py(python main.py --stacked --rnn --crf --encoder_v resnet101 --aux --gate --dataset twitter2015)

********************************************************************************************************************************

MMBERT-RE:
DATASET: 
* Download dataset from https://drive.google.com/file/d/1q5_5vnHJ8Hik1iLA9f5-6nstcvvntLrS/view?usp=sharing to data/RE_data

REQUIREMENTS:
* pytorch
* scikit-learn
* seqeval
* tensorboardX
* torch
* torchvision
* transformers
* wandb

INSTRUCTIONS:
* Run the command bash run_re_task.sh

********************************************************************************************************************************

NER2RE DATASET CREATION AND VALIDATION:
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


********************************************************************************************************************************

ROLE OF IMAGES IN PERFORMANCE ANALYSIS(Distilbert):
STEPS TO RUN ner_tagging.py:
* Run mnre_sentences.py to extract the sentences from the NER2RE dataset
* Give output text file of mnre_sentences.py as input to ner_tagging.py

