DATASETS: 
* Download Twitter-2015 dataset from https://qizhang.info/paper/data/aaai2018_multimodal_NER_data.zip to resources/datasets/twitter2015

* Download text-image relationship dataset from https://github.com/danielpreotiuc/text-image-relationship to resources/datasets/relationship

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