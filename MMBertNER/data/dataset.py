import torch
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os


class MyDataPoint:
    def __init__(self):
        self.feat: Optional[torch.Tensor] = None #Tensor expected to hold features for a data point
        self.label: Optional[int] = None #An integer representing the label


class MyToken(MyDataPoint):
    def __init__(self, text, label):
        super().__init__()
        self.text: str = text #Token's text
        self.label = label #Token's label


class MySentence(MyDataPoint):
    def __init__(self, tokens: List[MyToken] = None, text: str = None):
        super().__init__()
        self.tokens: List[MyToken] = tokens #Tokens in a sentence
        self.text = text #Complete text of sentence

    def __len__(self):
        return len(self.tokens) #Number of tokens in the sentence

    def __getitem__(self, index: int): #retrieving a specific token by index
        return self.tokens[index]

    def __iter__(self):
        return iter(self.tokens) #Iteration over the tokens in a sentence

    def __str__(self):
        return self.text if self.text else ' '.join([token.text for token in self.tokens]) #Returns the text of the sentence, or concatenates each token's text if self.text is none


class MyImage(MyDataPoint):
    def __init__(self, file_name: str):
        super().__init__()
        self.file_name: str = file_name #File path of the image
        self.data: Image = None #Image's data


class MyPair(MyDataPoint):
    def __init__(self, sentence, image, label=-1):
        super().__init__()
        self.sentence: MySentence = sentence #Sentence from text-image pair
        self.image: MyImage = image #Image from text-image pair
        self.label = label #Label for the pair


class MyDataset(Dataset):
    def __init__(self, pairs: List[MyPair], path_to_images: Path, load_image: bool = True):
        self.pairs: List[MyPair] = pairs
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int):
        pair = self.pairs[index]

        if self.load_image:
            image = pair.image

            if image.data is not None or image.feat is not None:
                return pair

            path_to_image = self.path_to_images / image.file_name
            if not os.path.exists(path_to_image):
                print(f"Warning: Image file not found: {path_to_image}")
                image.data = Image.new('RGB', (224, 224), color='black')
            else:
                image.data = Image.open(path_to_image).convert('RGB')
            image.data = self.transform(image.data)

        return pair

class MyCorpus:
    def __init__(self, train=None, dev=None, test=None):
        self.train: MyDataset = train
        self.dev: MyDataset = dev
        self.test: MyDataset = test