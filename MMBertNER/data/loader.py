import re #Regular expression
import csv #For reading CSV file (text-image relation)
import sys #System specific parameters and functions

from pathlib import Path #Working with file paths
from collections import Counter #To count occurences of specific items
from dataset import MyToken, MySentence, MyImage, MyPair, MyDataset, MyCorpus
import constants


# constants for preprocessing
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92'] #Unicode characters
IMGID_PREFIX = 'IMGID:'
URL_PREFIX = 'http://t.co/'
UNKNOWN_TOKEN = '[UNK]'


def normalize_text(text: str):
    # removing the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$' #RE Pattern for URL (Only URL's at end of text are removed)
    text = re.sub(url_re, '', text)
    return text #Text without URL


def load_itr_corpus(path: str, split: int = 3576, normalize: bool = True):
    path = Path(path) #Converting to Path Object
    path_to_images = path / 'ner_img' #Subdirectory containing images
    assert path.exists()
    assert path_to_images.exists()
    #Checking if path and the path to images actually exists (file location/existence error)

    with open(path/'textimage.csv', encoding='utf-8') as csv_file:
        #Reading as a dictionary of rows (csv.DictReader) each row as an ordered dictionary where column headers become the dictionary keys.
        #Ignore the double quotes
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        pairs = [MyPair(
            sentence=MySentence(text=normalize_text(row['tweet']) if normalize else row['tweet']),
            image=MyImage(f"T{row['tweet_id']}.jpg"),
            label=int(row['image_adds']) #Whether the image addds to the tweet or not
        ) for row in csv_reader]

    train = MyDataset(pairs[:split], path_to_images)
    test = MyDataset(pairs[split:], path_to_images)
    return MyCorpus(train=train, test=test)
    #Splitting into training and testing


def load_ner_dataset(path_to_txt: Path, path_to_images: Path, load_image: bool = True) -> MyDataset:
    tokens = []
    image_id = None
    pairs = []
    line_count = 0
    image_count = 0

    with open(str(path_to_txt), encoding='utf-8') as txt_file:
        for line in txt_file:
            line_count += 1
            line = line.strip() #Stripping whitespace
            if line.startswith('###'): # Checking if it reads the file properly and fully
                break
            if line.startswith(IMGID_PREFIX): #If line starts with IMGID:
                if tokens:
                    sentence = MySentence(tokens)
                    image = MyImage(f'{image_id}.jpg')
                    pairs.append(MyPair(sentence=sentence, image=image, label=None))
                    tokens = []
                image_id = line[len(IMGID_PREFIX):] #Taking the image ID like 12345 stripping IMGID:
                image_count += 1
            elif line:
                try:
                    text, label = line.split('\t') #Other lines 
                    if text == '' or text.isspace() or text in SPECIAL_TOKENS or text.startswith(URL_PREFIX):
                        text = UNKNOWN_TOKEN
                    tokens.append(MyToken(text, constants.LABEL_TO_ID[label])) #Get the labels 
                except ValueError:
                    print(f"Warning: Skipping malformed line: {line}")
    #If any tokens are left
    if tokens:
        sentence = MySentence(tokens)
        image = MyImage(f'{image_id}.jpg')
        pairs.append(MyPair(sentence=sentence, image=image, label=None))

    print(f"Debug: Processed {line_count} lines")
    print(f"Debug: Found {image_count} images")
    print(f"Debug: Created {len(pairs)} pairs")

    return MyDataset(pairs, path_to_images, load_image)

#load_image: If true will consider image data. If false, will not consider image data
def load_ner_corpus(path: str, load_image: bool = True) -> MyCorpus:
    path = Path(path)
    path_to_train_file = path / 'train.txt'
    path_to_dev_file = path / 'dev.txt'
    path_to_test_file = path / 'test.txt'
    path_to_images = path / 'ner_img'

    assert path_to_train_file.exists()
    assert path_to_dev_file.exists()
    assert path_to_test_file.exists()
    assert path_to_images.exists()

    train = load_ner_dataset(path_to_train_file, path_to_images, load_image)
    dev = load_ner_dataset(path_to_dev_file, path_to_images, load_image)
    test = load_ner_dataset(path_to_test_file, path_to_images, load_image)

    #return MyCorpus(train, dev, test)
    return MyCorpus(train,dev,test)


def type_count(dataset: MyDataset) -> str:
    #Counts each tag in the dataset number of PER number of LOC
    tag_counter = Counter()
    for pair in dataset.pairs:
        for token in pair.sentence:
            tag_counter[token.label] += 1

    num_total = len(dataset) #Number of pairs in the dataset
    num_per = tag_counter[constants.LABEL_TO_ID['B-PER']]
    num_loc = tag_counter[constants.LABEL_TO_ID['B-LOC']]
    num_org = tag_counter[constants.LABEL_TO_ID['B-ORG']]
    num_misc = tag_counter[constants.LABEL_TO_ID['B-MISC']]
    num_other = tag_counter[constants.LABEL_TO_ID['B-OTHER']]

    print("Debug: Full tag counter:")
    for tag, count in tag_counter.items():
        print(f"{constants.ID_TO_LABEL[tag]}: {count}")

    print(f"Debug: Total pairs in dataset: {num_total}")
    print(f"Debug: Total tokens: {sum(tag_counter.values())}")

    return f'{num_total}\t{num_per}\t{num_loc}\t{num_org}\t{num_misc}\t{num_other}'


def token_count(dataset: MyDataset) -> str:
    lengths = [len(pair.sentence) for pair in dataset] #list where each item is of Number of tokens in a sentence in a pair

    num_sentences = len(lengths) #Total number of sentences in the dataset
    num_tokens = sum(lengths) #Total number of tokens in the dataset

    return f'{num_sentences}\t{num_tokens}'


if __name__ == "__main__":
    twitter2015 = load_ner_corpus("enter path")
    twitter2015_train_statistic = type_count(twitter2015.train)
    twitter2015_dev_statistic = type_count(twitter2015.dev)
    twitter2015_test_statistic = type_count(twitter2015.test)
    assert twitter2015_train_statistic == '4000\t2217\t2091\t928\t0\t940' #Checking if data was loaded correctly
    print('-----------------------------------------------')
    print('2015\tNUM\tPER\tLOC\tORG\tMISC\tOTHER')
    print('-----------------------------------------------')
    print('TRAIN\t' + twitter2015_train_statistic)
    print('DEV\t' + twitter2015_dev_statistic)
    print('TEST\t' + twitter2015_test_statistic)
    print('-----------------------------------------------')

    print("\nDebug: First few pairs in the dataset:")
    for i, pair in enumerate(twitter2015.train.pairs[:5]):  # Printing first 5 pairs
        print(f"Pair {i}:")
        print(f"  Image: {pair.image.file_name}")
        for token in pair.sentence:
            print(f"  {token.text}: {constants.ID_TO_LABEL[token.label]}")
        print()
