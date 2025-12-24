import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.mmbert_re import REModel
from processor.dataset import MMREProcessor,MMREDataset
from modules.train import RETrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'MNRE': REModel,

}

TRAINER_CLASSES = {
    'MNRE': RETrainer,

}
DATA_PROCESS = {
    'MNRE': (MMREProcessor, MMREDataset),

}
#Give appropriate paths
DATA_PATH = {
    'MNRE': {
           
            'train': '',    
            'dev': '',
            'test': '',
        
            'train_auximgs': '',
            'dev_auximgs': '',
            'test_auximgs': '',
           
            're_path': ''
            }      
}

# give image data paths
IMG_PATH = {
    'MNRE': {'train': '',
            'dev': '',
            'test': ''},

}

# give auxiliary images paths
AUX_PATH = {
    'MNRE':{
            'train': '',
            'dev': '',
            'test': ''
    }

}

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str)
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--prompt_len', default=10, type=int)
    parser.add_argument('--prompt_dim', default=800, type=int)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--write_path', default=None, type=str)
    parser.add_argument('--notes', default="", type=str)
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float)

    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed) 
    if args.save_path is not None:  
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
   
    writer=None

    if not args.use_prompt:
        img_path, aux_path = None, None
        
    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer
    model = REModel(num_labels, tokenizer, args=args)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, processor=processor, args=args, logger=logger, writer=writer)

    if args.do_train:
        trainer.train()
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        trainer.test()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()