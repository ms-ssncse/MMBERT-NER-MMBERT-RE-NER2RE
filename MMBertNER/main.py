import os
import argparse #parsing the arguments
import json #results file
import numpy as np
import torch #to handle tensors
from torch.utils.data import DataLoader #to handle batches of data
from data import loader #import loader.py
from model import MyModel #import model.py
from utils import seed_worker, seed_everything, train, evaluate #import utils.py
import multiprocessing #parallel execution

def main():
    parser = argparse.ArgumentParser() #handling command line arguments
    parser.add_argument('--seed', type=int, default=0) #random number generated for shuffling the data
    parser.add_argument('--cuda', type=int, default=0) #CPU vs GPU
    parser.add_argument('--num_workers', type=int, default=2) #each worker simultaneously loads a different batch of data for processing
    parser.add_argument('--dataset', type=str, default='twitter2015')
    parser.add_argument('--encoder_t', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-large-uncased']) #loading bert-based-uncased encoder to process text input
    parser.add_argument('--encoder_v', type=str, default='resnet101') #loading resnet101 encoder to process visual input
    parser.add_argument('--stacked', action='store_true', default=True) #loading stacked encoders in model.py
    parser.add_argument('--rnn',   action='store_true',  default=True)
    parser.add_argument('--crf',   action='store_true',  default=True)
    parser.add_argument('--aux',   action='store_true',  default=True)
    parser.add_argument('--gate',   action='store_true',  default=True)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--optim', type=str, default='Adam')
    args = parser.parse_args()

    if (args.aux or args.gate) and args.encoder_v == '':
        raise ValueError('Invalid setting: auxiliary task or gate module must be used with visual encoder (i.e. ResNet)')

    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)# A seed is initialised for reproducibility and sets the processing device to CPU by default
    device = torch.device("cpu")
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true' #Setting multiprocessing options and environment variables for tokenizer parallelism.
    ner_corpus = loader.load_ner_corpus(f'resources/datasets/{args.dataset}', load_image=(args.encoder_v != ''))
    ner_train_loader = DataLoader(ner_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                                  shuffle=True, worker_init_fn=seed_worker, generator=generator)
    ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
    ner_test_loader = DataLoader(ner_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
    #loads NER data using DataLoader for efficient batch handling
    if args.aux: # checking if auxiliary task is enabled to load additional data for an image-text relationship classification task.
        itr_corpus = loader.load_itr_corpus('resources/datasets/relationship')
        itr_train_loader = DataLoader(itr_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                                      shuffle=True, worker_init_fn=seed_worker, generator=generator)
        itr_test_loader = DataLoader(itr_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)

    model = MyModel.from_pretrained(args) #The model is instantiated using the chosen configuration specified by args.
    #Learning rates for each module are defined based on args and the optimizer is initialized.
    params = [
        {'params': model.encoder_t.parameters(), 'lr': args.lr},
        {'params': model.head.parameters(), 'lr': args.lr * 100},
    ]
    if args.encoder_v:
        params.append({'params': model.encoder_v.parameters(), 'lr': args.lr}) #visual encoder layer
        params.append({'params': model.proj.parameters(), 'lr': args.lr * 100}) # projection layer
    if args.rnn:
        params.append({'params': model.rnn.parameters(), 'lr': args.lr * 100}) #Recurrent Neural network layer
    if args.crf:
        params.append({'params': model.crf.parameters(), 'lr': args.lr * 100}) #Conditional Random fields
    if args.gate:
        params.append({'params': model.aux_head.parameters(), 'lr': args.lr * 100}) #auxilary network that produces gating signals/weight that indicate how much of each modality should be used
    optimizer = getattr(torch.optim, args.optim)(params) #optimizer is initialized to determine how the weight are updated to reduce training loss

    print(args)
    dev_f1s, test_f1s = [], []
    ner_losses, itr_losses = [], []
    best_dev_f1, best_test_report = 0, None
    for epoch in range(1, args.num_epochs + 1):
        if args.aux:
            itr_loss = train(itr_train_loader, model, optimizer, task='itr', weight=0.05) #calling train function in utils.py
            itr_losses.append(itr_loss) # append the loss from the epoch
            print(f'loss of image-text relation classification at epoch#{epoch}: {itr_loss:.2f}')

        ner_loss = train(ner_train_loader, model, optimizer, task='ner') # calling train function in utils.py task is ner
        ner_losses.append(ner_loss) #appending loss
        print(f'loss of multimodal named entity recognition at epoch#{epoch}: {ner_loss:.2f}')

        dev_f1, dev_report = evaluate(model, ner_dev_loader,fn='dev.txt')
        dev_f1s.append(dev_f1)
        test_f1, test_report = evaluate(model, ner_test_loader,fn='test.txt')
        test_f1s.append(test_f1)
        print(f'f1 score on dev set: {dev_f1:.4f}, f1 score on test set: {test_f1:.4f}')
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_report = test_report

    print()
    print(best_test_report)

    #appending losses to a json file
    results = {
        'config': vars(args),
        'dev_f1s': dev_f1s,
        'test_f1s': test_f1s,
        'ner_losses': ner_losses,
        'itr_losses': itr_losses,
    }
    file_name = f'log/{args.dataset}/bs{args.bs}_lr{args.lr}_seed{args.seed}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
