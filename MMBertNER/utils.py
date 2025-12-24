import os
import random
import numpy as np
import torch
import constants 
from tqdm import tqdm #progress bar in terminal
from seqeval.metrics import classification_report, f1_score #sequence labelling
from seqeval.scheme import IOB2 #Used in metrics calculation, reduces score accordingly if prediction sequence is wrong (B-PER...B-PER)

#unique seed generation for multiple workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 #Numpy and random expect 32 bit initial_seed returns the initial random seed
    np.random.seed(worker_seed) #Random operations in the owrker invlolving numpy will be repeatable.
    random.seed(worker_seed) #Any operations in random worker also will be consistent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(loader, model, optimizer, task, weight=1.0):
    losses = []

    model.train()
    for batch in tqdm(loader):
        optimizer.zero_grad()# makes the gradients as zero so that gradients from previous batch don't affect the gradients from current batch
        loss, _ = getattr(model, f'{task}_forward')(batch)
        loss *= weight #loss is multiplied by weight factor to adjust the importance of the task's loss
        loss.backward()#backpropagation is performed which computes the gradients of the loss with respect to the model’s parameters.
        optimizer.step() #the computed gradients are used to adjust the model weights.
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(model, loader,fn):
    true_labels = []
    pred_labels = []
    sentences = []  # To store the sentence texts

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            _, pred = model.ner_forward(batch)
            
            # Extracting sentences and labels
            batch_sentences = [[token.text for token in pair.sentence] for pair in batch]
            true_labels += [[constants.ID_TO_LABEL[token.label] for token in pair.sentence] for pair in batch]
            pred_labels += pred
            sentences += batch_sentences
            
            # Printing the results
    print("Sentences:\n",sentences,'\n',"True Labels: \n",true_labels,'\n',"Predicted Labels:\n",pred_labels)
    if fn=='dev.txt':
        file_name=f'log/dev_labels_analysis.txt'
        #inga just put ,encoding='utf-8'
        with open (file_name,'w+',encoding='utf-8') as f:
            f.write("\n\nDEV.txt\n\n")
            c=0
            for sentence in sentences:
                f.write(' '.join(sentence)+'\n')
                f.write('True Labels:'+ ' '.join(true_labels[c])+'\n')
                f.write('Pred Labels:'+ ' '.join(pred_labels[c])+'\n\n')
                c+=1
    elif fn=='test.txt':
        file_name=f'log/test_labels_analysis.txt'
        with open (file_name,'w+',encoding='utf-8') as f:
            f.write("\n\nTEST.txt\n\n")
            c=0
            for sentence in sentences:
                f.write(' '.join(sentence)+'\n')
                f.write('True Labels:'+ ' '.join(true_labels[c])+'\n')
                f.write('Pred Labels:'+ ' '.join(pred_labels[c])+'\n\n')
                c+=1

    f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
    report = classification_report(true_labels, pred_labels, digits=4, mode='strict', scheme=IOB2)

    return f1, report