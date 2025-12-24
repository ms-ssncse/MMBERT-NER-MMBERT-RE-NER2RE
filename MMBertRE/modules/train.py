import torch #For tensors
from torch import optim #Optimizer : Adam
from tqdm import tqdm #Progress bars
from sklearn.metrics import classification_report as sk_classification_report
#from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup #To fine-tune the transformer models

from .metrics import eval_result

#Base class for defining structure of trainer
class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
#All the abstract methods that must be implemented
class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.re_dict = processor.get_relation_dict() #The dictionary mapping the relations to ID's
        self.logger = logger #To log training info
        self.writer = writer #For TensorBoard Logging
        self.refresh_step = 2 #Steps after which progress is logged in training
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None # Storing the epochs corresponding to the best metrics
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs #Total training steps (num of batches x num of epochs)
        self.step = 0 #Current training step
        self.args = args
        #if self.args.use_prompt: #checks if there is pre-trained resnet or bert
        self.before_multimodal_train()
        # else:
        #     self.before_train()

    def train(self):
        self.step = 0
        self.model.train() #Setting the model to training mode
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        #Initializing progress bar - Setting total num of training steps to display - Adding a placeholder for loss in progress bar - Clearing progress bar after training is complete
        #Adjusting bar to fit terminal - To make sure progress bar resumes from current training step
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0 #Average loss initialization
            for epoch in range(1, self.args.num_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs)) #Showing current epoch and total epochs on bar
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch) #Moving tensors to GPU. Dictionary or others will not move.
                    (loss, logits), labels = self._step(batch, mode="train")
                    #Performing a forward pass on the model that returns loss and logits(predictions) labels are the true labels
                    avg_loss += loss.detach().cpu().item()
                    #move loss to CPU and add to avg_loss   

                    loss.backward() #Computing gradients for all model parameters based on loss
                    self.optimizer.step() #Updating model parameters using the computed gradients
                    self.scheduler.step() #Adjusting learning rate
                    self.optimizer.zero_grad() #Clearing accumulated gradients

                    if self.step % self.refresh_step == 0: #Updating progress bar and logging average loss for every refresh step
                        avg_loss = float(avg_loss) / self.refresh_step #Computing the average loss
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output) #Updating the postfix with average loss
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx - logging the average loss on tensor board
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # Evaluating model after certain epoch
            
            pbar.close() #Closing and resetting the progress bar
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad(): #Disabling gradient computation to reduce memoryusage and speed up evaluation
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch) 
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1) #Predicted labels
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                true_relations = []
                pred_relations = []

                
                value_to_key = {v: k for k, v in self.re_dict.items()}

                for true_label, pred_label in zip(true_labels, pred_labels):
                    
                    true_relation = value_to_key.get(true_label, "Unknown")  
                    pred_relation = value_to_key.get(pred_label, "Unknown")  
                    
                   
                    true_relations.append(true_relation)
                    pred_relations.append(pred_relation)


                file_path = 'mention file path'

               
                with open(file_path, 'w', encoding='utf-8') as f:
                  
                    for true_relation, pred_relation in zip(true_relations, pred_relations):
                        f.write(f"True Relation: {true_relation}\nPredicted Relation: {pred_relation}\n\n")
                f.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4) #Rounding off accuracy and micro F1 score
                if self.writer:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)    #Name of the metric, its value and current epoch
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)    
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1 # updating best f1 score
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
               

        self.model.train()

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        if self.args.load_path is not None:  # loading model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)  
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)   
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))   
                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))
                    
        self.model.train()
        
    def _step(self, batch, mode="train"):
        if mode != "predict":
            if self.args.use_prompt:
                input_ids, token_type_ids, attention_mask, labels, images, focused_imgs = batch
            else:
                images, focused_imgs = None, None
                input_ids, token_type_ids, attention_mask, labels= batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, focused_imgs=focused_imgs)
            return outputs, labels


    
    def before_multimodal_train(self):
        optimizer_grouped_parameters = [] #Parameters for optimization
        params = {'lr':self.args.lr, 'weight_decay':1e-2} #Initializing a dictionary to group parameters for optimization.
        params['params'] = []
        #Storing BERT parameters for optimisation(weights,bias) 
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)
        #Storing ResNeT parameters for optimisation(weights,bias)
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        # freezing resnet to prevent overfitting (if features are already good)
        for name, param in self.model.named_parameters():
            if 'image_model' in name: #Checking if the parameter belongs to the image processing component of the model
                param.require_grad = False #freezing it
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr) #initialising the optimiser with the parameters
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, #Schedulerdynamically adjusts the learning rate during training, which helps in stabilizing and improving the training process.
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, #During the warmup phase, the learning rate is gradually increased linearly from a very small value (often close to zero) to the specified learning rate (lr).
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device) # model getting stored in device for cpu/gpu