import torch
import os
from torch import nn
import torch.nn.functional as F #For neural network connections
from .modeling_bert import BertModel #Importing BERT Model
from transformers.modeling_outputs import TokenClassifierOutput #Works on token classification used for logits,loss,..
from torchvision.models import resnet50

#Extracting visual features using ResNet
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
    
    def forward(self, x, focused_imgs=None):
        # Extracting feature prompts from the main input image
        prompt_guids = self.get_resnet_prompt(x)   #Feature maps of size [bsz, 256, 7, 7]
        
       #Processing the focused images
        if focused_imgs is not None:
            focused_prompt_guids = []   #Storing feature maps for focussed images
            focused_imgs = focused_imgs.permute([1, 0, 2, 3, 4]) #Reordering dimensions for batch processing (cause the focused images are in a group)
            for i in range(len(focused_imgs)):
                focused_prompt_guid = self.get_resnet_prompt(focused_imgs[i]) #Extracting feature prompts from focussed images
                focused_prompt_guids.append(focused_prompt_guid)   
            return prompt_guids, focused_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        
        prompt_guids = [] #Stores prompts from different ResNet layers
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue #Skipping fully connected and pooling layers
            x = layer(x)   #Applying each layer sequentially to the input image
            if 'layer' in name: #Capturing feature maps from residual block layers
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2 #Reducing feature size to 7x7 (defining kernel size for average pooling)
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    #Reducing spatial dimensions while preserving the features (compact representation)
                prompt_guids.append(prompt_kv)   #Adding compact feature map to prompts
        return prompt_guids


class REModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer)) #Resizing token embeddings to fit tokenizer
        self.args = args

        self.dropout = nn.Dropout(0.5) #Dropout layer to prevent overfitting (model shouldn't rely heavily on specific features)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels) #Classification layer
        #Multiplying the size of the hidden layers with 2 (<s> and <o> embeddings are concatenated)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>") #Token ID for head entity marker (selecting the entity embedding <s>)
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>") #Token ID for tail entity marker
        #Have to combine these 2 embeddings into a single feature vector
        self.tokenizer = tokenizer


        self.image_model = ImageModel() #ResNet Processing
        #Projecting image features to match BERT hidden dimensions
        self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),#Introducing non-linearity
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                    # 4 prompts one for each attention head group, 2 - key and value for attention, 768 - matches hidden size of BERT
                                )
        #Gating for integrating image prompts into BERT layers
        #4 prompts are taken from the image for each attention head of BERT, each prompt split into key and value (2) used for attention in BERT
        self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
        #The weights are adjusted using linear transformation and softmax activation. The softmax normalized weights dynamically decide which prompts are most relevant
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        focused_imgs=None,
    ):

        bsz = input_ids.size(0)
        prompt_guids = self.get_visual_prompt(images, focused_imgs) #Key-value pairs obtained from this function
        prompt_guids_length = prompt_guids[0][0].shape[2] # Length of the visual prompts
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)#Creating a binary mask for visual prompts
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1) #Combining the visual mask with input text attention mask

        #Forward pass through BERT with visual prompts as past_key_values
        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True, #Output of attention scores
                    return_dict=True
        )

        last_hidden_state = output.last_hidden_state #Contextualized embeddings for each token
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) #Placeholder to store concatenated embeddings for the head and tail entities
        for i in range(bsz): #Extracting hidden states for entity markers
            head_idx = input_ids[i].eq(self.head_start).nonzero().item() #Head entity index
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item() #Tail entity index
            head_hidden = last_hidden_state[i, head_idx, :].squeeze() #Hidden state for head entity
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze() #Hidden state for tail entity
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1) #Concatenating both entities into a single vector
        entity_hidden_state = entity_hidden_state.to(self.args.device) #moving to device
        logits = self.classifier(entity_hidden_state) #Computing logits for classification
        if labels is not None: #Computing loss during training 
            loss_fn = nn.CrossEntropyLoss() #Cross entropy loss between predicted logits and the true labels
            return loss_fn(logits, labels.view(-1)), logits #Returning logits and loss
        return logits
    #Preparing the images as Key-Value pairs for attention for BERT's 12 layers
    def get_visual_prompt(self, images, focused_imgs):
        bsz = images.size(0)
       
        prompt_guids, focused_prompt_guids = self.image_model(images, focused_imgs)  #Extracting feature maps from main and featured images
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   #Reshaping main image's prompts (Combining it all into 1 rich tensor)
        
       
        focused_prompt_guids = [torch.cat(focused_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for focused_prompt_guid in focused_prompt_guids]  #Reshaping focused image prompts
        prompt_guids = self.encoder_conv(prompt_guids)  #Ensuring main image match BERT's attention (encoding)
        focused_prompt_guids = [self.encoder_conv(focused_prompt_guid) for focused_prompt_guid in focused_prompt_guids] #same for focused
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)  #Splitting into Key-Value pairs 
        split_focused_prompt_guids = [focused_prompt_guid.split(768*2, dim=-1) for focused_prompt_guid in focused_prompt_guids]  
        
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4    #Stacking the split main prompts and averaging this will produce a summarized prompt representation 
        
        result = [] #To store Key-Value pairs for all 12 layers
        for idx in range(12):  # Iterating over BERT's layers
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1) #Computing gate weights
            #Passing each summarized main prompt through the gate for the current layer of BERT
            #(Producing raw scores for each prompt, applying non-linear activation (Relu) to each score and then using softmax converting it into normalized weights (probability distribution))

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  
            for i in range(4):
                key_val = key_val + prompt_gate[:, i].unsqueeze(-1) * split_prompt_guids[i]


            
            focused_key_vals = []  #Combining focused prompts
            for split_focused_prompt_guid in split_focused_prompt_guids:
                sum_focused_prompt_guids = torch.stack(split_focused_prompt_guid).sum(0).view(bsz, -1) / 4     
                focused_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_focused_prompt_guids)), dim=-1)
                focused_key_val = torch.zeros_like(split_focused_prompt_guid[0]).to(self.args.device)  
                for i in range(4):
                    focused_key_val = focused_key_val + focused_prompt_gate[:, i].unsqueeze(-1) * split_focused_prompt_guid[i]

                focused_key_vals.append(focused_key_val)
            key_val = [key_val] + focused_key_vals #Concatenating main feature maps with focused maps
            key_val = torch.cat(key_val, dim=1) #changing to 1 dimensional vector
            key_val = key_val.split(768, dim=-1) #Splitting the concatenated tensor into key-value pairs
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  #0 and 1 key and value of components of image prompts 
            temp_dict = (key, value) #passing them as pair
            result.append(temp_dict)
        return result
