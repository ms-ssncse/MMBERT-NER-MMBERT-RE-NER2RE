from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math as CRF 
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.data import Token as FlairToken
from flair.data import Sentence as FlairSentence
from data.dataset import MyDataPoint, MyPair
import constants


# constants for model
CLS_POS = 0 #Beginning of the sentence
SUBTOKEN_PREFIX = '##' #Prefix indicating subword tokens in BERT tokenization
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2 #resizing input images to encode using ResNet


def use_cache(module: nn.Module, data_points: List[MyDataPoint]): # Checks if the model can use cached visual features instead of recalculating them
    for parameter in module.parameters():
        if parameter.requires_grad: #checks if parameter requires gradients, thereby optimising computational power
            return False
    for data_point in data_points: #this refers to the datapoint in dataset.py
        if data_point.feat is None: #if datapoint.feat==None indicates the that the feature hasn't been cached
            return False 
    return True

#torchvision.models.resnet.ResNet defines the inbuilt functions called below
def resnet_encode(model, x): #x is a 4D image tensor of shape (batch_size, 3, 224, 224) for images resized to 224x224 with 3 color channels (RGB).
    x = model.conv1(x) #first convolutional layer to extract low-level features like edges and textures
    x = model.bn1(x) #batch normalisation stabilizes and accelerates training using the learned mean and variance of each feature map from conv1
    x = model.relu(x) # ReLU activation function introduces non-linearity by zeroing out negative values and enabling the model to learn complex patterns.
    x = model.maxpool(x) #maxpool layer reduces the spatial resolution, retaining only the most prominent features in each local region.

    x = model.layer1(x) # conv layers with skip connections to retain gradients -- refines feature extraction preserving spatial dimensions -- batch size,256,56,56
    x = model.layer2(x) # conv layers with residual connections used to downsample spatial dimensions - batch size,512,28,28
    x = model.layer3(x) # feature extraction with more complex filters batch size,1024,14,14
    x = model.layer4(x) #This block extracts the most abstract and complex features in the ResNet, representing high-level attributes in the image. batch_size, 2048, 7, 7


    x = x.view(x.size()[0], x.size()[1], -1) #Reshapes x to merge the spatial dimensions into a single dimension. batch_size, 2048, 49 in order simplify the representation for use in fully connected layers FC
    x = x.transpose(1, 2) #Transposes x to swap the channel dimension (2048) with the flattened spatial dimension (49). batch_size, 49, 2048 to make it compatible with models


    return x #vector (batch_size, 49, 2048) spatial dimensions=49 feature channels=2048.


class MyModel(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            encoder_t: PreTrainedModel,
            hid_dim_t: int,
            encoder_v: nn.Module = None,
            hid_dim_v: int = None,
            token_embedding: TokenEmbeddings = None,
            rnn: bool = None,
            crf: bool = None,
            gate: bool = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = token_embedding
        self.proj = nn.Linear(hid_dim_v, hid_dim_t) if encoder_v else None
        self.aux_head = nn.Linear(hid_dim_t, 2)
        if self.token_embedding:
            self.hid_dim_t += self.token_embedding.embedding_length
        if rnn:
            hid_dim_rnn = 256
            num_layers = 2
            num_directions = 2
            self.rnn = nn.LSTM(self.hid_dim_t, hid_dim_rnn, num_layers, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hid_dim_rnn * num_directions, constants.LABEL_SET_SIZE)
        else:
            self.rnn = None
            self.head = nn.Linear(self.hid_dim_t, constants.LABEL_SET_SIZE)
        self.crf = CRF(constants.LABEL_SET_SIZE, batch_first=True) if crf else None
        self.gate = gate
        #self.to(device)

    @classmethod #allows this method to be called on the class itself rather than on an instance of the class, useful for initializing objects with pre-trained models.
    def from_pretrained(cls, args):
        # device = torch.device(f'cuda:{args.cuda}')
        device = torch.device('cpu') #sets the device to cpu
        models_path = 'resources/models' #sets a base path where all models are stored for easy access

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}' #path to bert-based-uncased model
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)  #loads the bert tokenizer which tokenizes and prepares the text input
        encoder_t = AutoModel.from_pretrained(encoder_t_path) #loads the transformer model to generate embeddings based on language patterns
        config = AutoConfig.from_pretrained(encoder_t_path)  #retrieves the configuration of the transformer model to access its attributes
        hid_dim_t = config.hidden_size #retrieves the hidden size (dimension) of the transformer model, necessary for layer compatibility when processing output embeddings.

        if args.encoder_v: # If a visual encoder is specified, it loads a CNN model from torchvision
            encoder_v = getattr(torchvision.models, args.encoder_v)() #get the model specified(ResNet101)
            # encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth'))
            encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth', map_location=torch.device('cpu'),weights_only=True)) #pre-trained weights are loaded for the visual model from the specified path, enabling it to make accurate visual representations.
            hid_dim_v = encoder_v.fc.in_features #retrieves the input feature size of the fully connected layer representing the dimensionality of the CNN's output vector.
        else:
            encoder_v = None #optional if visual encoder is not used
            hid_dim_v = None

        if args.stacked:
            flair.cache_root = 'resources/models' #checks the path if files are there, otherwise downloads them
            flair.device = device
            token_embedding = StackedEmbeddings([ #loads multiple embeddings for each token
                WordEmbeddings('crawl'), #web data - wide range of language styles,topics, genres
                WordEmbeddings('twitter'), # twitter data - rich in informational language, abbreviations, hashtags and emojis
                FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward') #contextual embeddings from news data - formal,structured for understanding context events and topics
            ])
            # the model learns to recognize and adapt to various linguistic contexts.
            #Stacked embeddings allow the model to generalize better across different types of text.
            #  if the model encounters a phrase common in both social media and formal news articles, 
            # it can leverage the combined knowledge from both types of embeddings to better understand the phrase's meaning in context.
        else:
            token_embedding = None # relies only on BERT embeddings

        return cls( # returns an instance of the class initialized, creating a model with both text and optional visual encoding capabilities.
            device=device, #device: sets the computation device.
            tokenizer=tokenizer, #tokenizer, encoder_t, hid_dim_t: text encoding components.
            encoder_t=encoder_t,
            hid_dim_t=hid_dim_t,
            encoder_v=encoder_v, #encoder_v, hid_dim_v: visual encoding components
            hid_dim_v=hid_dim_v,
            token_embedding=token_embedding, #token_embedding: stacked token embeddings for richer token representations.
            rnn=args.rnn, #Additional flags (rnn, crf, gate) are set based on args, allowing for flexibility in model architecture choices.
            crf=args.crf, # CRF layer which considers label dependencies, helping the model make predictions that are more contextually and structurally consistent
            gate=args.gate, # gating mechanism allows the model to selectively pass or suppress information helping the model focus on the most relevant parts.
        )

 #Forward propagation with both text and image
    def _bert_forward_with_image(self, inputs, pairs, gate_signal=None):
        images = [pair.image for pair in pairs] #Taking the image from the pair of images and text
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)#inputs.input_ids contain token IDs (These tokens are converted into embeddings)
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)#Converting an image's data to a single stacked tensor and store in CPU
        #Checking if image already processed
        if not use_cache(self.encoder_v, images):
            visual_embeds = resnet_encode(self.encoder_v, visual_embeds)#Extracting visual features using ResNet
        visual_embeds = self.proj(visual_embeds)#projecting the visual embeddings to match the hidden dimension size expected by the text embeddings using a linear layer
        if gate_signal is not None:
            visual_embeds *= gate_signal #Considering images or not
        inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)#Concatenating both text and image Concatenating along dim=1 is like placing text and visual elements side by side in the same sequence, allowing the model to handle both seamlessly
        #visual_embeds will be like (batch_size, seq_length, embedding_dim)
        batch_size = visual_embeds.size()[0] #batch size -how many samples
        visual_length = visual_embeds.size()[1]#Sequence length of the visual features (For visual attention mask)

        attention_mask = inputs.attention_mask #Attention mask (which token should be considered(1) and which ignored(padding-0))
        #dtype(same size of text) here, all visual vecs taken as 1 as all of them are to be considered
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1) #attention mask for both text and image

        token_type_ids = inputs.token_type_ids #Text tokens will be 0
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)#A tensor of ones for the visual type IDs, indicating that all visual tokens belong to a single type
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds, #The combined text and visual embeddings.
            attention_mask=attention_mask, #The combined attention mask, marking both text and visual tokens.
            token_type_ids=token_type_ids, #The combined type IDs to distinguish text from visual tokens.
            return_dict=True
        )

    def ner_encode(self, pairs: List[MyPair], gate_signal=None): #generate embeddings for tokens in a batch of sentences MyPair contains text and corresponding image associated with it
        # used to control the extent of influence or integration of image features with text features, by modulating the image contribution.
        sentence_batch = [pair.sentence for pair in pairs] #Raw sentences for each pair.
        tokens_batch = [[token.text for token in sentence] for sentence in sentence_batch] #lists of token texts for each sentence, ready for tokenization.
        inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt',
                                return_special_tokens_mask=True, return_offsets_mapping=True).to(self.device)
        #Uses self.tokenizer (likely a BERT-based tokenizer) to tokenize tokens_batch.
        # is_split_into_words=True: Indicates that tokens are provided as pre-split words.
        # padding=True: Pads sentences to the same length.
        # return_tensors='pt': Returns tensors in PyTorch format.
        # return_special_tokens_mask=True: Generates a mask for special tokens as they don't carry any semantic content and should not influence model's interpretation
        # return_offsets_mapping=True: Adds offsets for each token, mapping tokens to positions in the original text.
        # .to(self.device): Moves tensors to the specified device (CPU or GPU).

        if self.encoder_v: # checks if images information is there
            outputs = self._bert_forward_with_image(inputs, pairs, gate_signal) # this function to integrate both text and image features
            #gate signal determines the strength/amount of image information to be included in the multimodal encoding process
            feat_batch = outputs.last_hidden_state[:, :-VISUAL_LENGTH] # the visual features are now excluded as the text features are already modulated and influenced by bert_forward_with_image
        else:
            outputs = self.encoder_t( # text encoder bert-based-uncased
                input_ids=inputs.input_ids, # provides input tokenm ids for the sentences Eg Hello world = [101, 7592, 2088, 102] CLS=101, SEP=102
                attention_mask=inputs.attention_mask,# indicates which tokens are to be considered(1) and which are padding tokens(0). In this way the model focuses only on the actual words.
                token_type_ids=inputs.token_type_ids, #specifies the type of each token which differentiates segments of input. In a case where two sentences are processed together, the first might be labeled with 0s and the second with 1s.
                return_dict=True #output returned as dictionary for easy access
            )
            feat_batch = outputs.last_hidden_state #last hidden states from the encoder's output containing the embeddings for each token in the input sequences.

        ids_batch = inputs.input_ids #A tensor of integers that represent the tokenized words in the input sentences, where each integer corresponds to a specific token in the vocabulary of the tokenizer
        offset_batch = inputs.offset_mapping #offset mappings for each token in the input sentences which is the start and end character positions for each token in the original text used for aligning tokenized input back to the original sentences.
        mask_batch = inputs.special_tokens_mask.bool().bitwise_not() 
        #The special_tokens_mask is a tensor where special tokens are marked as 1 and non-special tokens are marked as 0. 
        # This is converted into a boolean tensor and bitwise NOT is applied to invert the mask to have non-special tokens marked as True and special tokens marked as False.
        #special tokens don't carry semantic meaning and should not influence the model.
        for sentence, ids, offset, mask, feat in zip(sentence_batch, ids_batch, offset_batch, mask_batch, feat_batch):
            ids = ids[mask] #filters the ids tensor using the mask to retain only the IDs of the non-special tokens
            offset = offset[mask] #filtration for offset tensor
            feat = feat[mask] # filtration for feat tensor
            subtokens = self.tokenizer.convert_ids_to_tokens(ids) #filtered ids are converted back into their corresponding token strings using the tokenizer. The result is a list of subtokens created during tokenization.
            length = len(subtokens)

            token_list = []
            feat_list = []
            i = 0
            while i < length:
                j = i + 1 #used to find the end of a contiguous sequence of subtokens that belong to the same original word.
                while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)): # This inner loop continues as long as j is within bounds and either the current token is a part of a multi-token word or it starts with a specific prefix indicating that it’s a subtoken
                    j += 1
                token_list.append(''.join(subtokens[i:j])) #Grouping all the relevant subtokens into a single token and appending to the token_list.
                feat_list.append(torch.mean(feat[i:j], dim=0)) #mean of the feature vectors for the subtokens is computed aggregating features that correspond to same original word
                i = j
            assert len(sentence) == len(token_list) # Checking if number of tokens processed matches the number of tokens in the original sentence.

            for token, token_feat in zip(sentence, feat_list):
                token.feat = token_feat # feat attribute is assigned the averaged feature vector associating the features with the respective tokens.

            if self.token_embedding is not None: #Checking if there is a token embedding layer available for the model.
                flair_sentence = FlairSentence(str(sentence)) #Flair instance used to represent the sentence in NLP
                flair_sentence.tokens = [FlairToken(token.text) for token in sentence] #tokenising using Flair
                self.token_embedding.embed(flair_sentence) #creating embedding for each token
                for token, flair_token in zip(sentence, flair_sentence):
                    token.feat = torch.cat((token.feat, flair_token.embedding)) #storing the token and the updated features generated through flair embeddings

  #MyPair: Text and image
    def ner_forward(self, pairs: List[MyPair]):
        #Gating mechanism
        if self.gate:
            tokens_batch = [[token.text for token in pair.sentence] for pair in pairs] #The tokens from each sentence in a pair
            #Creating pytorch tensors
            inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt')
            #Move it to cpu for processing
            inputs = inputs.to(self.device)
            outputs = self._bert_forward_with_image(inputs, pairs)
            #Gets features of CLS token from last hidden state of all hidden states (Summarization of input)
            feats = outputs.last_hidden_state[:, CLS_POS]
            #Score (ignore visual features, use visual features)[0.8,2.5]
            logits = self.aux_head(feats)
            #convert to probabilities [0.27, 0.73] [:,1]-consider 2nd value(use visual) view(...)-reshapes according to visual embeddings cause you multiply
            gate_signal = F.softmax(logits, dim=1)[:, 1].view(len(pairs), 1, 1)
            gate_signal_list = gate_signal.view(-1).tolist()  # Flatten and convert to list
            c=0
            for pair in pairs:
                print("IMGID gate signal =",pair.image.file_name,gate_signal_list[c])
                c+=1
        else:
            gate_signal = None

        self.ner_encode(pairs, gate_signal)

        #Extracting sentences from pairs
        sentences = [pair.sentence for pair in pairs]
        #Size of batch: Number of sentences
        batch_size = len(sentences)
        #List of length of each sentence
        lengths = [len(sentence) for sentence in sentences]
        #Length of longest sentence so you can do padding accordingly
        max_length = max(lengths)

        #Features for all tokens in the batch
        feat_list = []
        #Tensor filled with zeroes used for padding
        zero_tensor = torch.zeros(max_length * self.hid_dim_t, device=self.device)
        for sentence in sentences:
            feat_list += [token.feat for token in sentence]
            num_padding = max_length - len(sentence)
            if num_padding > 0:
                padding = zero_tensor[:self.hid_dim_t * num_padding]
                feat_list.append(padding)
        #Concatenates all features into a single tensor and reshapes it
        feats = torch.cat(feat_list).view(batch_size, max_length, self.hid_dim_t)

        if self.rnn is not None:
            #Takes padded(feats) and unpadded(lengths) it returns packed feats without padding
            #batch-first: batch dimension is first in feats, meaning the shape is (batch_size, max_seq_len, embedding_dim)
            #enforce_sorted  to handle unsorted sequences (descending order)
            feats = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
            #Apply RNN
            feats, _ = self.rnn(feats)
            #Pad the output
            feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)
    
        logits_batch = self.head(feats) #maps the RNN’s output features for each token in each sequence to a set of logits

        #Label Preparation
        #Initializing a tensor for the labels with dimensions (batch_size, max_length), filled with zeros.
        labels_batch = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        #List of labels into a tensor
        for i, sentence in enumerate(sentences):
            labels = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            #Assigning the label tensor labels to the corresponding row in labels_batch for the current sentence.
            labels_batch[i, :lengths[i]] = labels

        if self.crf:
            #CRF needs mask to find out real and padded tokens
            mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1
            #After this loop, the mask tensor has True values marking valid tokens and False values marking padding tokens in each sentence.

            #Negative likelihood: How well model's predicted labels match true labels (This is loss, model will try to reduce,update its weights )
            loss = -self.crf(logits_batch, labels_batch, mask, reduction='mean')
            #takes out predicted IDs
            pred_ids = self.crf.decode(logits_batch, mask)
            #Converts them to labels (PER,ORG)
            pred = [[constants.ID_TO_LABEL[i] for i in ids] for ids in pred_ids]
        else:
            #CRF not used , initialize loss to zero
            loss = torch.zeros(1, device=self.device)
            #Calculate cross-entropy loss for everything (Summing losses over batch)
            for logits, labels, length in zip(logits_batch, labels_batch, lengths):
                loss += F.cross_entropy(logits[:length], labels[:length], reduction='sum')
            #Average the total loss
            loss /= batch_size
            #Computing the predicted class IDs by taking the argmax over the logits for each token in each sentence.
            pred_ids = torch.argmax(logits_batch, dim=2).tolist()
            pred = [[constants.ID_TO_LABEL[i] for i in ids[:length]] for ids, length in zip(pred_ids, lengths)]

        return loss, pred

    def itr_forward(self, pairs: List[MyPair]):#pairs of sentence and corresponding image defined in dataset.py for a batch of sentences coming from utils.py
        text_batch = [pair.sentence.text for pair in pairs]
        inputs = self.tokenizer(text_batch, padding=True, return_tensors='pt').to(self.device)
        outputs = self._bert_forward_with_image(inputs, pairs)#tokenised sentences, sentencea and imagea are sent
        feats = outputs.last_hidden_state[:, CLS_POS]
        logits = self.aux_head(feats)

        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        pred = torch.argmax(logits, dim=1).tolist()

        return loss, pred