import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # load in resnet pretrained model
        resnet = models.resnet50(pretrained=True)
        # freeze the features from being trained 
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        print('modules ='+str(modules))
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,batch_size = 64, drop_out = 0.2):
        super(DecoderRNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers= num_layers , batch_first = True )
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
#         self.embed_size = embed_size
#         self.vocab_size = vocab_size

        
       
        
        self.dropout = nn.Dropout(drop_out)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state 
        self.hidden = self.init_hidden(batch_size)
        
    def init_hidden(self,batch_size):
        # The axes dimensions are (num_layers, batch_size, hidden_size). batch_size explicitly made = 1
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        
        


        
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embed = self.embedding_layer(captions)
        #reshape features
        features = features.unsqueeze(1)
        #combine features and embed
        #embed is the input tensor
        embed = torch.cat((features, embed), dim =1)
        # lstm_outputs shape : (batch_size, seq_len, hidden_size)
        lstm_outputs, self.hidden = self.lstm(embed)
        lstm_outputs_shape = lstm_outputs.shape
        lstm_outputs_shape = list(lstm_outputs_shape)
        #get the probability for the next word
        #vocab outputs shape ; (batch_size*seq, vocab_size)
        vocab_outputs = self.linear(lstm_outputs)        
        return vocab_outputs

    
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        inputs = torch.tensor(inputs)
        for i in range(max_len):
            input_,states = self.lstm(inputs,states)
            
           
            input_ = self.linear(input_)
            
            input_ = input_.squeeze(1)

            value, index = torch.max(input_,dim=1)
            pred_index = index.cpu().numpy()[0].item()
            output.append(pred_index)
            
            if (pred_index == 1):
                break
            
            inputs = self.embedding_layer(index)
            inputs = inputs.unsqueeze(1)
        
        return output                