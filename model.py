import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        self.n_layers = num_layers
        # get the hidden layer size
        self.hidden_dim = hidden_size
        
        # embedding layer that converts back the vocab dict
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer; inputs: the embedded vectors, outputs: hidden state of hidden dim, dropout = 0 
        self.Lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout = 0)
        
        # Fully connected layer
        self.fcl = nn.Linear(hidden_size, vocab_size)
        
        # initialize a hidden state(n_layers, batch_size, hidden_dim)
        #self.hidden = self.init_hidden()
        
    #def init_hidden(self):
            
    #    return (torch.zeros(1, 1, self.hidden_dim),
    #           torch.zeros(1, 1, self.hidden_dim))

    
    def forward(self, features, captions):
        #embedding = self.embedding(captions)
        embedding = self.embed(captions[:,:-1])
        #captions = [10,13], features = [10, 256]
        rnn_embedding = torch.cat((features.unsqueeze(dim=1), embedding), dim=1)
        # extract lstm output
        lstm_out, hidden_out = self.Lstm(rnn_embedding)
        # pass lstm output through fully connected layer
        output = self.fcl(lstm_out)
        
        return output
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_list = []

        #states = self.init_hidden(inputs.shape[0])
    
      #  states = (torch.zeros(1, 1, self.hidden_dim),
      #         torch.zeros(1, 1, self.hidden_dim))

        for i in range(max_len):
            
            lstm_out, states = self.Lstm(inputs, states)
            
            output = self.fcl(lstm_out)
            #print(output)
            #output = output.squeeze(1)
            #out_pred_score = output.argmax(dim=1)
            #out_pred = output.max(1)[1]
            out_prob, out_pred_score = output.max(2)
            #out_pred_score = out_pred[0]
            sampled_list.append(out_pred_score.item())

            #output = self.fcl(Lstm_out.squeeze(1))
            
            inputs = self.embed(out_pred_score)
            
        return sampled_list
        