import torch
from torch import nn
import copy
from utils.data_utils.json_handled import JsonHandler
from torch.autograd import Variable
from torch.nn import functional as F

class multiHead(nn.Module):
    def __init__(self,model_name,n_model,d_embd,n_events,hidden_size,n_layer = 2, dataset_name = 'HDFS'):
        super(multiHead, self).__init__()
        model = self.create_model(model_name,input_size=int(d_embd),hidden_size =hidden_size,n_layer =n_layer)
        self.models = self.clones(model,n_model)

        self.Ws = self.para_clones(nn.Parameter(torch.zeros(hidden_size)), n_model)
        self.bs = self.para_clones(nn.Parameter(torch.zeros(hidden_size)), n_model)

        self.fc_input = nn.Linear(128, d_embd*n_model)
        self.fc_out = nn.Linear(hidden_size*n_model, n_events)
        self.model_name = model_name
        self.n_head = n_model
        self.input_size = int(d_embd)
        self.hidden_size = hidden_size
        self.n_layer =  n_layer
        event2vec = JsonHandler('../data/multiHeadRNN/' + dataset_name + '_logevent2vec.json').read_json()
        self.embd = self.init_embeding(event2vec)
        self.init_weight()

    def forward(self,x,device = 'cuda'):
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        if x.size(2) == 1:
           x = x.squeeze(2)
        x = self.fc_input(self.embd(x.long())).to(device)# [batch_size,seq_len] => [batch_size,seq_len,embd_dim]
        reshape_x = x.view(batch_size, -1, self.n_head, self.input_size).transpose(1, 2)
        hidden_outs = list()
        h0 = torch.zeros(self.n_layer, reshape_x.size(0),
                         self.hidden_size).to(reshape_x.device)
        c0 = torch.zeros(self.n_layer, reshape_x.size(0),
                         self.hidden_size).to(reshape_x.device)
        for i,rnn_model in enumerate(self.models):
           if self.model_name == 'gru':
               hid_outs,_ = rnn_model(reshape_x[:, i, :, :], (h0))
           else:
               hid_outs, _ = rnn_model(reshape_x[:, i, :, :], (h0, c0))
           hid_out = self.Ws[i]*hid_outs[:,-1,:]+self.bs[i]
           hidden_outs.append(hid_out)
        out = torch.cat(hidden_outs,dim=1)
        out = self.fc_out(out)
        return out

    def create_model(self,model_name,input_size,hidden_size,n_layer):
        '''
        LSTM，GRU
        :param rnn_model:
        :return:
        '''
        model = None
        if model_name == 'lstm':
            model = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=n_layer,
                           batch_first=True)
        elif model_name == 'gru':
            model = nn.GRU(input_size,
                            hidden_size,
                            num_layers=n_layer,
                            batch_first=True)
        return model
    def clones(self,module,N):
        assert  module is not None,'模型没有被初始化！'
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    def para_clones(self,parameter,N):
        assert parameter is not None,'模型没有被初始化！'
        return nn.ParameterList([copy.deepcopy(parameter) for _ in range(N)])
    def init_weight(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.kaiming_normal_(p)
    def init_embeding(self, event2vec):
        # pretrained_embeddings = dict()
        # for key, value in event2vec.items():
        #     pretrained_embeddings[int(key)] = torch.tensor(event2vec[key])

        embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor([vec for vec in event2vec.values()], dtype=torch.float),
            freeze=False
        )
        return embedding_layer
