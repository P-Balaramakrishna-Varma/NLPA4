from dataloading import *

import torch



class ELMo(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.forward_lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.projection = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.forward_lstm(x)
        logits_f = self.projection(h)
        return logits_f
    

#ignore index 0
def train(model, train_loader, optimizer, loss_func, device):
    model.train()
    for X, y in train_loader:
        X = X.to(device)
        seq_len = X.shape[1]
        
        # Forward pass
        logits_f = model(X)
        classes = X[:, 1:]
        logits_f = torch.transpose(logits_f, 1, 2)
        loss_value_f = loss_func(logits_f, classes)
        loss_value = loss_value_f
        
        # Backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()


def eval(model, data_loader, loss_func, device):
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            seq_len = X.shape[1]

            logits_f = model(X)
            
            classes = X[:, 1:]
            logits_f = torch.transpose(logits_f, 1, 2)
            loss_value_f = loss_func(logits_f, classes)
            loss_value = loss_value_f
            total_loss += loss_value.item() / seq_len
            total_count += X.shape[0]
    return total_loss / total_count