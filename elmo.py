from dataloading import *

import torch
import matplotlib.pyplot as plt




class ELMo(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0):
        super().__init__()
        # gloabl embedding
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)   # replace with word to vec
        
        # contexual embedding
        self.forward_lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        self.forward_lstm_2 = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        self.bacward_lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        self.bacward_lstm_2 = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        
        # task specific
        self.weights = torch.nn.Parameter(torch.ones(2, dtype=torch.float))
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # global embeddings
        x = self.embedding(x)

        # forward contexual embedding
        f1, _ = self.forward_lstm_1(x)
        f2, _ = self.forward_lstm_2(f1)

        # backward contexual embedding
        b1, _ = self.bacward_lstm_1(x.flip(1))
        b2, _ = self.bacward_lstm_2(b1)

        # concat
        e1 = torch.cat([f1, b1.flip(1)], dim=2)
        e2 = torch.cat([f2, b2.flip(1)], dim=2)

        # combination
        w = self.weights.softmax(dim=0)
        embeddings = self.scale * (w[0] * e1 + w[1] * e2)
        return embeddings


class ELMoSentiment(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sen_embd_dim, dropout=0):
        super().__init__()
        self.elmo = ELMo(vocab_size, embedding_dim, hidden_dim, dropout)
        self.linear_classifier = torch.nn.Linear(sen_embd_dim, 2)

    def forward(self, x):
        # word embeddings
        x = self.elmo(x)
        
        # sentence_embeddings
        stentence_embedding = x.mean(dim=1)

        # classification
        logits = self.linear_classifier(stentence_embedding)
        return logits


def train_stt(model, train_loader, optimizer, loss_func, device):
    model.train()
    for X, y in train_loader:
        # data gathering
        X, y = X.to(device), y.to(device)
        
        # forward pass
        logits = model(X)
        loss = loss_func(logits, y)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_stt(model, dataloader, loss_func, device):
    model.eval()
    loss = 0
    count = 0
    for X, y in dataloader:
        # data gathering
        X, y = X.to(device), y.to(device)
        
        # forward pass
        logits = model(X)
        loss += loss_func(logits, y).item() 
        count += 1

    return loss / count   


def visulaize_losses(train_losses, valid_losses):
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.legend()
    plt.show()


def sst_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data creation part
    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset["train"])
    
    train_dataset = SSTDataset(dataset["train"], vocab)
    test_dataset = SSTDataset(dataset["test"], vocab)
    valid_dataset = SSTDataset(dataset["validation"], vocab)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_stt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_stt)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_stt)

    # Model, optimizer, loss funnction, hyperparameters
    model = ELMoSentiment(len(vocab), 300, 400, 800).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()
    epochs = 1

    # Training
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(epochs)):
        train_stt(model, train_loader, optimizer, loss_func, device)
        train_losses.append(eval_stt(model, train_loader, loss_func, device))
        valid_losses.append(eval_stt(model, valid_loader, loss_func, device))
    
    return train_losses, valid_losses




if __name__ == "__main__":
    train_losses, valid_losses = sst_train()
    #visulaize_losses(train_losses, valid_losses)
    print(train_losses , valid_losses)