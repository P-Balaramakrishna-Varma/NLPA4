from dataloading import *

import torch



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
    for X, y in tqdm(train_loader):
        # data gathering
        X, y = X.to(device), y.to(device)
        
        # forward pass
        logits = model(X)
        loss = loss_func(logits, y)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset["train"])
    train_dataset = SSTDataset(dataset["train"], vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_stt)

    model = ELMoSentiment(len(vocab), 300, 400, 800).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    train_stt(model, train_loader, optimizer, loss_func, device)