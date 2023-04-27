from dataloading import *

import torch
import matplotlib.pyplot as plt
import torchmetrics



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


def get_pred_stt(model, dataloader, device):
    model.eval()
    preds, acts = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds.append(logits.argmax(dim=1))
        acts.append(y)
    return torch.cat(preds, dim=0), torch.cat(acts, dim=0)


def get_stats_sst(preds, acts):
    acc = torchmetrics.functional.accuracy(preds, acts, task='binary')
    f1_score = torchmetrics.functional.f1_score(preds, acts, task='binary')
    precision = torchmetrics.functional.precision(preds, acts, task='binary')
    recall = torchmetrics.functional.recall(preds, acts, task='binary', average='macro')
    return acc, precision, recall, f1_score


def visulaize_losses_stt(train_losses, valid_losses):
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

    # Evaluation
    preds, acts = get_pred_stt(model, test_loader, device)
    stats = get_stats_sst(preds, acts)
    return train_losses, valid_losses, stats







class ELMoNli(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sen_embd_dim, mlp_hidden, dropout=0):
        super().__init__()
        self.elmo = ELMo(vocab_size, embedding_dim, hidden_dim, dropout)
        self.mlp1 = torch.nn.Linear(2 * sen_embd_dim, mlp_hidden)
        self.mlp2 = torch.nn.Linear(mlp_hidden, 3)

    def forward(self, x1, x2):
        # word embeddings
        x1, x2 = self.elmo(x1), self.elmo(x2)
  
        # sentence_embeddings
        premise_emd, hyothesis_emd = x1.mean(dim=1), x2.mean(dim=1)
        final_emd = torch.cat((premise_emd, hyothesis_emd), dim=1)

        # classification
        logits = self.mlp2(self.mlp1(final_emd))
        return logits


def train_nli(model, train_loader, optimizer, loss_func, device):
    model.train()
    for X1, X2, y in train_loader:
        # data gathering
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        
        # forward pass
        logits = model(X1, X2)
        loss = loss_func(logits, y)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_nli(model, dataloader, loss_func, device):
    model.eval()
    loss = 0
    count = 0
    for X1, X2, y in dataloader:
        # data gathering
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        
        # forward pass
        logits = model(X1, X2)
        loss += loss_func(logits, y).item() 
        count += 1

    return loss / count   


def get_pred_nli(model, dataloader, device):
    model.eval()
    preds, acts = [], []
    for X1, X2, y in dataloader:
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        logits = model(X1, X2)
        preds.append(logits.argmax(dim=1))
        acts.append(y)
    return torch.cat(preds, dim=0), torch.cat(acts, dim=0)


def get_stats_nli(preds, acts):
    acc = torchmetrics.functional.accuracy(preds, acts, task='multiclass', num_classes=3)
    f1_score = torchmetrics.functional.f1_score(preds, acts, task='multiclass', num_classes=3)
    precision = torchmetrics.functional.precision(preds, acts, task='multiclass', num_classes=3, average='macro')
    recall = torchmetrics.functional.recall(preds, acts, task='multiclass', num_classes=3, average='macro')
    return acc, precision, recall, f1_score


def nli_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data creation part
    dataset = datasets.load_dataset('multi_nli')
    vocab = create_vocab_nli(dataset["train"])
    
    train_dataset = NLIDataset(dataset["train"], vocab)
    train_sampler = torch.utils.data.RandomSampler(range(len(train_dataset)//100))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn_nli, sampler=train_sampler)

    test1_dataset = NLIDataset(dataset["validation_matched"], vocab)
    test1_sampler = torch.utils.data.RandomSampler(range(len(test1_dataset)//100))
    test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=64, collate_fn=collate_fn_nli, sampler=test1_sampler)

    test2_dataset = NLIDataset(dataset["validation_mismatched"], vocab)
    test2_sampler = torch.utils.data.RandomSampler(range(len(test2_dataset)//100))
    test2_dataloader = torch.utils.data.DataLoader(test2_dataset, batch_size=64, collate_fn=collate_fn_nli, sampler=test2_sampler)

    # Model, optimizer, loss funnction, hyperparameters
    model = ELMoNli(len(vocab), 300, 400, 800, 50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()
    epochs = 1

    train_losses, test1_losses, test2_losses = [], [], []
    for _ in tqdm(range(epochs)):
        train_nli(model, train_dataloader, optimizer, loss_func, device)
        train_losses.append(eval_nli(model, train_dataloader, loss_func, device))
        test1_losses.append(eval_nli(model, test1_dataloader, loss_func, device))
        test2_losses.append(eval_nli(model, test2_dataloader, loss_func, device))
    
    # Evaluation
    preds, acts = get_pred_nli(model, test1_dataloader, device)
    stats = get_stats_nli(preds, acts)

    return train_losses, test1_losses, test2_losses, stats


def visulaize_losses_stt(train_losses, test1_loss, test2_loss):
    plt.plot(train_losses, label="train")
    plt.plot(test1_loss, label="test1")
    plt.plot(test2_loss, label="test2")
    plt.legend()
    plt.show()






if __name__ == "__main__":
    out1, out2, out3, stats = nli_train()
    print("acc:", stats[0])
    print("precion:", stats[1])
    print("recall:", stats[2])
    print("f1_score:", stats[3])