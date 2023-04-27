from dataloading import *

import torch



class ELMo(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)   # replace with word to vec
        self.forward_lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, dropout=dropout)

    def forward(self, x):
        # global embeddings
        x = self.embedding(x)

        # forward contexual embedding
        h, _ = self.forward_lstm(x)
        return h
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset["train"])
    train_dataset = SSTDataset(dataset["train"], vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_stt)

    model = ELMo(len(vocab), 300, 400).to(device)
    for X, y in tqdm(train_loader):
        X = X.to(device)
        y = y.to(device)
        logits = model(X)