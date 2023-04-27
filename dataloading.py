from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import torch
import datasets
import torchtext
import string
from tqdm import tqdm


def preprocess_sentence(sentence, ps):
    token_words = word_tokenize(sentence)
    token_words_no_punc = [word for word in token_words if word not in set(string.punctuation)]
    stem_words = [ps.stem(word) for word in token_words_no_punc]
    return stem_words


def create_vocab_stt(dataset):
    ps = PorterStemmer()
    stemed_words = []
    for sample in dataset:
        stems = preprocess_sentence(sample['sentence'], ps)
        stems = [[stem] for stem in stems]
        stemed_words.extend(stems)
    vocab = torchtext.vocab.build_vocab_from_iterator(stemed_words, specials=["<pad>", "<unk>"], special_first=True, min_freq=2)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab):
        super().__init__()
        self.dataset = dataset
        self.vocab = vocab
        self.ps = PorterStemmer()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = preprocess_sentence(sample['sentence'], self.ps)
        x = [self.vocab[token] for token in x]
        y = round(sample['label'])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_vocab_nli(dataset):
    ps = PorterStemmer()
    stemed_words = []
    for i in tqdm(range(10000)):
        sample = dataset[i]
        stems = preprocess_sentence(sample['premise'], ps) + preprocess_sentence(sample['hypothesis'], ps)
        stems = [[stem] for stem in stems]
        stemed_words.extend(stems)
    vocab = torchtext.vocab.build_vocab_from_iterator(stemed_words, specials=["<pad>", "<unk>"], special_first=True, min_freq=2)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab):
        super().__init__()
        self.dataset = dataset
        self.vocab = vocab
        self.ps = PorterStemmer()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x1 = preprocess_sentence(sample['premise'], self.ps)
        x1 = [self.vocab[token] for token in x1]
        x2 = preprocess_sentence(sample['hypothesis'], self.ps)
        x2 = [self.vocab[token] for token in x2]
        y = sample['label']
        return torch.tensor(x1, dtype=torch.long), torch.tensor(x2, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn_stt(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    assert (x.shape[0] == y.shape[0])
    assert (len(y.shape) == 1)
    assert (len(x.shape) == 2)
    return x, y


def collate_fn_nli(batch):
    x1 = [item[0] for item in batch]
    x2 = [item[1] for item in batch]
    y = [item[2] for item in batch]
    x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True, padding_value=0)
    x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x1, x2, y


if __name__ == '__main__':
    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset["train"])
    train_dataset = SSTDataset(dataset["train"], vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_stt)
    for X, y in tqdm(train_loader):
        pass

    dataset = datasets.load_dataset('multi_nli')
    vocab = create_vocab_nli(dataset["train"])
    train_dataset = NLIDataset(dataset["train"], vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_nli)
    for X1, X2, y in tqdm(train_loader):
        pass