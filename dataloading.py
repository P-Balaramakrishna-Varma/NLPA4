from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import torch
import datasets
import torchtext
import string


def preprocess_sentence(sentence, ps):
    token_words = word_tokenize(sentence)
    token_words_no_punc = [word for word in token_words if word not in set(string.punctuation)]
    stem_words = [ps.stem(word) for word in token_words_no_punc]
    return stem_words


def create_vocab_stt(dataset):
    ps = PorterStemmer()
    stemed_words = []
    for sample in dataset['train']:
        stems = preprocess_sentence(sample['sentence'], ps)
        stems = [[stem] for stem in stems]
        stemed_words.extend(stems)
    vocab = torchtext.vocab.build_vocab_from_iterator(stemed_words, specials=["<pad>", "<unk>"], special_first=True)
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
        y = round(sample['label']) + 1     # 0 -> 1, 1 -> 2 (for cross entropy) class 0 for padding
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



if __name__ == '__main__':
    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset)
    train_dataset = SSTDataset(dataset["train"], vocab)
    print(len(dataset["train"]), len(train_dataset))
    print(train_dataset[0])