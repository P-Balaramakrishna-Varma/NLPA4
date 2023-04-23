from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import torch
import datasets
import torchtext



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





if __name__ == '__main__':
    dataset = datasets.load_dataset('sst')
    vocab = create_vocab_stt(dataset)