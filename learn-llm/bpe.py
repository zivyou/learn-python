import collections
import re


def get_vocabulary(text: str) -> dict[str, int]:
    vocabs = collections.defaultdict(int)
    words = text.strip().split()
    for word in words:
        vocabs[' '.join(list(word)) + ' </w>'] += 1
    return vocabs

def get_pairs(vocabs: dict[str, int]) -> dict[tuple[str,str], int]:
    pairs = collections.defaultdict(int)
    for word, count in vocabs.items():
        symbols = word.strip().split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += 1
    return pairs

def merge_vocabs(pair: tuple[str,str], vocabs: dict[str, int]) -> dict[str, int]:
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocabs:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = vocabs[word]
    return v_out

def get_tokens(vocabs: dict[str, int]) -> dict[str, int]:
    tokens = collections.defaultdict(int)
    for word, count in vocabs.items():
        word_tokens = word.strip().split()
        for token in word_tokens:
            tokens[token] += 1
    return tokens

def main():
    text = f"""loydHub is the fastest way to build, train and deploy deep learning models.
     Build deep learning models in the cloud. Train deep learning models."""
    print('='*20)
    vocabs = get_vocabulary(text)
    print(f"vocabs: {vocabs}")
    print('='*20)
    tokens = get_tokens(vocabs)
    print(f"tokens: {tokens}")
    print('='*20)

    for i in range(10):
        pairs = get_pairs(vocabs)
        if not pairs:
            break
        best = max(pairs, key=lambda x: pairs[x])
        print("="*20)
        print(f"best pair: {best}")
        vocabs = merge_vocabs(best, vocabs)
        print("="*20)
        print(f"new vocabs: {vocabs}")
        tokens = get_tokens(vocabs)
        print("="*20)
        print(f"tokens: {tokens}")
        print('='*20)


if __name__ == '__main__':
    main()
