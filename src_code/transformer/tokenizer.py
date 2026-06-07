"""
A simple byte-pair encoding tokenizer.
First, a tokenizer of strings is built, then generalized to UTF-8 strings.
"""

from typing import List, Union, Dict, Tuple

class Trie:
    """Prefix tree.
    A prefix tree is iterable, in which case it yields pairs (string, token) for each string stored in the tree.
    """
    class Node:
        """A node in a prefix tree.
        To each node, we associate the path allowing to reach the node from the root of the tree.
        This path corresponds to a string, because each node to node transition is interpreted as a character.      
        _children[char] is the node obtained when applying the transition char to the current node.
        Entries in the dictionary are the strings associated to the nodes with a not None token.
        """
        def __init__(self):
            self._children: Dict[str, Trie.Node] = {}
            self._token: Union[int, None] = None

    def __init__(self):
        self._root: Trie.Node = Trie.Node()
        # the empty string is associated the UNK token: it is returned when a string key is not found in the dictionary
        self._root._token = -1
    
    def __iter__(self):
        """Iterate over pairs (string, token) for each string stored in the tree."""
        nodes = [("", self._root)]
        while nodes:
            new_nodes = []
            for sub, node in nodes:
                if node._token is not None:
                    yield sub, node._token
                for char, child in node._children.items():
                    new_nodes.append((sub + char, child))
            nodes = new_nodes

    def add(self, s: str, token: int):
        """Add the string `s` as an entry in the tree, with `token` as its associated token.

        Parameters
        ----------
        s : str
            _description_
        """
        node = self._root
        for i, char in enumerate(s):
            try:
                node = node._children[char]
            except KeyError:
                break
        for char in s[i:]:
            node._children[char] = Trie.Node()
            node = node._children[char]
        node._token = token

    def largest_sub(self, s: str) -> str:
        """Find the largest substring in `s` starting at position 0 that is part of the dictionary.

        Parameters
        ----------
        s : str
            the input string

        Returns
        -------
        str
            the dictionary entry that is the longest match
        """

        sub = [""]
        index = 0
        node = self._root
        last_match = 1

        while True:
            try:
                next_char = s[index]
            except IndexError:
                return ''.join(sub[:last_match])
            
            # test whether adding the next character to the substring yields a prefix stored in the dictionary
            try:
                node = node._children[next_char]
                sub.append(next_char)
                index += 1
                if node._token is not None:
                    last_match = index + 1
            except KeyError:
                return ''.join(sub[:last_match])

    def get_token(self, sub: str) -> int:
        """Return the token associated to the stored string `sub`.

        Parameters
        ----------
        sub : str
            the string we search in the dictionary ; given that we know it is part of it
        """
        node = self._root
        for char in sub:
            node = node._children[char]
        return node._token

class Tokenizer:
    """
    A tokenizer can be seen as a lookup table, mapping a string to a unique integer (starting from 0).
    The strings registered in the table correspond to entries in a dictionary, and the associated integers are called the tokens.
    The number of tokens is the vocabulary size of the tokenizer.

    When converting a string into a sequence of tokens, the string is read from left to right.
    At each position, the largest substring matching an entry in the lookup table will be replaced by the corresponding token.

    To compress the tokenizer and efficiently encoding a string, the lookup table is implemented as a trie (prefix tree).
    """
    def __init__(self, vocab_size: int, vocab: Union[Trie, None]=None):
        self.vocab_size = vocab_size
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Trie()

    def str_to_tokens(self, s: str) -> List[int]:
        """Parse a string into a sequence of tokens.

        Parameters
        ----------
        s : str
            the string to be converted into a sequence of tokens (integers)
        """
        res = []
        index = 0
        while index < len(s):
            # find the largest substring in `s` starting at `index` that is part of the vocabulary
            sub = self.vocab.largest_sub(s[index:])

            # replace the substring by its corresponding token and update the index
            res.append(self.vocab.get_token(sub))
            index += max(1, len(sub))
        return res

    def train(self, corpus: str):
        """Builds the vocabulary (mapping substring --> token) out of the substrings found in the corpus.

        Parameters
        ----------
        corpus : str
            the corpus (string) from which the vocabulary is built
        """

        # the first pass will identify the individual characters present in the corpus, and convert it into a sequence of integers

        tok_to_char: Dict[int, Union[str, int]] = {}
        found_chars = {}
        new_corpus: List[int] = []
        for char in corpus:
            if char not in found_chars:
                new_tok = len(tok_to_char)
                found_chars[char] = new_tok
                tok_to_char[new_tok] = char
            new_corpus.append(found_chars[char])
        corpus = new_corpus

        # now at each pass of `new_corpus`, we will identify the most frequent pair of adjacent integers, as well as their positions in the sequence
        # then, we will replace the occurrences of the most frequent pair by a new token (integer)
        while len(tok_to_char) < self.vocab_size:
            # found_chars[pair] = list of the positions at which `pair` is found in `new_corpus`
            found_chars: Dict[Tuple[int, int], List[int]] = {}

            # the most frequent pair and its count
            most_freq_pair = None
            pair_count = 0

            pos = 0
            while pos < len(corpus) - 1:
                pair = (corpus[pos], corpus[pos + 1])
                try:
                    found_chars[pair].append(pos)
                except KeyError:
                    found_chars[pair] = [pos]
                if pair_count < len(found_chars[pair]):
                    pair_count = len(found_chars[pair])
                    most_freq_pair = pair
                
                # take care that, if a character is repeated three times in the corpus, it should not count as 2 pairs
                pos += 1 + (pair[0] == pair[1])
            
            # replace the occurrences of the most frequent pair with a new token
            new_tok = len(tok_to_char)
            tok_to_char[new_tok] = most_freq_pair
            pos_list = found_chars[most_freq_pair]
            new_corpus = corpus[:pos_list[0]]
            new_corpus.append(new_tok)
            for k, pos in enumerate(pos_list[:-1]):
                new_corpus.extend(corpus[pos + 2: pos_list[k + 1]])
                new_corpus.append(new_tok)
            new_corpus.extend(corpus[pos_list[-1] + 2:])

            corpus = new_corpus

        # determine the words to insert in the vocabulary, together with their associated token
        # for this, we unfold each token in `tok_to_char` as a string (recursive implementation)
        def find_word(tok: int) -> str:
            """find the word (string) associated to a token in `tok_to_char`"""
            val = tok_to_char[tok]
            if isinstance(val, tuple):
                return find_word(val[0]) + find_word(val[1])
            else:
                return val
        
        # finally, add the words to the vocabulary
        for tok in tok_to_char:
            self.vocab.add(find_word(tok), tok)

    def tokens_to_str(self, seq: List[int]) -> str:
        """Parse a sequence of tokens into a string.

        Parameters
        ----------
        seq : List[int]
            the sequence of tokens (special tokens have negative values)

        Returns
        -------
        str
            the output string (special characters are enclosed between angle brackets <>)
        """
        tok_to_str = {}
        for entry, token in self.vocab:
            tok_to_str[token] = entry
        tok_to_str[-1] = "<UNK>"

        res = []
        for tok in seq:
            res.append(tok_to_str[tok])
        return ''.join(res)

def test_tok_encoding():
    """Test that our tokenizer correctly encodes strings.
    """
    words_to_insert = ["he", "hello"]
    vocab_size = len(words_to_insert)

    vocab = Trie()
    for token, word in enumerate(words_to_insert):
        vocab.add(word, token)

    # check the vocabulary has been correctly set up
    for word, token in vocab:
        print("entry:", word)
        print("token:", token)
        print()

    tok = Tokenizer(vocab_size, vocab=vocab)

    s = "hello world"
    print("input string:", s)
    print("parsed string:", tok.str_to_tokens(s))

def test_tok_building():
    """Test that our tokenizer is correctly built out of a corpus string.
    """
    tok = Tokenizer(vocab_size=6)
    corpus = "aaabdaaabac"
    tok.train(corpus)
    
    # print all entries stored in the tokenizer
    for word, token in tok.vocab:
        print("stored entry:", word)
        print("token:", token)
        print()

def final_test():
    """Train a tokenizer on a corpus, then use it to encode and decode back a new string.
    """

    corpus = """In computing, byte-pair encoding (BPE), or digram coding, is an algorithm, first described in 1994 by Philip Gage, """ + \
            """for encoding strings of text into smaller strings by creating and using a translation table. """ + \
            """A slightly modified version of the algorithm is used in large language model tokenizers. """ + \
            """The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte """ + \
            """that was not contained in the initial dataset. A lookup table of the replacements is required to rebuild the initial dataset. """ + \
            """The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters """ + \
            """(including single digits or single punctuation marks) to whole words (even long compound words)"""

    tok = Tokenizer(vocab_size=100)
    tok.train(corpus)
    # print all entries stored in the tokenizer
    for word, token in tok.vocab:
        print("stored entry:", word)
        print("token:", token)
        print()

    new_sentence = """The original BPE algorithm operates by iteratively replacing the most common contiguous """ + \
                """sequences of characters in a target text with unused 'placeholder' bytes."""

    parsed_string = tok.str_to_tokens(new_sentence)
    print("input string:", new_sentence)
    print()
    print("parsed string:", parsed_string)
    print()
    print("string parsed back:", tok.tokens_to_str(parsed_string))

if __name__ == "__main__":
    final_test()
