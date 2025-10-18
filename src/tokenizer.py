import regex as re
from typing import Dict, Tuple, Iterable, List
from tqdm import tqdm

from src.utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN

def get_pairs(ids: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """ Return a set of pairs in int ids """
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """ Update the ids by merging the pairs """
    new_ids = []
    i = 0
    while i < len(ids):
        curr_pair = tuple(ids[i:i+2])
        if curr_pair == pair:
            new_ids.append(new_id)
            i += 1
        else:
            new_ids.append(ids[i])
        i += 1
    return new_ids

def _fix_vocab(vocab_i_to_b: Dict[int, bytes], vocab_b_to_i: Dict[str, bytes]):
    """ Make sure all bytes are in the vocab """
    for i in range(256):
        byte = bytes([i])
        if byte not in vocab_b_to_i:
            vocab_b_to_i[byte] = len(vocab_b_to_i)
            vocab_i_to_b[len(vocab_i_to_b)] = byte
    return dict(id_to_types=vocab_i_to_b, bytes_to_id=vocab_b_to_i)

class Tokenizer:
    """ A simple tokenizer that uses a vocabulary and merges to encode and decode text.
    It supports special tokens and can be initialized from vocab and merges files.
    """
    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: Iterable[Tuple[bytes, bytes]], 
                 special_tokens: Iterable[str]=None):
        """ Initialize the tokenizer with a vocabulary and merges.  
        Args:
            vocab: A dictionary mapping token ids to bytes.
            merges: An iterable of tuples containing pairs of bytes to be merged.
            special_tokens: An iterable of special tokens to be added to the vocabulary.            
        """
        self.vocab = {}
        # from id to bytes
        self.vocab['id_to_types'] = vocab
        # from bytes to id
        self.vocab['bytes_to_id'] = {v: k for k, v in vocab.items()}
        # all single byte tokens should be in the vocab
        self.vocab = _fix_vocab(self.vocab['id_to_types'], self.vocab['bytes_to_id'])

        # reorganzie merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            # b'a, b'b'
            id_pair = (self.vocab['bytes_to_id'][a], self.vocab['bytes_to_id'][b])
            # (10, 20) = b'a' + b'b' = b'ab'
            # what if a+b does not exist in bytes_to_id?
            self.merges[id_pair] = self.vocab['bytes_to_id'][a+b]
        
        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab['bytes_to_id']:
                    self.vocab['bytes_to_id'][token_byte] = len(self.vocab['bytes_to_id'])
                    self.vocab['id_to_types'][len(self.vocab['id_to_types'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['id_to_types'])
                else:
                    self.special_tokens[token] = self.vocab['bytes_to_id'][token_byte]
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, **kwargs):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab['id_to_types'])
    
    def _encode_chunk(self, text: str) -> List[int]:
        """
        Encode the text without special tokens.
        """
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else:
            text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
            result = []
            for chunk in text_chunks:
                #text_bytes = chunk.encode("utf-8")
                ids = [self.vocab['bytes_to_id'][bytes([b])] for b in chunk.encode("utf-8")]
                while len(ids)>=2:
                    pairs = get_pairs(ids)
                    high_priority_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
                    if high_priority_pair not in self.merges:
                        break
                    new_id = self.merges[high_priority_pair]
                    ids = update(ids, high_priority_pair, new_id)
                result.extend(ids)
            return result


    def encode(self, text: str, progress_bar: bool=False) -> List[int]:
        """
        Encode the text into a list of token ids.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]
        ids = []
        for chunk in tqdm(special_split_chunk, disable=not progress_bar,
                          desc=f"Encoding {len(special_split_chunk)} documents"):
            ids += self._encode_chunk(chunk)
        return ids
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[List[int]]:
        """
        Encode the texts into a list of token ids.
        """
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: List[int]) -> str:
        """
        Decode the token ids into the original text.
        """
        text_bytes = b''.join([self.vocab['id_to_types'][i] for i in ids])
        return text_bytes.decode("utf-8", errors="replace")