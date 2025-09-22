"""
Train a byte pair encoding (BPE) tokenizer on a text file for a vocab dictionary.
"""
import regex as re
from typing import Iterable
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter
import concurrent.futures
from typing import BinaryIO
import os
from heapq import heappush, heappop

from src.utils.io import GPT2_PRETOKENIZER_PATTERN

GPT2_PRETOKENIZER_REGEX = re.compile(GPT2_PRETOKENIZER_PATTERN)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def find_pretokens(text: str):
    """
    Find the pretokens in the text.
    """
    # logging.info(f"Pre-tokenizing the text of length {len(text)}")
    return Counter(GPT2_PRETOKENIZER_REGEX.findall(text))

def find_pretokens_with_special(text: str):
    print(f"PID {os.getpid()} processing text of length {len(text)}")
    pretokens = Counter()
    for cont in text.split("<|endoftext|>"):
        pts = find_pretokens(cont)
        pretokens.update(pts)
    return pretokens


def _read_text_file(input_path: str, num_worker: int, special_tokens: Iterable[str]):
    """
    Read the text file at the given path.
    Return the text as pretoken frequency table.
    """
    
    logging.info("Initializing pretoken frequency table")
    pretokens = Counter()

    if num_worker == 1:
        # Read the input text file
        with open(input_path, "r") as file:
            text = file.read()        
            # TODO: Use special tokens in input arguments. Hardcode for now.
            # Special_tokens is not counted but why it is vocab?
            for cont in text.split("<|endoftext|>"):
                pts = find_pretokens(cont)
                pretokens.update(pts)
    else:        
        """chunk_size = len(text) // num_worker
        print(f"Splitting text into {num_worker} chunks of size {chunk_size}")
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            pretokens = executor.map(_find_pretokens, text_chunks)
        """
        #text_chunks =[]
        #for cont in text.split("<|endoftext|>"):
        #    text_chunks.append(cont)
        #text = ''.join(text_chunks)

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_worker, b"<|endoftext|>")
            i = 0
            text_chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                i += 1
                text_chunks.append(chunk)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            pretokens = executor.map(find_pretokens_with_special, text_chunks)
        
        pretokens = sum(pretokens, Counter())
    
    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode("utf-8")])
    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq
    
    return pretoken_freq


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix

import functools
def log_parameters(func):
    """
    A decorator that logs the name and parameters of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        if args:
            print(f"  Positional arguments: {args}")
        if kwargs:
            print(f"  Keyword arguments:")
            for key, value in kwargs.items():
                print(f"    {key} = {value}")
        
        result = func(*args, **kwargs)
        return result
    return wrapper

class ReverseCmp:
    def __init__(self, s):
        self.s = s

    # This method is what heapq uses for comparison
    def __lt__(self, other):
        # We want a max heap, so we invert the less-than comparison
        return self.s > other.s

    def __repr__(self):
        return f"'{self.s}'"

@log_parameters
def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str],
              progress_bar: bool = False, num_workers: int = 1, 
              debug: bool = False, save_vocab: bool = False):
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)
    logging.debug(f"\n>> pretoken_freq {len(pretoken_freq)}: {pretoken_freq}\n")

    logging.info("Initializing byte pair frequency table")
    pair_freq = Counter()
    # pair_freq = defaultdict(int)
    # for pretoken_tuple, freq in tqdm(pretoken_freq.items(), disable=not progress_bar):
    for pretoken_tuple, freq in pretoken_freq.items():  
        pl = len(pretoken_tuple) -1  
        for i in range(pl):
            pair = pretoken_tuple[i:i+2]
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq
    logging.debug(f"\n>> pair_freq initialized with {len(pair_freq)}: {pair_freq}\n")

    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    logging.debug(f"\n>> vocab initialized with {len(vocab)}: {vocab}\n")

    logging.info("Performing BPE algorithm")
    pre_merge_vocab_size = len(vocab)
    pbar = tqdm(total=vocab_size-pre_merge_vocab_size) if progress_bar else None
    merges = []
    maxh = []
    for pair, freq in pair_freq.items():
        heappush(maxh,(-freq, ReverseCmp(pair), pair))

    while len(vocab) < vocab_size:
        # Find the most frequent pair
        # TODO: use sorted to find the most frequent pair
        # most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        nf, _, most_freq_pair = heappop(maxh)
        while -nf != pair_freq[most_freq_pair]:            
            if -nf > pair_freq[most_freq_pair]: # frequency reduce not applied                
                heappush(maxh, (-pair_freq[most_freq_pair], ReverseCmp(most_freq_pair), most_freq_pair))
            nf, _, most_freq_pair = heappop(maxh)
        # Add the pair to the merges list
        merges.append(most_freq_pair)
        
        # Update the vocab
        # new_id = max(vocab.keys()) + 1
        new_id = len(vocab)
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            i=0
            while i < len(pretoken_tuple):
                pair = pretoken_tuple[i:i+2]
                if pair == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(pretoken_tuple, i)

                    # Update the pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                        heappush(maxh, (-pair_freq[add_pair], ReverseCmp(add_pair), add_pair))
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                        heappush(maxh, (-pair_freq[add_pair], ReverseCmp(add_pair), add_pair))
                    pair_freq[most_freq_pair] -= freq
                    heappush(maxh, (-pair_freq[most_freq_pair], ReverseCmp(most_freq_pair), most_freq_pair))
                i+=1
            # Update the pre-token frequency table
            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
        pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n) if progress_bar else None
    pbar.close() if progress_bar else None

    if debug:
        print(f"\n>> Final vocab with {len(vocab)}: {vocab}\n")
        print(f"\n>> Final pretoken_freq with {len(pretoken_freq)}: {pretoken_freq}\n")
        print(f"\n>> Final pair_freq with {len(pair_freq)}: {pair_freq}\n")
        print(f"\n>> Final merges with {len(merges)}: {merges}\n")

    return vocab, merges
