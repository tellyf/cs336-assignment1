import json
import cProfile
import timeit
from src.train_bpe import train_bpe
import pathlib
import pstats
import tracemalloc
import pickle
import os

TOP_PATH = (pathlib.Path(__file__).resolve().parent) 

def run():
    # input_path = TOP_PATH / "tests/fixtures/tinystories_sample_5M.txt"
    # input_path = TOP_PATH / "data/TinyStoriesV2-GPT4-valid.txt"
    input_path = TOP_PATH / "data/TinyStoriesV2-GPT4-train.txt"
    vocab, _ = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers = os.cpu_count(),
        progress_bar=True,
    )
    return vocab


if __name__ == "__main__":
    save_vocab = False  # Set to True to save the vocab  
    mem_profile = False
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = timeit.default_timer()
    if mem_profile:
        tracemalloc.start()

    vocab = run()

    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    profiler.disable()
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(10)
    
    if mem_profile:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')    
        print("[ Top memory 10 ]")
        for stat in top_stats[:10]:
            print(stat)

    if save_vocab:
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)            

