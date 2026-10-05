import statistics

import tiktoken
from transformers import AutoTokenizer

from src.ingestion import (
    DATA_PATH,
    build_children,
    build_parents,
    load_mineru,
)


data = load_mineru(DATA_PATH)
parents = build_parents(data)
children = build_children(parents)

texts = [child.content for child in children]


# OpenAI text-embedding-3-large tokenizer
openai_tokenizer = tiktoken.encoding_for_model(
    "text-embedding-3-large"
)

openai_counts = [
    len(openai_tokenizer.encode(text))
    for text in texts
]


# zembed-1 is based on Qwen3
# This downloads tokenizer files only, NOT the 4B model weights.
zembed_tokenizer = AutoTokenizer.from_pretrained(
    "zeroentropy/zembed-1-embedding"
)

zembed_counts = [
    len(
        zembed_tokenizer.encode(
            text,
            add_special_tokens=True,
        )
    )
    for text in texts
]


def show_stats(name: str, counts: list[int]) -> None:
    sorted_counts = sorted(counts)

    print(f"\n{name}")
    print("-" * 40)
    print("Children:", len(counts))
    print("Total tokens:", sum(counts))
    print("Average:", round(statistics.mean(counts), 1))
    print("Median:", statistics.median(counts))
    print(
        "P95:",
        sorted_counts[int(len(sorted_counts) * 0.95)],
    )
    print("Maximum:", max(counts))


show_stats("OpenAI text-embedding-3-large", openai_counts)
show_stats("ZeroEntropy zembed-1", zembed_counts)

