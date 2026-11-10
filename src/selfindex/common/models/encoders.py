END_OF_TEXT_TOKEN = "<|endoftext|>"

ENCODERS = {
    "bge": dict(
        model="BAAI/bge-large-en-v1.5",
        dim=1024,
        pooling="cls",
        dtype="float16",
        maxlen=512,
        append_eod=False,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "qwen3": dict(
        model="Qwen/Qwen3-Embedding-8B",
        dim=4096,
        pooling="last",
        dtype="bfloat16",
        maxlen=512,
        append_eod=True,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: ",
    ),
}
