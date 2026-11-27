import math
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.types import BM25ParsedDocs, TokenRegistry
from heapq import nlargest

K1 = 1.5
B = 0.75

def bm25_build(docs: list[str]) -> BM25ParsedDocs:
    registry = TokenRegistry()
    total_tokens = 0
    lens: list[int] = []
    postings: dict[int, list[tuple[int, int]]] = {}

    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)

        tokens_len = len(tokens)
        lens.append(tokens_len)
        total_tokens += tokens_len
        sparse_doc: dict[int, int] = {}

        for token in tokens:
            id, _ = registry.set(token)
            sparse_doc[id] = sparse_doc.get(id, 0) + 1

        for id, count in sparse_doc.items():
            if id in postings:
                posting = postings[id]
            else:
                posting = []
                postings[id] = posting
            posting.append((doc_id, count))

    rows = len(docs)
    avgdl = total_tokens / rows if rows != 0 else 0

    return BM25ParsedDocs(
        token_registry=registry,
        avgdl=avgdl,
        lens=lens,
        length_factors=[
            K1 * (1 - B + B * length / avgdl) if avgdl != 0 else 0
            for length in lens
        ],
        postings=postings,
    )

def bm25_search(index: BM25ParsedDocs, query: str, k: int):
    if index.avgdl == 0:
        return []

    N = len(index.lens)
    query_token_ids: list[int] = []
    for s in set(tokenize(query)):
        token = index.token_registry.get(s)
        if token is None:
            continue
        query_token_ids.append(token)

    scores: dict[int, float] = {}

    for token in query_token_ids:
        posting = index.postings[token]
        doc_freq = len(posting)
        idf = math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
        term_weight = idf * (K1 + 1)
        for doc_id, term_freq in posting:
            scores[doc_id] = scores.get(doc_id, 0) + term_weight * term_freq / (term_freq + index.length_factors[doc_id])

    top = nlargest(k, scores.items(), key=lambda pair: (pair[1], -pair[0]))

    return top
