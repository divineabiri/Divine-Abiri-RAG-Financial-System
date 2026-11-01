# helper_utils.py

def word_wrap(text, width=72):
    """
    Wraps a long string so it prints nicely in the terminal instead of
    running off the edge of the screen. Purely cosmetic — it has nothing
    to do with retrieval or generation, just readability when you print
    chunks or answers to check your work.
    """
    if len(text) < width:
        return text
    # Find the last space before the width limit, so we don't cut a word in half
    split_point = text[:width].rsplit(' ', 1)[0]
    remainder = text[len(split_point) + 1:]
    return split_point + '\n' + word_wrap(remainder, width)

import numpy as np
from tqdm import tqdm

def project_embeddings(embeddings, umap_transform):
    """
    Projects a list of high-dimensional embeddings down into 2D space
    using an already-fitted UMAP transformer, so they can be plotted on
    a normal x/y chart. This is purely for visualization — it doesn't
    affect retrieval or generation at all, it just lets you *see* how
    close or far apart your chunks and queries are from each other.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings