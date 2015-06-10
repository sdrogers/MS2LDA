def word_indices(document):
    """
    Turns a document vector of word counts into a vector of the indices
     words that have non-zero counts, repeated for each count
    e.g. 
    >>> word_indices(np.array([3, 0, 1, 2, 0, 5]))
    [0, 0, 0, 2, 3, 3, 5, 5, 5, 5, 5]
    """
    results = []
    for nnz in document.values.nonzero()[1]:
        for n in range(int(document[nnz])):
            results.append(nnz)
    return results