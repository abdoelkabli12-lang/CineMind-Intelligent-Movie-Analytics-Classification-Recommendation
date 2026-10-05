X['genres'] = X['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
X['keywords'] = X['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))   