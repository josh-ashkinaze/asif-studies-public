"""
This implements log-odds with Dirchelet priors, plus an optional z-score. Positive values favor the RIGHT class (class_b).

Based on this implementation: https://people.ischool.berkeley.edu/~dbamman/info256_slides/3_distinctive_words.pdf

It assumes you have texts and binary class labels as lists, numpy arrays, or pandas Series.

Examples:
    Basic usage with default settings:
        >>> import pandas as pd
        >>> texts = ["Cats are playful and cute", "Kittens love naps and strings",
        ...          "Dogs are loyal and friendly", "Puppies enjoy walks and parks"]
        >>> labels = [0, 0, 1, 1]
        >>> analyzer = LogOddsAnalyzer(pseudocount=0.01, to_normalize=False)
        >>> res = analyzer.fit(texts, labels)
        >>> print(res.head())

    Normalized (z-score) log-odds:
        >>> analyzer = LogOddsAnalyzer(pseudocount=0.01, to_normalize=True)
        >>> res = analyzer.fit(texts, labels)
        >>> print(res.head())  # ranked by z-score now

    From a DataFrame (pass columns directly):
        >>> df = pd.DataFrame({"text": texts, "label": labels})
        >>> res = analyzer.fit(df["text"], df["label"])

    From numpy arrays:
        >>> import numpy as np
        >>> res = analyzer.fit(np.array(texts), np.array(labels))

    Explicit class ordering (control which class is "left" vs "right"):
        >>> res = analyzer.fit(texts, labels, class_order=[0, 1])  # positive => class 1 terms

    Bigrams and unigrams together:
        >>> res = analyzer.fit(texts, labels, ngram_range=(1, 2))
        >>> print(res[res["term"].str.contains(" ")].head())  # bigrams only

    Disable stopword removal:
        >>> res = analyzer.fit(texts, labels, stopwords=None)

    Add custom stopwords on top of English defaults:
        >>> res = analyzer.fit(texts, labels, stopwords="english", add_stopwords=["cats", "dogs"])

    Get top/bottom N distinctive terms:
        >>> extremes = analyzer.get_extreme(n=5)
        >>> print(extremes)

    Plot distinctive terms:
        >>> from matplotlib import pyplot as plt
        >>> fig = analyzer.plot(n=5, figsize=(8, 6), write_arrows=True)
        >>> plt.title("Cats vs Dogs vocabulary")
        >>> plt.show()

    String class labels:
        >>> str_labels = ["cats", "cats", "dogs", "dogs"]
        >>> res = analyzer.fit(texts, str_labels, class_order=["cats", "dogs"])

    Larger pseudocount = stronger prior, smooths rare terms more aggressively:
        >>> conservative = LogOddsAnalyzer(pseudocount=1.0, to_normalize=True)
        >>> aggressive   = LogOddsAnalyzer(pseudocount=0.001, to_normalize=True)
        >>> res_c = conservative.fit(texts, labels)
        >>> res_a = aggressive.fit(texts, labels)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


ENGLISH_STOPWORDS = frozenset({
    'a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already',
    'also','although','always','am','among','amongst','amount','an','and','another','any','anyhow','anyone',
    'anything','anyway','anywhere','are','around','as','at','back','be','became','because','become','becomes',
    'becoming','been','before','beforehand','behind','being','below','beside','besides','between','beyond',
    'both','but','by','can','cannot','could','de','describe','detail','did','do','does','doing','done','down',
    'during','each','either','else','elsewhere','empty','enough','etc','even','ever','every','everyone',
    'everything','everywhere','except','few','first','five','for','former','formerly','found','four','from',
    'front','full','further','get','give','go','had','has','have','having','he','hence','her','here',
    'hereafter','hereby','herein','hereupon','hers','herself','him','himself','his','how','however','hundred',
    'i','ie','if','in','indeed','into','is','it','its','itself','just','keep','last','latter','latterly',
    'least','less','ltd','made','many','may','me','meanwhile','might','mill','mine','more','moreover','most',
    'mostly','move','much','must','my','myself','name','namely','neither','never','nevertheless','next','nine',
    'no','nobody','none','nor','not','nothing','now','nowhere','of','off','often','on','once','one','only',
    'onto','or','other','others','otherwise','our','ours','ourselves','out','over','own','part','perhaps',
    'please','put','rather','re','same','see','seem','seemed','seeming','seems','serious','several','she',
    'should','show','side','since','sincere','six','sixty','so','some','somehow','someone','something',
    'sometime','sometimes','somewhere','still','such','system','take','ten','than','that','the','their',
    'theirs','them','themselves','then','thence','there','thereafter','thereby','therefore','therein',
    'thereupon','these','they','third','this','those','though','three','through','throughout','thru','thus',
    'to','together','too','top','toward','towards','twelve','twenty','two','un','under','until','up','upon',
    'us','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter',
    'whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever',
    'whole','whom','whose','why','will','with','within','without','you','your','yours','yourself','yourselves'
})

_ALPHA_RE = re.compile(r'[a-z0-9]+')


class LogOddsAnalyzer:
    """Log-odds with Dirichlet priors.

        δ = log( (c_b + α) / (n_b + α0 − α) ) − log( (c_a + α) / (n_a + α0 − α) )
        z ≈ δ / sqrt( 1/(c_a + α) + 1/(c_b + α) )

    Sign: Positive → RIGHT class (class_b); negative → LEFT (class_a).
    """

    def __init__(self, pseudocount=0.01, to_normalize=False):
        self.alpha = float(pseudocount)
        self.target = "log_odds_normalized" if to_normalize else "log_odds"
        self.results = None
        self.class_names = None

    ####################
    # Input Helpers
    ####################

    def _to_series(self, x, name):
        """Coerce list, array, or Series to a pandas Series."""
        if isinstance(x, pd.Series):
            return x.reset_index(drop=True)
        return pd.Series(x, name=name)

    def _build_stopwords(self, stopwords, add_stopwords):
        if stopwords == "english":
            base = ENGLISH_STOPWORDS
        elif stopwords is None:
            base = frozenset()
        else:
            base = frozenset(stopwords)
        return base | frozenset(add_stopwords) if add_stopwords else base

    ####################
    # Tokenization
    ####################

    def _tokenize_corpus(self, texts, ngram_range, stop):
        """Tokenize an entire corpus at once, returning a Counter."""
        full = " ".join(t for t in texts if isinstance(t, str) and t)
        tokens = _ALPHA_RE.findall(full.lower())

        if stop:
            tokens = [t for t in tokens if t not in stop]

        lo, hi = ngram_range
        if hi == 1:
            return Counter(tokens)

        counts = Counter(tokens) if lo == 1 else Counter()
        for n in range(max(2, lo), hi + 1):
            counts.update(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        return counts

    ####################
    # Fitting
    ####################

    def fit(self, texts, labels, ngram_range=(1, 1),
            stopwords='english', add_stopwords=None, class_order=None):
        """Compute log-odds (and z) for each term.

        Args:
            texts: Text data as a list, numpy array, or pandas Series.
            labels: Binary class labels as a list, numpy array, or pandas Series.
            ngram_range: (min_n, max_n) for n-grams.
            stopwords: 'english', a collection, or None.
            add_stopwords: Optional extra stopwords.
            class_order: Optional 2-sequence specifying (class_a, class_b).

        Returns:
            DataFrame with columns:
            ['term', 'count_a', 'count_b', 'log_odds', 'log_odds_normalized'].
        """
        texts = self._to_series(texts, "text")
        labels = self._to_series(labels, "label")

        if class_order is not None:
            classes = list(class_order)
            if len(classes) != 2:
                raise ValueError(f"class_order must have exactly 2 classes, got {len(classes)}")
        else:
            classes = sorted(labels.dropna().unique(), reverse=True)
            if len(classes) != 2:
                raise ValueError(f"Need exactly 2 classes, got {len(classes)}")

        class_a, class_b = classes
        self.class_names = (class_a, class_b)

        stop = self._build_stopwords(stopwords, add_stopwords)
        mask = labels == class_a

        counts_a = self._tokenize_corpus(texts[mask].tolist(), ngram_range, stop)
        counts_b = self._tokenize_corpus(texts[~mask].tolist(), ngram_range, stop)

        all_terms = list(set(counts_a) | set(counts_b))
        if not all_terms:
            self.results = pd.DataFrame(columns=["term","count_a","count_b","log_odds","log_odds_normalized"])
            return self.results

        ####################
        # Vectorized Scoring
        ####################

        V = len(all_terms)
        alpha, alpha0 = self.alpha, self.alpha * V
        n_a = sum(counts_a.values())
        n_b = sum(counts_b.values())
        denom_a = n_a + alpha0 - alpha
        denom_b = n_b + alpha0 - alpha

        ca = np.array([counts_a.get(t, 0) for t in all_terms], dtype=np.float64)
        cb = np.array([counts_b.get(t, 0) for t in all_terms], dtype=np.float64)

        log_odds = np.log((cb + alpha) / denom_b) - np.log((ca + alpha) / denom_a)
        z_scores = log_odds / np.sqrt(1.0 / (ca + alpha) + 1.0 / (cb + alpha))

        self.results = (
            pd.DataFrame({
                "term": all_terms,
                "count_a": ca.astype(int),
                "count_b": cb.astype(int),
                "log_odds": log_odds,
                "log_odds_normalized": z_scores,
            })
            .sort_values(self.target, ascending=False)
            .reset_index(drop=True)
        )
        return self.results

    ####################
    # Output
    ####################

    def get_extreme(self, n):
        """Top n and bottom n terms by the active metric."""
        if self.results is None:
            raise ValueError("Must fit first")
        s = self.results.sort_values(self.target, ascending=True)
        return pd.concat([s.head(n), s.tail(n)]).reset_index(drop=True)

    def plot(self, n=10, figsize=(10, 8), write_arrows=False):
        """Plot top and bottom n terms by the active metric."""
        if self.results is None:
            raise ValueError("Must fit first")

        data = self.get_extreme(n)
        vals = data[self.target].values
        colors = ['#F45B69' if v < 0 else '#826AED' for v in vals]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(data)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['term'])
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(self.target.replace('_', ' ').title())

        if self.class_names:
            a, b = self.class_names
            ax.set_title(f'{a} (Left) vs {b} (Right)')
            if write_arrows:
                ax.text(0.01, 0.98, f'← {a}', transform=ax.transAxes, va='top')
                ax.text(0.99, 0.98, f'{b} →', transform=ax.transAxes, ha='right', va='top')

        plt.tight_layout()
        return fig