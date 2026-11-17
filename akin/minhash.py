import numpy as np
import mmh3


class _MinHash:
    """ MinHash base class.

    Attributes:
        n_gram (int): Number of characters or terms in each shingle.
        n_gram_type (str): Whether shingles contain characters or terms.
        permutations (int): Number of minhash permutations in each signature.
        hash_bits (int): Hash bit size used for signatures.
        seed (int): Seed used to hash signatures.

    """
    def __init__(
        self,
        n_gram=9,
        n_gram_type='char',
        permutations=100,
        hash_bits=64,
        seed=1
    ):
        """Configure shingling and hashing for MinHash implementation.

        Args:
            n_gram (int): Number of characters or terms per shingle. Must be > 2.
            n_gram_type (str): Shingle text by 'char' or 'term'.
            permutations (int): Number of hash values in each signature.
            hash_bits (int): Hash size; must be 32, 64, or 128.
            seed (int): 32bit seed for reproducible hash generation.

        """
        if not isinstance(n_gram, int):
            raise TypeError('N gram must be an integer.')
        elif n_gram < 2:
            raise ValueError('N gram must be greater than 1.')

        self.n_gram = n_gram

        if n_gram_type not in ['char', 'term']:
            raise ValueError(
                'Only "char" and "term" n_gram types are supported.'
            )

        self.n_gram_type = n_gram_type

        if not isinstance(permutations, int):
            raise TypeError('Permutations must be an integer.')
        elif permutations < 1:
            raise ValueError('Permutations must be at least 1.')

        self.permutations = permutations

        if not isinstance(hash_bits, int):
            raise TypeError('Hash bits must be an integer.')
        elif hash_bits not in [32, 64, 128]:
            raise ValueError(
                'Only 32, 64 and 128 bit hashes are supported.'
            )

        self.hash_bits = hash_bits

        if not isinstance(seed, int):
            raise TypeError('Seed must be an integer.')
        elif not 0 <= seed < 2**32:
            raise ValueError('Seed must be between 0 and 2**32 - 1.')

        self.seed = seed
        self.hash_range = np.random.default_rng(seed)

    def _k_shingles(self, texts):
        """Yield a set of shingles for each input text.

        Each shingle contains n_gram overlapping characters or whitespace-separated
        terms.

        Args:
            texts (str | iterable[str]): One text, or an iterable of texts.

        Yields:
            set[str]: Unique shingles from each input text.

        """
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            if self.n_gram_type == 'char':
                shingles = {
                    text[char:char + self.n_gram] for char in range(
                        len(text) - self.n_gram + 1
                    )
                }
            else:
                terms = text.split()
                shingles = {
                    ' '.join(terms[term:term + self.n_gram])
                    for term in range(len(terms) - self.n_gram + 1)
                }

            if not shingles:
                raise ValueError(
                    'Shingle "n_gram" size must not exceed minimum text '
                    'length.'
                )

            yield shingles

    def _hashing(self, shingle, seed):
        """ Performs hashing using seed generated hash.

        Args:
            shingle (str): Text shingle to hash.
            seed (int): Seed for the hash function.

        Returns:
            int: Hash value of the shingle.

        """
        if self.hash_bits == 64:
            hashed_shingle = mmh3.hash64(shingle, seed, signed=False)[0]
        elif self.hash_bits == 32:
            hashed_shingle = mmh3.hash(shingle, seed, signed=False)
        else:
            hashed_shingle = mmh3.hash128(shingle, seed, signed=False)

        return hashed_shingle


class KMinHash(_MinHash):
    """Generate signatures using independent seeded hash functions.

    For each permutation, the minimum hash value across a document's unique
    shingles is selected.

    Attributes:
        hash_seeds (numpy.ndarray): Randomly generated distinct seeds, one for each
            permutation.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hash_seeds = self.hash_range.choice(
            2**32,
            size=self.permutations,
            replace=False
        )

    def _multi_hash(self, shingles):
        """Generate one signature per document using multiple independent hashes.

        Each seed produces a hash for every shingle, and the minimum hash for that seed
        becomes the corresponding signature component.

        Slower but more stable than the single hash densified OPH method of UniMinHash.

        Args:
            shingles (iterable[set[str]]): Unique shingles for each document.

        Returns:
            list[tuple[int, ...]]: One signature of length permutations per document.

        """
        signatures = []
        for document in shingles:
            signature = []
            for seed in np.nditer(self.hash_seeds):
                min_value = None

                for shingle in document:
                    hash_value = self._hashing(shingle, int(seed))

                    if min_value is None:
                        min_value = hash_value
                    elif min_value > hash_value:
                        min_value = hash_value

                signature.append(min_value)

            signatures.append(tuple(signature))

        return signatures

    def transform(self, text_corpus):
        """Convert text into signatures using multiple independent seeded hashes.

        Args:
            text_corpus (iterable[str]): Iterable containing one text string per
            document.

        Returns:
            list[tuple[int, ...]]: One signature of length "permutations" per document.

        """
        shingles = self._k_shingles(text_corpus)

        return self._multi_hash(shingles)


class OPMinHash(_MinHash):
    """Generate signatures using one-permutation hashing (OPH) with densification.

    Each shingle is hashed once and assigned to a bin (one bin per permutation).
    The minimum hash in each bin is used; empty bins are filled from a non-empty bin
    using a deterministic probe sequence.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_seed = (self.seed + 1) % (2**32)
        self._probe_cache = {}

    def _probe(self, bin_id, attempt):
        """ Return the bin to borrow from for an empty bin.

        Probe sequences are cached per bin and extended on demand.
        """
        sequence = self._probe_cache.setdefault(bin_id, [])
        while len(sequence) <= attempt:
            sequence.append(
                self._hashing(
                    f'{bin_id}:{len(sequence)}',
                    self._probe_seed
                ) % self.permutations
            )
        return sequence[attempt]

    def _one_permutation_hash(self, shingles):
        """Generate and densify one-permutation signatures.

        Args:
            shingles (iterable[set[str]]): Unique shingles for each document.

        Returns:
            list[tuple[int, ...]]: One fully populated signature of length
            "permutations" per document.

        """
        n_bins = self.permutations
        bits = self.hash_bits
        signatures = []

        for document in shingles:
            bins = [None] * n_bins
            for shingle in document:
                hash_value = self._hashing(shingle, seed=self.seed)
                bin_id = (hash_value * n_bins) >> bits
                current = bins[bin_id]
                if current is None or hash_value < current:
                    bins[bin_id] = hash_value

            filled = list(bins)
            for bin_id in range(n_bins):
                if bins[bin_id] is None:
                    attempt = 0
                    while bins[self._probe(bin_id, attempt)] is None:
                        attempt += 1
                    filled[bin_id] = bins[self._probe(bin_id, attempt)]

            signatures.append(tuple(filled))

        return signatures

    def transform(self, text_corpus):
        """ Transform text to Minhash arrays using one permutation hashing (OPH).

        Args:
            text_corpus (terable[str]): Iterable containing one text per document.

        Returns:
            list[tuple[int, ...]]: One signature of length "permutations" per document.

        """
        shingles = self._k_shingles(text_corpus)

        return self._one_permutation_hash(shingles)
