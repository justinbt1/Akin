import heapq
import numpy as np
import mmh3


class MinHash:
    """ MinHash base class.

    Attributes:
        n_gram (int): Number of characters used in each shingle.
        n_gram_type (str): Type of n gram used for shingles.
        permutations (int): Number of random permutations used to generate signatures.
        hash_bits (int): Hash value size used to generate signatures.
        seed (int): Seed used to generate signatures.

    """

    def __init__(self, n_gram=9, n_gram_type='char', permutations=100, hash_bits=64, seed=None):
        """ Generates a minhash signature matrix for texts in a corpus.

        Args:
            n_gram (int): Number of characters to be used in each shingle.
            n_gram_type (str): Type of n gram to use for shingles, must be char or term.
            permutations (int): Number of hash values in each document signature.
            hash_bits (int): Hash value size, must be 32, 64 or 128 bit.
            seed (int): Seeds from which to generate random hash function.

        """
        self.n_gram = n_gram

        if n_gram_type not in ['char', 'term']:
            raise ValueError(
                'Only "char" and "term" n_gram types are supported.'
            )
        self.n_gram_type = n_gram_type

        self.permutations = permutations

        if hash_bits not in [32, 64, 128]:
            raise ValueError(
                'Only 32, 64 and 128 bit hashes are supported.'
            )
        self.hash_bits = hash_bits

        if seed:
            self.seed = seed
            np.random.seed(seed)

    def _k_shingles(self, texts):
        """ Generates shingles for each input text.

        Break strings into k overlapping shingles consisting of characters or terms
        of n_gram size.

        Args:
            texts (list): list, array or Pandas series of input texts.

        Yields:
            list: Shingle list generated for each input text.

        """
        trim_overflow = (self.n_gram - 1) * -1

        if type(texts) == str:
            texts = [texts]

        for text in texts:
            if self.n_gram_type == 'char':
                shingles = [
                    text[char:char + self.n_gram] for char in range(len(text))
                ][:trim_overflow]
            else:
                terms = text.split()
                shingles = [
                    ' '.join(terms[term:term + self.n_gram]) for term in range(len(terms))
                ][:trim_overflow]

            if not shingles:
                raise ValueError(
                    'Shingle "n_gram" size must not exceed minimum text length.'
                )

            yield shingles

    def _hashing(self, shingle, seed):
        """ Performs hashing using seed generated hash.

        Args:
            shingle(str): Text shingle to hash.
            seed(int): Seed for generating hash function.

        Returns:
            int: Hashed shingle.

        """
        if self.hash_bits == 64:
            hashed_shingle = mmh3.hash64(shingle, seed)[0]
        elif self.hash_bits == 32:
            hashed_shingle = mmh3.hash(shingle, seed)
        else:
            hashed_shingle = mmh3.hash128(shingle, seed)

        return hashed_shingle


class MultiHash(MinHash):
    """ Generates minhash signatures by concatenating the min value of j permutation hashes.

    Attributes:
        hash_seeds (np.array): randomly generated seeds for each of j permutation hashes.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_seeds = np.random.randint(low=1, high=100000000, size=self.permutations)

    def _multi_hash(self, shingles):
        """ Generates a texts minhash signature using multi-hash method.

        Uses i random hashes for j permutations selecting the minimum hash value
        each time to build each texts hash signature.

        Slower but more stable than bottom-k hash method.

        Returns:
            list: List of text signatures generated using multi-hash method.

        """
        signatures = []
        for document in shingles:
            signature = []
            for seed in np.nditer(self.hash_seeds):
                min_value = None

                for shingle in document:
                    hash_value = self._hashing(shingle, int(seed))

                    if not min_value:
                        min_value = hash_value
                    elif min_value > hash_value:
                        min_value = hash_value

                signature.append(min_value)

            signatures.append(tuple(signature))

        return signatures

    def transform(self, text_corpus):
        """ Transform text to Minhash arrays using multi-hash method.

        Args:
            text_corpus(list): 2D Iterable containing text content of each document.

        Returns:
            list: List of minhash tuple signatures.

        """
        shingles = self._k_shingles(text_corpus)

        return self._multi_hash(shingles)


class BottomK(MinHash):
    """ Generates minhash signatures using k-smallest values of a single permutation hash.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _k_smallest_hash(self, shingles):
        """ Generates a texts minhash signature using k smallest neighbours method.

        Use a single random hash to simulate a shuffle of each text's shingles.
        Then selecting i-smallest minimum hash values for j permutations.

        Faster but less stable than multi hash method.

        Returns:
            list: List of text signatures generated using k smallest neighbours method.

        """
        signatures = []
        for document in shingles:
            signature = []
            # Uses a heap queue to make finding n smallest values more efficient.
            heapq.heapify(signature)

            if len(document) <= self.permutations:
                raise ValueError(
                    'N permutations must not be >= n shingles for k_smallest_values method'
                )

            for shingle in document:
                hashed_shingle = self._hashing(shingle, self.seed)
                heapq.heappush(signature, hashed_shingle)

            k_smallest_hashes = heapq.nsmallest(self.permutations, signature)
            signatures.append(tuple(k_smallest_hashes))

        return signatures

    def transform(self, text_corpus):
        """ Transform text to Minhash arrays using k-smallest hash method.

        Args:
            text_corpus(list): 2D Iterable containing text content of each document.

        Returns:
            list: List of minhash tuple signatures.

        """
        shingles = self._k_shingles(text_corpus)

        return self._k_smallest_hash(shingles)
