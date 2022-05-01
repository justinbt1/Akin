import numpy as np
import mmh3
import heapq


class MinHash:
    """ MinHash.

    Attributes:
        n_gram (int): Number of characters used in each shingle.
        n_gram_type (str): Type of n gram used for shingles.
        permutations (int): Number of random permutations used to generate signatures.
        hash_bits (int): Hash value size used to generate signatures.
        method (str): Method used to generate signatures.
        seed (int): Seed used to generate signatures.
        signatures (np.array): Matrix of minhash signatures, m represents each texts
            minhash signature with n representing each permutation's minimum hash value.

    """

    def __init__(
            self,
            text,
            method='multi_hash',
            n_gram=9,
            n_gram_type='char',
            permutations=100,
            hash_bits=64,
            seed=None
    ):
        """ Generates a minhash signature matrix for texts in a corpus.

        Args:
            text (list, np.array): Iterable containing text content of each document.
            method (str): Method to be used for minhash function, must be multi_hash
                or bottom_k.
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

        self.seed = None
        if seed:
            self.seed = seed
            np.random.seed(seed)

        if method not in [
            'multi_hash',
            'k_smallest_values'
        ]:
            raise ValueError(
                'Only "multi_hash" and "bottom_k" hash methods are supported.'
            )
        self.method = method

        self._hash_seeds = self._generate_random_seeds()

        # Run methods.
        self._shingles = self._k_shingles(text)
        self.signatures = self._min_hash()

    def _generate_random_seeds(self):
        """ Generates a list of pseudo-random integers to use as hash seeds.

        Returns:
            np.array: Hash seed integers.

        """
        if self.method == 'multi_hash':
            hash_seeds = np.random.randint(low=1, high=100000000, size=self.permutations)
        else:
            hash_seeds = np.random.randint(low=1, high=100000000)

        return hash_seeds

    def _k_shingles(self, texts):
        """ Generates shingles for each input text.

        Break strings into k overlapping shingles consisting of characters or terms
        of n_gram size.

        Args:
            texts (list, np.array): list, array or Pandas series of input texts.

        Yields:
            List: Shingle list generated for each input text.

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

    def _multi_hash(self, document):
        """ Generates a texts minhash signature using multi-hash method.

        Uses i random hashes for j permutations selecting the minimum hash value
        each time to build each texts hash signature.

        Slower but more stable than bottom-k hash method.

        Args:
            document (list): List of document shingles.

        Returns:
            list: List of text signatures generated using bottom-k neighbours method.

        """
        signature = []
        for seed in np.nditer(self._hash_seeds):
            self._min_value = None

            for shingle in document:
                if self.hash_bits == 64:
                    # mmh3 returns two 64bit hash values, only the first is used.
                    hash_value = mmh3.hash64(shingle, int(seed))[0]
                elif self.hash_bits == 32:
                    hash_value = mmh3.hash(shingle, int(seed))
                else:
                    hash_value = mmh3.hash128(shingle, int(seed))

                if not self._min_value:
                    self._min_value = hash_value
                elif self._min_value > hash_value:
                    self._min_value = hash_value

            signature.append(self._min_value)

        return signature

    def _k_smallest_hash(self, document):
        """ Generates a texts minhash signature using k smallest neighbours method.

        Uses a single random hash to simulate a shuffle of each text's shingles.
        Then selecting i-smallest minimum hash values for j permutations.

        Faster but less stable than multi hash method.

        Args:
            document (list): List of text shingles.

        Returns:
            list: List of text signatures generated using k smallest neighbours method.

        """
        signature = []
        # Uses a heap to make calculating n smallest values more efficient.
        heapq.heapify(signature)

        if len(document) <= self.permutations:
            raise ValueError(
                'N permutations must not be >= n shingles for k_smallest_values method'
            )

        for shingle in document:
            if self.hash_bits == 64:
                hashed_shingle = mmh3.hash64(shingle, self._hash_seeds)[0]
            elif self.hash_bits == 32:
                hashed_shingle = mmh3.hash(shingle, self._hash_seeds)
            else:
                hashed_shingle = mmh3.hash128(shingle, self._hash_seeds)

            heapq.heappush(signature, hashed_shingle)

        return heapq.nsmallest(self.permutations, signature)

    def _min_hash(self):
        """ Calculates document signature by calling the selected hashing method.

        Returns:
             np.array: Matrix of minhash signatures, m represents each texts minhash
                signature with n representing each permutation's minimum hash value.

        """
        signatures = []
        for document in self._shingles:
            if self.method == 'multi_hash':
                signature = self._multi_hash(document)
                signatures.append(signature)
            elif self.method == 'k_smallest_values':
                signature = self._k_smallest_hash(document)
                signatures.append(signature)

        return np.array(signatures)
