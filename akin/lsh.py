import mmh3

from akin import DictionaryArray, optimise


class LSH:
    """ Locality Sensitive Hashing.

    Attributes:
        permutations (int): Number of permutations used in MinHash.
        no_of_bands (int): Number of bands used in model.
        seed (int): Random seed used for hashing.

    """

    def __init__(
            self,
            permutations,
            jaccard_similarity=None,
            probability=0.95,
            n_bands=None,
            n_rows=None,
            seed=1
    ):
        """ Initialize the LSH object.

        Args:
            no_of_bands (int): Number of bands to break minhash signature into.
            permutations (int): Number of permutations (hashes) in minhash
                signature.
            seed (int): Random seed used for hashing.

        """
        if n_bands and n_rows:
            if n_bands * n_rows != permutations:
                raise ValueError('n_bands * n_rows must equal permutations')

            self.n_bands = n_bands
            self.n_rows = n_rows

        elif n_bands or n_rows:
            raise ValueError('Both n_bands and n_rows must be provided!')

        elif jaccard_similarity is not None:
            self.n_bands, self.n_rows, _ = optimise.hyperparameters(
                permutations,
                jaccard_similarity,
                probability
            )

        else:
            raise ValueError(
                'Target Jaccard similarity or n_bands and n_rows must be provided!'
            )

        self._buckets = DictionaryArray(self.n_bands)
        self.seed = seed
        self.permutations = permutations
        self.keys = {}

    def _lsh(self, signature):
        """ Break signatures into bands and hash components to buckets.

        Args:
            signature (tuple): a minhash signature.

        """
        band_hashes = []
        for i in range(0, self.permutations, self.n_rows):
            band = str(signature[i:i + self.n_rows])
            bucket_id = mmh3.hash64(band, self.seed)[0]
            band_hashes.append(bucket_id)

        return band_hashes

    def _estimate_jaccard(
            self,
            query_signature,
            candidates
    ):
        """ Identify candidate duplicates and check Jaccard Similarity.

        Args:
            query_signature (tuple): Query minhash signature.
            candidates (dict): List of bucket ids.

        Returns:
            list: Near duplicate document ids.

        """
        matches = []
        for candidate in candidates:
            estimated_jaccard_ratio = sum(
                a == b for a, b in zip(query_signature, self.keys[candidate])
            ) / len(query_signature)

            matches.append((estimated_jaccard_ratio, candidate))

        return matches

    def update(self, minhash_signatures, labels):
        """ Updates LSH object with new MinHash matrix and labels.

        Args:
            minhash_signatures (list): new minhash signatures to add to LSH
                object.

        """
        if len(minhash_signatures[0]) != self.permutations:
            raise IndexError(
                'Number of permutations in minhash must be '
                f'{self.permutations} to match LSH model initialisation params.'
            )

        if len(minhash_signatures) != len(labels):
            raise IndexError('Length of minhash_signatures and labels must match.')

        for signature, label in zip(minhash_signatures, labels):
            if self.keys.get(label):
                raise KeyError(
                    f'Label must be unique, however "{label}" already exists '
                    'in model.'
                )

            self.keys[label] = signature
            for band_id, bucket_id in enumerate(self._lsh(signature)):
                self._buckets.update(band_id, key=bucket_id, value=label)

    def remove(self, labels):
        """ Remove label and associated text signature from model.

        Args:
            minhash_signatures (list): minhash signatures to remove from model.

        """
        for label in labels:
            signature = self.keys[label]
            for band_id, bucket_id in enumerate(self._lsh(signature)):
                self._buckets.remove_value(
                    band_id,
                    key=bucket_id,
                    value=label
                )

            del self.keys[label]

    def query(
            self,
            label,
            include_similarity=False
    ):
        """ Returns near duplicates from model.

        Takes a provided text label and returns a list of labels for texts
        whose similarity with the provided text is above a provided threshold.

        Can be used to create a recommendation model.

        Args:
            minhash_signature (tuple): minhash signature of text for which to
                return near duplicates.
            include_similarity (bool): return similarity alongside estimated
                near duplicates.

        Returns:
            list: Candidate duplicates for provided text label.

        """
        candidates = set()
        minhash_signature = self.keys[label]
        for band_id, bucket_id in enumerate(self._lsh(minhash_signature)):
            candidates.update(self._buckets.get(band_id, bucket_id))

        candidates.discard(label)

        if include_similarity:
            near_duplicates = self._estimate_jaccard(minhash_signature, candidates)
        else:
            near_duplicates = list(candidates)

        return near_duplicates
