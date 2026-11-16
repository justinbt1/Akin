import mmh3

from akin import DictionaryArray, optimise


class LSH:
    """ Index labelled MinHash signatures using locality sensitive hashing.

    Attributes:
        permutations (int): Number of values in each MinHash signature.
        n_bands (int): Number of bands in each signature.
        n_rows (int): Number of signature values in each band.
        seed (int): Seed used to hash bands into buckets.
        keys (dict): Mapping from document labels to stored signatures.

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
        """ Initialize an index with explicit or automatically selected bands.

        Provide both n_bands and n_rows, with their product equal to permutations, or
        provide jaccard_similarity for automatic selection. The probability model used
        for automatic selection assumes independent MinHash positions and is
        approximate for densified OPH signatures.

        Args:
            permutations (int): Number of values in each MinHash signature.
            jaccard_similarity (float, optional): Target Jaccard similarity used for
                automatic band selection, not a query filter.
            probability (float, optional): Target probability of retrieving a candidate
                at jaccard_similarity. Defaults to 0.95.
            n_bands (int, optional): Explicit number of bands.
            n_rows (int, optional): Explicit number of values per band.
            seed (int, optional): Seed for band hashing. Defaults to 1.

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
            signature (tuple): MinHash signature to split into bands.

        Returns:
            list: Bucket identifiers in band order.

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
        """ Estimate Jaccard similarity from matching signature positions.

        Args:
            query_signature (tuple): Query MinHash signature.
            candidates (set): Candidate document labels.

        Returns:
            list: Unsorted (estimated_similarity, label) tuples.

        """
        matches = []
        for candidate in candidates:
            estimated_jaccard_ratio = sum(
                a == b for a, b in zip(query_signature, self.keys[candidate])
            ) / len(query_signature)

            matches.append((estimated_jaccard_ratio, candidate))

        return matches

    def update(self, minhash_signatures, labels):
        """ Add document signatures and their labels to the index.

        All signatures must contain permutations values and use the same MinHash
        configuration, hash seeds and container representation.

        Args:
            minhash_signatures (list): List of MinHash signatures.
            labels (list): Corresponding unique, hashable document labels.

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
            labels (iterable): Labels of indexed documents to remove.

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
        """ Returns near duplicates from model for estimated similarity threshol

        Args:
            label (hashable): Label of the indexed document to query.
            include_similarity (bool, optional): Include positional MinHash
                similarity estimates. Defaults to False.

        Returns:
            list: Candidate labels, or (estimated_similarity, label) tuples
                when include_similarity is True.

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
