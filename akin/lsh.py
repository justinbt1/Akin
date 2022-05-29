from collections import Counter
from copy import copy
import numpy as np
import mmh3

from akin import DictionaryArray


class LSH:
    """ Locality Sensitive Hashing.

    Attributes:
        no_of_bands (int): Number of bands used in model.
        permutations (int): Number of permutations used in MinHash.

    """

    def __init__(self, no_of_bands=None, seed=1):
        """ Initialize the LSH object.

        Args:
            no_of_bands (int): Number of bands to break minhash signature into.

        """
        self.no_of_bands = no_of_bands
        self.seed = seed
        self._buckets = DictionaryArray(no_of_bands)
        self.permutations = None

    def _lsh(self, signatures):
        """ Break signatures into bands and hash components to buckets.

        Args:
            signatures (np.array): MinHash signature Matrix.

        """
        for signature in signatures:
            hashable_signature = tuple(hash_value for hash_value in signature)

            bands = np.hsplit(
                np.array(signature),
                self.no_of_bands
            )

            for band_id, band in enumerate(bands):
                bucket_id = mmh3.hash64(tuple(band), self.seed)[0]
                yield band_id, bucket_id, hashable_signature

    def update(self, minhash_signatures):
        """ Updates LSH object with new MinHash matrix and labels.

        Args:
            minhash_signatures(np.array): MinHash object containing new minhash signatures to
                add to LSH object.

        """
        if not self.permutations:
            self.permutations = minhash_signatures.shape[1]
        else:
            if minhash_signatures.shape[1] != self.permutations:
                raise IndexError(
                    f'Number of permutations in minhash must be {self.permutations} to match LSH model.'
                )

        if not self.no_of_bands:
            self.no_of_bands = minhash_signatures[0].shape[1] // 2

        band_id, bucket_id, hashable_signature = self._lsh(minhash_signatures)
        self._buckets.update(band_id, key=bucket_id, value=hashable_signature)

    def remove(self, minhash_signatures):
        """ Remove label and associated text signature from model.

        Args:
            minhash_signatures (str, int, float): Label for text to be removed from model.

        """
        band_id, bucket_id, hashable_signature = self._lsh(minhash_signatures)
        self._buckets.remove_value(band_id, key=bucket_id, value=hashable_signature)

    def _candidate_duplicates(self, signature, candidates, sensitivity, jaccard_threshold):
        """ Identify candidate duplicates and check Jaccard Similarity.

        Args:
            candidates (list): List of bucket ids.
            sensitivity (int): Number of identical buckets two ids must occur
                in to be considered a near duplicate pair.
            jaccard (float): Minimum Jaccard Similarity for documents to be
                counted as near duplicates.

        Returns:
            List: Near duplicate document ids.

        """
        """
        # Apply sensitivity threshold.
        if sensitivity > 1:
            for key in list(candidates):
                if candidates[key] < sensitivity:
                    del candidates[key]
        """

        # Apply Jaccard threshold and unzip pairs.
        matches = []

        if jaccard_threshold:
            for candidate in candidates:
                intersection = len(set(signature) & set(candidate))
                union = len(set(signature) | set(candidate))
                jaccard_ratio = intersection / union
                if jaccard_ratio < jaccard_threshold:
                    matches.append(candidate)

        return matches

    def query(self, minhash_signature, min_jaccard=None, sensitivity=1):
        """ Returns near duplicates from model.

        Takes a provided text label and returns a list of labels for texts whose
        similarity with the provided text is above a provided threshold.

        Can be used to create a recommendation model.

        Args:
            minhash_signature (str, int, float): Label of text for which to return near duplicates.
            min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as
                near duplicates.
            sensitivity (int): Number of unique buckets two ids must co-occur in to be
                considered a near duplicate pair.

        Returns:
            List: Candidate duplicates for provided text label.

        """
        if sensitivity > self.no_of_bands:
            raise ValueError('Sensitivity must be <= no of bands.')

        bands = np.hsplit(
            np.array(minhash_signature),
            self.no_of_bands
        )

        hashable_signature = tuple(hash_value for hash_value in minhash_signature)

        candidates_set = set()
        for band_id, band in enumerate(bands):
            bucket_id = mmh3.hash64(tuple(band), self.seed)[0]
            candidates = self._buckets.get(band_id, bucket_id)
            candidates_set.update(candidates)

        candidates_set.remove(hashable_signature)

        return self._candidate_duplicates(
            hashable_signature,
            candidates_set,
            sensitivity,
            min_jaccard
        )

    def adjacency_list(self, min_jaccard=None, sensitivity=1):
        """ Returns adjacency list.

        Iterates over texts, pairing each text with a list of labels whose relationships with
        each text are above a certain threshold.

        Can be used to create an undirected graph for texts in the LSH object.

        Args:
            min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as near
                duplicates.
            sensitivity (int): Number of unique buckets two ids must co-occur in to be
                considered a near duplicate pair.

        Returns:
            Dict: Adjacency list.

        """
        if sensitivity > self.no_of_bands:
            raise ValueError(
                'Sensitivity must be <= no of bands.'
            )

        adjacency_list = {}

        for label in self._i_bucket.keys():
            buckets = self._i_bucket.get(label)
            candidates = self._candidate_duplicates(
                buckets, label, sensitivity, min_jaccard
            )
            adjacency_list[label] = candidates

        return adjacency_list
