import math
import mmh3

from akin import DictionaryArray


class LSH:
    """ Locality Sensitive Hashing.

    Attributes:
        no_of_bands (int): Number of bands used in model.
        seed (int): Random seed used for hashing.
        permutations (int): Number of permutations used in MinHash.

    """

    def __init__(self, no_of_bands, permutations=None, seed=1):
        """ Initialize the LSH object.

        Args:
            no_of_bands (int): Number of bands to break minhash signature into.
            permutations (int): Number of permutations (hashes) in minhash signature.
            seed (int): Random seed used for hashing.

        """
        self.no_of_bands = no_of_bands
        self.seed = seed
        self._buckets = DictionaryArray(no_of_bands)
        self.permutations = permutations

    def _lsh(self, signature):
        """ Break signatures into bands and hash components to buckets.

        Args:
            signature (tuple): a minhash signature.

        """
        band_hashes = []
        band_size = math.ceil(len(signature) / self.no_of_bands)
        for i in range(0, self.permutations, band_size):
            band = str(signature[i:i + band_size])
            bucket_id = mmh3.hash64(band, self.seed)[0]
            band_hashes.append(bucket_id)

        return band_hashes

    @staticmethod
    def _jaccard_similarity(query_signature, candidate):
        """ Estimate jaccard similarity ratio of signatures.

        Args:
            query_signature (tuple): Query minhash signature.
            candidate (dict): Candidate signature.

        Returns:
            float: Estimated jaccard ratio.

        """
        intersection = len(set(query_signature) & set(candidate))
        union = len(set(query_signature) | set(candidate))
        jaccard_ratio = intersection / union

        return jaccard_ratio

    def _candidate_duplicates(
            self,
            query_signature,
            candidates,
            sensitivity=1,
            jaccard_threshold=None,
            include_similarity=False
    ):
        """ Identify candidate duplicates and check Jaccard Similarity.

        Args:
            query_signature (tuple): Query minhash signature.
            candidates (dict): List of bucket ids.
            sensitivity (int): Number of identical buckets two ids must occur
                in to be considered a near duplicate pair.
            jaccard_threshold (float): Minimum Jaccard Similarity for documents to be
                counted as near duplicates.
            include_similarity (bool): return similarity alongside estimated near duplicates.

        Returns:
            list: Near duplicate document ids.

        """
        # Apply Jaccard threshold and unzip pairs.
        if jaccard_threshold or sensitivity != 1 or include_similarity:
            matches = []
            for candidate, occurrence_count in candidates.items():
                if sensitivity != 1:
                    if occurrence_count < sensitivity:
                        continue

                if jaccard_threshold or include_similarity:
                    jaccard_ratio = self._jaccard_similarity(query_signature, candidate)

                    if jaccard_threshold:
                        if jaccard_ratio < jaccard_threshold:
                            continue

                    if include_similarity:
                        matches.append((jaccard_ratio, candidate))
                    else:
                        matches.append(candidate)

                else:
                    matches.append(candidate)

            return matches

        else:
            return list(candidates)

    def update(self, minhash_signatures):
        """ Updates LSH object with new MinHash matrix and labels.

        Args:
            minhash_signatures (list): new minhash signatures to add to LSH object.

        """
        if not self.permutations:
            self.permutations = len(minhash_signatures[0])
        else:
            if len(minhash_signatures[0]) != self.permutations:
                raise IndexError(
                    f'Number of permutations in minhash must be {self.permutations} to match LSH model.'
                )

        for signature in minhash_signatures:
            for band_id, bucket_id in enumerate(self._lsh(signature)):
                self._buckets.update(band_id, key=bucket_id, value=signature)

    def remove(self, minhash_signatures):
        """ Remove label and associated text signature from model.

        Args:
            minhash_signatures (list): minhash signatures to remove from model.

        """
        for signature in minhash_signatures:
            for band_id, bucket_id in enumerate(self._lsh(signature)):
                self._buckets.remove_value(band_id, key=bucket_id, value=signature)

    def query(self, minhash_signature, min_jaccard=None, sensitivity=1, include_similarity=False):
        """ Returns near duplicates from model.

        Takes a provided text label and returns a list of labels for texts whose
        similarity with the provided text is above a provided threshold.

        Can be used to create a recommendation model.

        Args:
            minhash_signature (tuple): minhash signature of text for which to return near duplicates.
            min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as
                near duplicates.
            sensitivity (int): Number of unique buckets two ids must co-occur in to be
                considered a near duplicate pair.
            include_similarity (bool): return similarity alongside estimated near duplicates.

        Returns:
            list: Candidate duplicates for provided text label.

        """
        if sensitivity > self.no_of_bands:
            raise ValueError('Sensitivity must be <= no of bands.')

        candidates_dict = {}
        for band_id, bucket_id in enumerate(self._lsh(minhash_signature)):
            candidates = self._buckets.get(band_id, bucket_id)
            for candidate in candidates:
                existing_key = candidates_dict.get(candidate)

                if existing_key:
                    existing_key += 1
                else:
                    candidates_dict[candidate] = 1

        del candidates_dict[minhash_signature]

        near_duplicates = self._candidate_duplicates(
            minhash_signature,
            candidates_dict,
            sensitivity,
            min_jaccard,
            include_similarity
        )

        return near_duplicates

    def get_minhashes(self):
        """ Returns set of minhashes contained in LSH model.

        Returns:
            set: set of all unique minhashes within LSH model.

        """
        values = self._buckets.values()
        return values

    def adjacency_list(self, minhash_signatures=None, min_jaccard=None, sensitivity=1):
        """ Returns adjacency list.

        Iterates over texts, pairing each text with a list of labels whose relationships with
        each text are above a certain threshold.

        Can be used to create an undirected graph for texts in the LSH object.

        Args:
            minhash_signatures (list): list of minhash signatures to use as keys.
            min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as near
                duplicates.
            sensitivity (int): Number of unique buckets two ids must co-occur in to be
                considered a near duplicate pair.

        Returns:
            dict: Adjacency list.

        """
        if not minhash_signatures:
            minhash_signatures = self._buckets.values()

        if sensitivity > self.no_of_bands:
            raise ValueError(
                'Sensitivity must be <= no of bands.'
            )

        adjacency_list = {}

        for signature in minhash_signatures:
            adjacency_list[signature] = self.query(signature, min_jaccard, sensitivity)

        return adjacency_list
