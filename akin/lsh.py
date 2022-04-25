from collections import defaultdict
import numpy as np
from copy import copy


class LSH:
    """ Locality Sensitive Hashing.

    Attributes:
        no_of_bands (int): Number of bands used in model.
        permutations (int): Number of permutations used in MinHash.

    """

    def __init__(self, minhash=None, labels=None, no_of_bands=None):
        """ Initialize the LSH object.

        Args:
            minhash (np.array): Object returned by MinHash class.
            labels (list, np.array): Iterable, array or pandas series containing labels.
            no_of_bands (int): Number of bands to break minhash signature into.

        """
        # Create default variables
        self.no_of_bands = no_of_bands
        self._buckets = defaultdict(list)
        self._i_bucket = defaultdict(list)
        self.permutations = None

        # Run methods if minhash and labels provided
        if minhash and labels:
            self.permutations = minhash.permutations
            self._lsh(minhash.signatures, labels)
        elif minhash:
            raise ValueError(
                'labels cannot be None if LSH initialised with minhash object.'
            )
        elif labels:
            raise ValueError(
                'minhash object cannot be None if LSH initialised with labels.'
            )

    def _lsh(self, signatures, labels):
        """ Break signatures into bands and hash components to buckets.

        Args:
            signatures (np.array): MinHash signature Matrix.
            labels (list): List of labels for MinHash signatures.

        """
        if not self.no_of_bands:
            self.no_of_bands = self.permutations // 2

        for label, signature in zip(labels, signatures):
            bands = np.hsplit(
                np.array(signature), self.no_of_bands
            )
            for band in bands:
                bucket_id = hash(tuple(band))
                self._buckets[bucket_id].append(label)
                self._i_bucket[label].append(bucket_id)

    def _candidate_duplicates(self, bucket_ids, label, sensitivity, jaccard):
        """ Identify candidate duplicates and check Jaccard Similarity.

        Args:
            bucket_ids (list): List of bucket ids.
            label (str, int, float): Text label.
            sensitivity (int): Number of identical buckets two ids must occur
                in to be considered a near duplicate pair.
            jaccard (float): Minimum Jaccard Similarity for documents to be
                counted as near duplicates.

        Returns:
            List: Near duplicate document ids.

        """
        candidates = defaultdict(int)
        # Retrieve candidate duplicate pairs from model.
        for bucket_id in bucket_ids:
            matches = copy(self._buckets.get(bucket_id))
            matches.remove(label)
            for match in matches:
                candidates[match] += 1

        # Apply sensitivity threshold.
        if sensitivity > 1:
            for key in list(candidates):
                if candidates[key] < sensitivity:
                    del candidates[key]

        # Apply Jaccard threshold and unzip pairs.
        if jaccard:
            for key in list(candidates):
                jaccard_ratio = candidates[key] / self.no_of_bands
                if jaccard_ratio < jaccard:
                    del candidates[key]

        candidates = list(candidates)

        return candidates

    def update(self, minhash, new_labels):
        """ Updates LSH object with new MinHash matrix and labels.

        Args:
            minhash (minhash): MinHash object containing new minhash signatures to
                add to LSH object.
            new_labels (list): List of new labels to add to LSH object.

        """
        if self._i_bucket:
            # Check if texts already exist in model.
            if set(
                    self._i_bucket.keys()
            ).intersection(
                set(new_labels)
            ) != set():
                raise ValueError(
                    'At least one provided label already exists in model.'
                )

            if self.permutations != minhash.permutations:
                raise ValueError(
                    'Number of permutations in minhash must be {} to match LSH model.'.format(
                        self.permutations
                    )
                )

        else:
            # Create parameters for new model.
            self.permutations = minhash.permutations

        # Update model.
        self._lsh(minhash.signatures, new_labels)

    def query(self, label, min_jaccard=None, sensitivity=1):
        """ Returns near duplicates from model.

        Takes a provided text label and returns a list of labels for texts whose
        similarity with the provided text is above a provided threshold.

        Can be used to create a recommendation model.

        Args:
            label (str, int, float): Label of text for which to return near duplicates.
            min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as
                near duplicates.
            sensitivity (int): Number of unique buckets two ids must co-occur in to be
                considered a near duplicate pair.

        Returns:
            List: Candidate duplicates for provided text label.

        """
        if sensitivity > self.no_of_bands:
            raise ValueError(
                'Sensitivity must be <= no of bands.'
            )

        buckets = self._i_bucket.get(label)

        if not buckets:
            raise KeyError(
                'Label {} does not exist in model'.format(label)
            )

        return self._candidate_duplicates(
            buckets, label, sensitivity, min_jaccard
        )

    def remove(self, label):
        """ Remove label and associated text signature from model.

        Args:
            label (str, int, float): Label for text to be removed from model.

        """
        buckets = self._i_bucket.get(label)

        if not buckets:
            raise KeyError(
                'Label {} does not exist in model.'.format(label)
            )

        for bucket in buckets:
            self._buckets[bucket].remove(label)

            if not self._buckets[bucket]:
                del self._buckets[bucket]

        del self._i_bucket[label]

    def contains(self):
        """ Returns a list of all labels contained in the model.

        Returns:
             List: All labels for texts contained in the model.

        """
        return list(self._i_bucket)

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
