
class DictionaryArray:
    """ Instantiates object containing an array of n dictionaries.

    Each dictionary key has an associated set containing multiple unique values.

    Attributes:
        n_arrays (int): Number of dictionary arrays to include.

    """

    def __init__(self, n_arrays):
        """ Wrapper for easy manipulation of a list of dictionary arrays.

        Args:
            n_arrays (int): Number of dictionary arrays to include.

        """
        if n_arrays < 1:
            raise ValueError('Number of arrays must be an integer of 1 or greater')

        self.n_arrays = n_arrays
        self._hash_arrays = [dict() for i in range(n_arrays)]

    def update(self, array_id, key, value=None):
        """ Updates specified dictionary with key, value.

        By default, values are stored in an empty set.

        Args:
            array_id (int): Location of dictionary to update.
            key: Dictionary key for specified dictionary.
            value: Value to add to set in specified dictionary key.

        """
        if value:
            bucket = self._hash_arrays[array_id].get(key)
            if bucket:
                bucket.add(value)
            else:
                self._hash_arrays[array_id][key] = {value}
        else:
            if not self._hash_arrays[array_id].get(key):
                self._hash_arrays[array_id][key] = set()

    def get(self, array_id, key):
        """ Retrieve set of values for specified dictionary and key.

        Args:
            array_id (int): Location of dictionary to remove key from.
            key: Dictionary key for specified dictionary.

        Returns:
            set: Set of unique values.

        """
        return self._hash_arrays[array_id][key]

    def remove_key(self, array_id, key):
        """ Key to delete from specified dictionary.

        Deletes key from specified dictionary including the associated set of values.

        Args:
            array_id (int): Location of dictionary to remove key from.
            key: Dictionary key for specified dictionary.

        """
        del self._hash_arrays[array_id][key]

    def remove_value(self, array_id, key, value):
        """ Removes values from specified dictionary array.

        Empty dictionary keys are deleted.

        Args:
            array_id (int): Location of dictionary to remove value from.
            key: Dictionary key for specified dictionary.
            value: Value to remove from set in specified dictionary.

        """
        bucket = self._hash_arrays[array_id][key]
        bucket.remove(value)

        if not bucket:
            del self._hash_arrays[array_id][key]

    def values(self):
        """ Returns unique values from dictionaries.
        
        Returns:
            set: Set of all values in dictionary arrays.

        """
        values = set()
        for array in self._hash_arrays:
            for value_set in array.values():
                values.update(value_set)

        return values
