
class DictionaryArray:
    def __init__(self, n_arrays):
        """ Wrapper for easy manipulation of a list of dictionary arrays.

        Args:
            n_arrays(int): Number of dictionary arrays to include.

        """
        if n_arrays < 1:
            raise ValueError('Number of arrays must be an integer of 1 or greater')

        self.n_arrays = n_arrays
        self._hash_arrays = [dict() for i in range(n_arrays)]

    def update(self, array_id, key, value=None):
        """

        Args:
            array_id(int):
            key(int):
            value(int):

        Returns:

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

    def remove_key(self, array_id, key):
        """

        Args:
            array_id(int):
            key(int):

        Returns:

        """
        del self._hash_arrays[array_id][key]

    def remove_value(self, array_id, key, value):
        """

        Args:
            array_id(int):
            key(int):
            value(int):

        Returns:

        """
        bucket = self._hash_arrays[array_id][key]
        bucket.remove(value)
        if not bucket:
            del bucket
