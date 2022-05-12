from collections.abc import MutableMapping


class DictionaryArray:
    def __init__(self, n_arrays):
        if n_arrays < 1:
            raise ValueError('Number of arrays must be an integer of 1 or greater')

        self._hash_arrays = [{}] * n_arrays

    def update(self, array, key, value=None):
        pass

    def remove_key(self, array, key):
        del self._hash_arrays[array][key]

    def remove_value(self, array, key, value):
        pass


class BiDirectionalDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.inverse_store = dict()

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __repr__(self):
        pass
