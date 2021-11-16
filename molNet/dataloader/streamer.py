from tqdm import tqdm

from molNet import MOLNET_LOGGER


class DataStreamer:
    def __init__(
            self, dataloader, cached=False, progress_bar_kwargs=None, iter_None=True
    ):
        self._iter_None = iter_None
        if progress_bar_kwargs is None:
            progress_bar_kwargs = {}
        self._progress_bar_kwargs = progress_bar_kwargs
        self.dataloader = dataloader
        self._cached = cached
        self._cache_data = []
        self._all_cached = False

        self._position = 0
        self._in_iter = False

    @classmethod
    def generator(cls, **kwargs):
        def _generator(*args, **skwargs):
            return cls(*args, **{**kwargs, **skwargs})
        return _generator

    def get_all_entries(self, *args, **kwargs):
        return self.get_n_entries(self.dataloader.expected_data_size, *args, **kwargs)

    def get_n_entries(self, n: int, progress_bar=True):
        dat = []
        if len(self._cache_data) < n and not self._all_cached:
            if progress_bar:
                g = tqdm(enumerate(self), total=n, **self._progress_bar_kwargs)
            else:
                g = enumerate(self)
            if self.cached:
                for j, d in g:
                    if j >= n:
                        break
            else:
                for j, d in g:
                    dat.append(d)
                    if j >= n:
                        break
            g.close()
            self.close()

        else:
            l = len(self._cache_data)
            if l < n:
                n = l
            if progress_bar:
                return [
                    self._cache_data[i]
                    for i in tqdm(range(n), total=n, **self._progress_bar_kwargs)
                ]
        if self.cached:
            return self._cache_data[:n]
        return dat[:n]

    @property
    def cached(self):
        return self._cached

    def clear_cache(self):
        self._cache_data = []
        self._all_cached = False

    @cached.setter
    def cached(self, cached: bool):
        if cached != self._cached:
            self.clear_cache()
            self._cached = cached

    def get_iterator(self):
        raise NotImplementedError()

    # def next(self):

    def __next__(self):
        try:
            k = next(self._iter)
            if k is not None:
                k = self.update_data(k)

            if not self._iter_None:
                while k is None:
                    self.dataloader.expected_data_size -= 1
                    self._removed += 1
                    k = next(self._iter)
                    if k is not None:
                        k = self.update_data(k)

        except StopIteration:
            if self._cached:
                self._all_cached = True
            if self._position != self.dataloader.expected_data_size + self._removed:
                MOLNET_LOGGER.warning(
                    f"{self.dataloader} returns a different size ({self._position}) than expected({self.dataloader.expected_data_size}), {self._removed} entries where removed"
                )

            raise StopIteration

        if self._cached:
            self._cache_data.append(k)
        self._position += 1

        return k

    def close(self):
        if self._in_iter:
            self._iter.close()
            self._in_iter = False

    def __iter__(self):
        if not self._in_iter:
            self._position = 0
            self._removed = 0
            self._iter = self.get_iterator()
            self._in_iter = True
        return self

    def update_data(self, d):
        return d
