import torch
import torch.nn as nn

class BufferList(nn.Module):
    def __init__(self, buffers=None, *, persistent: bool = True):
        super().__init__()
        self._size = 0
        if buffers is not None:
            for i, buf in enumerate(buffers):
                name = str(i)
                if not name or "." in name:
                    raise KeyError("Buffer name must be non-empty and contain no '.'")
                self.register_buffer(name, buf, persistent=persistent)
            self._size = len(buffers)

    def __getitem__(self, idx):
        if not (0 <= idx < self._size):
            raise IndexError(idx)
        return getattr(self, str(idx))

    def __len__(self):
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield getattr(self, str(i))

    def append(self, buf, *, persistent: bool = True):
        """Append a new buffer to the list."""
        name = str(self._size)
        self.register_buffer(name, buf, persistent=persistent)
        self._size += 1

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"
