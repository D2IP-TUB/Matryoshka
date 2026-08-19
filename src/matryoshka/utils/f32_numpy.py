from typing import Any, Optional, Union

import numpy as np

_original_array = np.array


def patched_array(
    object: Any,
    dtype: Optional[Union[np.dtype, str]] = None,
    copy: Optional[bool] = None,
    order: Optional[str] = 'K',
    subok: bool = True,
    ndmin: int = 0,
    like: Optional[Any] = None
) -> np.ndarray:
    if dtype is not None:
        return _original_array(
            object, dtype=dtype, copy=copy, order=order,
            subok=subok, ndmin=ndmin, like=like
        )

    temp_array = _original_array(
        object, copy=None, order=order, subok=subok, ndmin=ndmin, like=like
    )

    if temp_array.dtype == np.float64:
        dtype = np.float32
    else:
        dtype = temp_array.dtype

    return _original_array(
        object, dtype=dtype, copy=copy, order=order,
        subok=subok, ndmin=ndmin, like=like
    )


def apply_patch():
    """Apply the monkey patch to numpy.array"""
    np.array = patched_array


def remove_patch():
    """Remove the monkey patch and restore original numpy.array"""
    np.array = _original_array


def is_patched():
    """Check if the patch is currently applied"""
    return np.array is patched_array


# Apply the patch automatically when the module is imported
# apply_patch()


def array32(*args, **kwargs):
    """Convenience function that always creates float32 arrays when possible"""
    if 'dtype' not in kwargs:
        kwargs['dtype'] = np.float32
    return _original_array(*args, **kwargs)


def array64(*args, **kwargs):
    """Convenience function that always creates float64 arrays when possible"""
    if 'dtype' not in kwargs:
        kwargs['dtype'] = np.float64
    return _original_array(*args, **kwargs)
