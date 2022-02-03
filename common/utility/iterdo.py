import numpy as np
import tensorflow as tf
from nested_lookup import nested_delete
import re

def plot_to_image(figure):
    """Convert mpl plot to tf summary image."""
    import io
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0) # add batch dim
    return image

def compare_tensors(T1, T2):
    """Compare all elements of two tensors.
    Return False if one element is not equal."""
    for t1, t2 in zip(T1, T2):
        if t1 and t2 and not all(tf.nest.map_structure(np.array_equal, t1, t2)):
            yield False # not equal
    yield True # equal

def val_in_dict(val, d, exclude="nEpochs"):
    """Return key for val in dict if it exists."""
    val = nested_delete(val, exclude)
    d   = nested_delete(d, exclude)
    for k, v in d.items():
        if v == val:
            return k

def split(data, frac=0.8):
    """Split array in two by fraction."""
    lim = int(data.shape[0] * frac)
    x, y = np.split(data, [lim])
    return (x, y)

def camel_case_split(ccstr):
    """Convert CamelCase string to array."""
    return re.sub('([a-z])([A-Z])', r'\1 \2', ccstr).split()
