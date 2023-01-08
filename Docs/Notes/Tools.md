---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (math-583)
  language: python
  metadata:
    debugger: true
  name: math-583
  resource_dir: /home/user/.local/share/jupyter/kernels/math-583
---

```{code-cell}
:tags: [hide-cell]

%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'  # Use greyscale as a default.
import mmf_setup; mmf_setup.nbinit()
import logging
logging.getLogger('PIL').setLevel(logging.ERROR)  # Suppress PIL messages
import PIL
```

# Tools

Here we describe some tools and utilities for working with data etc. in this course.
This is not core material, but often needed to get other things done.

## Images

:::{margin}
**plethora:** any unhealthy repletion or excess.  Note that this generally has a
negative connotation.
:::
There is a [plethora](https://www.oed.com/oed2/00181532) of different image formats.  To
deal with this, we use the [Pillow][] (PIL) library which an read, convert, display, and
generally manipulate images.  We will use this to convert images into NumPy arrays for
processing, making use of the following formats.  Consider an image of size `(Nx, Ny)`
where there are `Nx` pixels in the horizontal direction and `Ny` pixels vertically.

* `L`, `shape=(Nx, Ny)`: 8-bit pixels (`dtype=uint8`) gray-scale images.  Each pixel has
    a value from 0 (black) to 255 (white).  By keeping images in this format, we save
    space, but often in processing, we will convert to floating point.
* `RGB`, `shape=(Nx, Ny, 3)`: Each pixel is represented by 3 `uint8` numbers
    representing the Red, Green, and Blue channels.
* `RGBA`, `shape=(Nx, Ny, 4)`: Each pixel is represented by 4 `uint8` numbers
    representing the Red, Green, Blue, and Alpha channels.  The alpha channel represents
    transparency, and is used when combining or "blending" images.

:::{warning}
Floating point numbers (double precision or `float64`) have 64 bits, while `uint8` bytes
have 8 bits.  Converting an array to double precision will increase the memory usage by
a factor of 8.  This can be a problem if you process many images.
:::

```{code-cell}
from math_583 import denoise
im = denoise.Image()
for mode in ["L", "RGB", "RGBA"]:
    u = np.asarray(im.image.convert(mode=mode))
    print(f"{mode=:4}: {u.shape=!r:13}, {u.dtype=}, {u.max()=}, {u.min()=}")
```

Once you have the image as an array, you can display it using matplotlib, or with PIL.

```{code-cell}
fig, ax = plt.subplots()
ax.imshow(u)
display(fig)
plt.close('all')  # Otherwise, the matplotlib image will stay open and display below.

display(PIL.Image.fromarray(u))
```

:::{warning}
Note that [`plt.imshow()`][] respects the pixel order of PIL with the origin at the upper
left.  If you want to plot images in mathematical order, you will need to specify
`origin='lower'` and specify the extents of the array.  See [*origin* and *extent* in
`imshow`][].

:::{margin}
See [Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html) for more
information about the matplotlib color maps and a discussion about human perception of
color, and [Color Wheels are wrong?](https://blog.asmartbear.com/color-wheels.html).
:::
If using matplotlib to display a grey-scale image (`mode='L'`), you will need to specify
the color-map `cmap=gray` and may need to specify `vmin=0` and `vmax=255`, otherwise you
will get the default `viridis` color-map with auto-scaled ranges.  We often set the
default on import (i.e. in the `math_583.denoise` package).

Finally, `ax.axis('off')` is useful.
:::

```{code-cell}
im = denoise.Image()
with plt.style.context({}, after_reset=True):
    fig, axs = plt.subplots(1, 2, figsize=(4,2))
    u = np.asarray(im.image.convert(mode='L'))
    ax = axs[0]
    ax.imshow(u)
    
    # The right way:
    ax = axs[1]
    ax.imshow(u, vmin=0, vmax=255, cmap='gray')
    ax.axis('off')
```

[`plt.imshow()`]: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>
[Pillow]: <https://python-pillow.org/>

[*origin* and *extent* in `imshow`]: <https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html>
