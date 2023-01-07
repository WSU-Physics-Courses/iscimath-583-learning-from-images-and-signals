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
  name: math-583
---

```{code-cell}
import sys
print(sys.executable)
```

```{code-cell}
:cell_style: center
:hide_input: false

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%pylab inline --no-import-all
```

# Assignment 0: Introduction
## Assignment Preparation

+++

Complete the following in one of two ways:

1. A simple brute force solution that you completely understand but which may not be fully accurate.
2. A more sophisticated solution that is accurate to close to machine precision.

For the second method, please feel free to use any tools in the [NumPy](https://numpy.org/doc/stable/) or [SciPy](https://docs.scipy.org/doc/scipy/reference/) libraries.  (You may use [mpmath](https://mpmath.org) to check, but don't use it as part of your solution.)

+++

### Series

+++

Numerically check the following formula:

$$
  \sum_{n=1}^{M} n = \frac{M(M+1)}{2}
$$

```{code-cell}

```

$$
  \sum_{n=1}^{\infty} \frac{1}{n^{2}} = \zeta(2) = \frac{\pi^2}{6}
$$

where $\zeta(s)$ is the [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function).

```{code-cell}

```

### Integrals

+++

Numerically check the following integrals for all viable values of $p$ (both positive and negative):

$$
  \int_0^1 x^p \d{x} = \frac{1}{p+1}
$$

```{code-cell}

```

What about infinite limits?

$$
  \int_{-\infty}^{\infty} e^{-x^2} \d{x} = \sqrt{\pi}
$$

```{code-cell}

```

This one is probably challenging to do accurately:

$$
  \int_{-\infty}^{\infty} e^{-x^2}\sin\frac{1}{x^2}\d{x} = \sqrt{\pi}e^{-\sqrt{2}}\sin(\sqrt{2})
$$

```{code-cell}

```

This should be easy if you could do the previous one:

$$
  \int_0^{\infty} xe^{-x^2}\sin^2\frac{1}{x}\d{x} = \frac{\sqrt{\pi}}{4} G_{14}^{13}\left(
    \left.
    \begin{matrix}
      \tfrac{1}{2}\\
      \tfrac{1}{2} & \tfrac{1}{2} & 0 & -\tfrac{1}{2}
    \end{matrix}
    \right| z=1
  \right)
  =
  0.32006330909018418888\cdots, %37810082287661243051390948375855688739690521501945667936210716620524245639515708
$$
where $G_{mn}^{pq}\bigl(\begin{smallmatrix}\vect{a}_p\\ \vect{b}_q\end{smallmatrix}\big|z\bigr)$ is the [Meijer G-function](https://en.wikipedia.org/wiki/Meijer_G-function).

```{code-cell}

```

### Roots

+++

Find all solutions $x$ to the following equations:

+++

#### Polynomials

+++

$$
  x^2 - (1+\epsilon)x + \epsilon = 0, \qquad
  x = \{1, \epsilon\}
$$

for $\epsilon \in \{1, 10^{-10}, 10^{-20}\}$.

```{code-cell}

```

$$
  x^4 + 3x^2 - 6x + 10 = 0, \qquad
  x = \{1\pm\I, -1\pm 2\I\}
$$

```{code-cell}

```

#### Lambert W function

+++

$$
  xe^x = w, \qquad
  x = L_0(1)
$$

for $w = 1$.


The function $w = W_k(x)$ is the [Lambert W function](https://en.wikipedia.org/wiki/Lambert_W_function).

+++

## More Details

+++

Write a function {func}`math_583.assignment_0.quadratic_equation` which returns the roots of the equation

$$
  ax^2 + bx + c = 0.
$$

For example:

$$
  x^2 - 3x + 2 = (x-2)(x-1)
$$

so we expect

```{code-cell}
from math_583.assignment_0 import quadratic_equation
np.allclose(quadratic_equation(a=1, b=-3, c=2), [1, 2])
```

Note: if you attempt to blindly use the quadratic formula as is done in the default version, you will encounter errors when $b \approx \pm \sqrt{b^2 - 4ac}$ because the two terms can cancel.  This is the main source of error associated with floating point comptations.

1. When will this become a problem?
2. How can you overcome this issue?
3. How will you test your code to make sure it works well?

The goal should be for every function to return an answer that has a relative error comparable to the **machine precision** of the computer: sometimes called $\epsilon = $`eps`.  This is not always possible if the problem is ill-conditioned.

```{code-cell}
for epsilon in [1, 1e-10, 1e-20]:
    roots = quadratic_equation(a=1, b=-(1+epsilon), c=epsilon)
    exact_roots = [1, epsilon]
    print(f"epsilon = {epsilon}")
    print(f"roots =       {roots}")
    print(f"exact roots = {exact_roots}")
    print(f"Roots allclose? {np.allclose(roots, exact_roots)}")
    print()
```

### Floating Point Numbers

+++

**Readings:**
* {cite:p}`Gezerlis:2020` Chapter 2.  Try some of the "experiments" Alex suggests.
* {cite:p}`Goldberg:1991` "What Every Computer Scientist Should Know About Floating-Point Numbers."  This is a rather technical, but complete account about floating point numbers, and contains a detailed analysis of this assignment.

We can use NumPy to see the properties of the floating point numbers.  The defult `float` in python is the IEEE double-precision floating point number which has $64 = 52 + 11 + 1$ bits. 52 of these are used for the **mantissa**, 11 for the **exponent** and 1 for the sign.  Roughly, the relative error in a floating point number is `eps`=$\epsilon = 1/2^{52}\approx 2.22\times 10^{-16}$ (16 digits of precision in decimal), with an exponent that can range from $\pm 2^{10} = \pm 1024$ with a smallest value of `tiny`$\approx 2^{-1024} \approx 5.56\times 10^{-309}$ and largest value of `max`$=2^{1024}\approx 1.798\times 10^{208}$.

*Try to compute these numbers!  In particular, `2**1024.0` causes an overflow... how can you compute $2^{1024}\approx 1.798\times 10^{208}$ using floating point numbers?  Hint:*

$$
  2^n = a\times 10^m, \qquad
  n\log_{10}(2) = \log_{10}(a) + m.
$$

The actual values are a little different as we see below (e.g. `tiny`$=2^{-1022}=2.225\times 10^{-308}$) because of some subtleties in the actual implementation of the IEEE standard.

```{code-cell}
import numpy as np
print(np.finfo(float))
```

```{code-cell}

```
