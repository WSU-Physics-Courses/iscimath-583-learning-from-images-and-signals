---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3 (math-583)
  language: python
  name: math-583
---

```{code-cell} ipython3
:hide-cell:

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:PhaseReconstruction)=
Phase Reconstruction
====================

Consider the following problem: Given a set of images $I_n(x)$, how can we reconstruct $\phi(x)$?
\begin{gather*}
  I_n(x) = b(x) + a(x)\cos\Bigl(\phi(x) + \theta_n\Bigr) + \eta_n(x).
\end{gather*}
Here we assume that $\eta_n(x)$ is noise and relatively small.  In what follows, we will
suppress the index $x$.

A simple strategy is to note that
\begin{gather*}
  I_n = b + a\Bigl(\cos(\phi)\cos(\theta_n) - \sin(\phi)\sin(\theta_n)\Bigr) + \eta_n.
\end{gather*}
Thus, up to the noise $\eta_n$, the image lives in a 3-dimensional subspace spanned by
the basis vectors
\begin{gather*}
  \sp 
\end{gather*}



Phase reconstruction 
