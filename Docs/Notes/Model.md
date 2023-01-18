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

# The Basic Model

## Introduction

Image analysis is typically done on 1-D, 2-D, or 3-D images that may be in greyscale or color. A 1-D image is called a signal and is typically thought of as a function that conveys some information about a phenomenon. A 2-D image is what we typically think of when we hear the word "image" and is what is being displayed to you right now on your computer's monitor. 3-D images arise all the time in various forms of modeling and medical imaging, and in particular from CT scans of patients.

The atomic elements of a 2-D image on a screen (which will be our area of focus) are called pixels, and the number of pixels in a given image is called the image's resolution. When a manufacturer claims to produce a "three-megapixel" camera, they are claiming that their camera produces images that have $3,000,000$ pixels. A $1920\times 1080$ display has a total of $1920\times 1080 = 2,073,600$ pixels (2.07 MP) that it can use to display an image.

A 2-d greyscale image can be thought of as a scalar valued function $f$ that assigns points on the plane to a brightness $(x,y) \mapsto f(x,y)$ at that point. Color images are then vector valued functions $(x,y) \mapsto (f_R,f_G,f_B)$ wherein the components of the output are the intensities in the red, green, and blue color channels respectively (This is the RGB color model - there are others).

Images are digitized by sampling the coordinates $(x,y)$ and quantizing their intensity $f(x,y)$ so that we end up with a finite number of (hopefully standardized) points and values to work with. The output of these processes is a digital image represented by an $M \times N$ dimensional matrix, which for a greyscale image is given by:

$$f = [f_{i,j}]$$

The number of quantized intensity levels will then be a fixed finite number $L$ often chosen to be $L = 256$ so that $f_{i,j} \in \{0,1,2,...,255\}$ wherein $f_{i,j} = 0$ represents a black pixel at location $i,j$ in the image, $f_{i,j} = 255$ instead represents a white pixel and levels in between are varying intensites of grey. For a color image, $f_{i,j} = (255,0,0)$ is a red pixel, $(0,255,0)$ a green pixel, $(0,0,255)$ a blue pixel, and combinations inbetween will be an appropriate mix of red, green, and blue.

Images are susceptible to noise or other imperfections that may degrade the information they contain. Thus arises the need for pre-processing to remove or reduce artificial noise and restore the image. In this course we will explore some basic methods of denoising, deblurring, and inpainting (filling in missing information). Implementing these methods often comes down to solving inverse problems.

## Inverse Problems

Solving an inverse problem is tantamount to analyzing observations and calculating to the best of our ability the factors that caused them. Inverse problems are said to be *well-posed* if:

1. A solution exists
2. The solution is unique
3. The behavior of solutions change continuously with respect to the data

The third condition ensures that small deviations in measurement do not lead to very different outcomes in our solution. When such problems are well-posed, they are often amenable to being solved by computer algorithms. If an inverse problem is not well posed, we made need to tweak it before we can employ numerical techniques - we will often use regularization methods to make such tweaks.

## Ridge Regression

Ridge regression, also known as Tikhonov regularization, is one particular method of regularizing ill-posed problems.

Suppose that we have an image $f$ that has degraded according to some process $K$. We would like to find the original undegraded image $u$. This is then tantamount to solving the problem:

$$Ku=f$$

If $K^{-1}$ exists and is continuous then the problem is well-posed and we can easily find $u$:

$$u = K^{-1}f$$

If this is not the case then the problem is ill-posed and we need to do some more work. Often in image processing $K$ is a linear and continuous (and therefore bounded) operator, and we can use least squares to instead solve the adjacent problem:

$$\min_u \|Ku-f\|^2$$

The quantity to be minimized above is called the *discrepancy* of $u$ and can be thought of as the distance between the images $Ku$ and $f$. Note that if we could find a $u$ that makes the discrepancy $0$, then we would have found the true image, since $\|Ku-f\|^2 = 0$ implies $Ku = f$. If no such $u$ exists for us to find, we try to get as close as possible by getting the discrepancy as close to 0 as we can.

By the method of least squares, we know that the following three properties are equivalent to each other:

1. $Ku = f$ has a unique least-squares solution
2. The columns of $K$ are linearly independent
3. $K^*K$ is invertible

Where $K^*$ denotes the adjoint of $K$.

Furthermore, if any of these conditions hold then the least squares solution is given by $\hat{u} = (K^*K)^{-1}K^*f$ which minimizes the discrepancy.

However, this method is relatively simple and will fail if $K^*K$ is not invertible, and even if it is invertible, if it is ill-conditioned we may fail to numerically solve the problem. Hence we may then choose to include a *regularization term*:

$$\|Ku-f\| + \|\Gamma u\|^2$$

For an appropriately chosen Tikhonov matrix $\Gamma$. We have a couple choices for the form of $\Gamma$; in the special case where $\Gamma = \lambda I$ we have a scalar multiple of the identity, our problem becomes:

$$\|Ku-f\| + \lambda\|u\|^2$$

Where $\lambda > 0$. This choice gives preference to those solutions $u$ with smaller norms and is called $L_2$ regularization. Notice that if we set $\lambda = 0$ we are back to our unregularized solution. $\lambda$ then may be thought of a parameter that adjusts the amount of regularization we want.

Often is is useful to make $\Gamma$ a high-pass operator such as the gradient, Laplacian, or some weighted Fourier operator. These can enforce smoothness conditions and improve the conditioning of the problem so that we can solve the problem directly using those numerical techniques. In general if $\Gamma$ is a linear continous operator our solution $\hat{u}$ is given by:

$$\hat{u} = (K^*K + \Gamma^*\Gamma)^{-1}K^*f$$

## Getting a Handle on Noise

All that we have done so far would be sufficient if we knew the exact degradration process $K$. Often though we're simply given an already degraded image and asked to clean it up. Suppose this is the case; we are given a noisy image and asked to denoise it to the best of our abilities. If we assume that the noise is distributed across the image according to some underlying probability desnsity function, we may then view $f$ and $u$ as random variables and make a *maximum a posteriori probability estimate* of $u$. This is a fancy way to say that we would like to find the most likely value of $u$ given $f$ and is expressed as:

$$\arg \max_u P(u|f)$$

If we apply Baye's theorem we have:

$$P(u|f) = \frac{P(f|u)P(u)}{P(f)}$$

Which tells us that if we wish to find the $u$ that gives us $\max_u P(u|f)$ then this is equivalent to finding the $u$ that gives us $\max_{u} P(u)P(f|u)$, noting that $P(f)$ is simply a constant and independent of $u$ and thus we can ignore it. Furthermore we can reframe the problem to finding the $u$ which satisfies:

$$\min_{u} -\log P(u) - \log P(f|u)$$

Recalling that because $\log$ is a monotonically increasing function, $\arg \max \phi = \arg \max \log(\phi)$ for any function $\phi$ and maximizing any function is equivalent to minimizing the negative of said function. The reason why we would want to do this will become obvious soon.

The first term $-\log P(u)$ is called the *prior* on $u$ and can act as either a regularization term or as an assumption on what $u$ is likely to be. The second term $-\log P(f|u)$ describes how $f$ was produced from $u$.

If we assume that our random variables are normally distributed we can write:


$$P(f|u) = \frac{1}{\sqrt{2\pi}\sigma}e^{- \frac{||u-f||^2}{2\sigma^2}}$$
  
$$P(u) = \frac{1}{\sqrt{2\pi}\tau}e^{-\frac{||Lu||^2}{2\tau^2}}$$

Where $\sigma^2 > 0$ and $\tau^2 > 0$ are the variance of the two distributions. Plugging these in our problem becomes to find the $u$ that satisfies:

$$\min_u \frac{1}{2\sigma^2}\|u-f\|^2 + \frac{1}{2\tau^2}\|Lu\|^2 - \log\left(\frac{1}{2\pi \tau \sigma} \right)$$

If we ignore the constant term, factor out $\frac{1}{2\sigma^2}$ and let $\lambda = \frac{\sigma^2}{\tau^2}$ then the above becomes:

$$\arg \min_u \|u-f\|^2 + \lambda\|Lu\|^2$$

Which is a special case of ridge regression with $K = I$.

## Blur and Convolutions

In image processing it is useful to model blurring, either as an effect we would like to add to or remove from an image. To do this we calculate the *convolution* of a true/sharp image $u$ with a function $k$ (called the kernel or point spread function). The convolution of $k$ and $u$ is denoted:

$$f = k * u$$

And for measurable functions $k$ and $u$ is defined as follows:

$$k*u(x) = \int_0^x k(x-y)u(y)dy$$

The discrete convolution for two functions $f(x,y)$ and $k(x,y)$ of size $M \times N$ is given by:

$$f*h = \frac{1}{MN}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} k(x-m,y-n)f(m,n)$$

The convolution is commutative $k*u = u*k$, associative $(k*u)*v = k*(u*v)$, and for translation by $\tau$ we have $\tau(k*u) = (\tau k)*u = k*(\tau u)$.

## Filtering and Fourier Transforms

Often we find the need to filter an image to remove noise or enhance certain features (sharpening, smoothing, etc.) and in doing so we often employ the Fourier transform. The Fourier transform takes a function and decomposes it into its frequency components. Applying the Fourier transform to a function of time yields a function of temporal frequency and transforming a function of space will output a function of spatial frequency.

If $\int_{\mathbb{R}^N} |u(x)|dx < \infty$ then the Fourier transform of $u$ is:

$$\mathcal{F}(u)(\xi) = \int_{\mathbb{R}^N}u(x)e^{-2\pi i\xi\cdot x}dx$$

The Fourier transform enjoys some nice properties:

1. Translation invariance
2. $\mathcal{F}(u*v)(\xi) = \mathcal{F}(u)(\xi)\mathcal{F}(v)(\xi)$

We can also define the inverse Fourier transform:

$$\mathcal{F}^{-1}(u)(x) = \int_{\mathbb{R}^N} u(\xi)e^{2\pi i\xi \cdot x}d\xi$$

And we have that if both $u$ and its Fourier transform have finite integrals, then $u = u_0 = \mathcal{F}^{-1}(\mathcal{F}(u)) = \mathcal{F}(\mathcal{F}^{-1}(u))$ almost everywhere. Furthermore, we have that if $\int |u(x)|dx^2 < \infty$ and $\int|\mathcal{F}(u)(x)dx^2 < \infty$ then $\int |u(x)|dx^2  = \int |\mathcal{F}(u)(x)|dx^2$. 

For ease of notation, we may denote the Fourier transform of $u$ by $\hat{u}$ and the inverse Fourier transform by $\check{u}$.

## Calculus of Variations

We will often use variational methods in image processing; the Calculus of Variations is a field of analysis that uses small perturbations in functions to find their extrema. A common type of function to analyze with this field is called a functional, which we take to be a map from some set of functions to the real numbers (and often in the form of a definite integral).

The typical problem is to find a $u$ that yields:

$$\inf_{u \in V} I(u)$$

Where $I(u) = \int_\Omega f(x,u(x),\nabla u(x))dx$, $\Omega \subset \mathbb{R}^N$ is bounded, $V$ is some specified set of admissible functions (often a Banach space), and $f(x,u,\xi)$ is given.
