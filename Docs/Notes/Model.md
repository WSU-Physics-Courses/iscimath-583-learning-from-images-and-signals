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
  resource_dir: /home/user/.local/share/jupyter/kernels/math-5
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

### Inverse Problems

Solving an inverse problem is tantamount to analyzing observations and calculating to the best of our ability the factors that caused them. Inverse problems are said to be *well-posed* if:

1. A solution exists
2. The solution is unique
3. The behavior of solutions change continuously with respect to the data

The third condition ensures that small deviations in measurement do not lead to very different outcomes in our solution. When such problems are well-posed, they are often amenable to being solved by computer algorithms. If an inverse problem is not well posed, we made need to tweak it before we can employ numerical techniques - we will often use regularization methods to make such tweaks.

### Ridge Regression

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

### Getting a Handle on Noise

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

### Blur and Convolutions

In image processing it is useful to model blurring, either as an effect we would like to add to or remove from an image. To do this we calculate the *convolution* of a true/sharp image $u$ with a function $k$ (called the kernel or point spread function). The convolution of $k$ and $u$ is denoted:

$$f = k * u$$

And for measurable functions $k$ and $u$ is defined as follows:

$$k*u(x) = \int_0^x k(x-y)u(y)dy$$

The discrete convolution for two functions $f(x,y)$ and $k(x,y)$ of size $M \times N$ is given by:

$$f*h = \frac{1}{MN}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} k(x-m,y-n)f(m,n)$$

The convolution is commutative $k*u = u*k$, associative $(k*u)*v = k*(u*v)$, and for translation by $\tau$ we have $\tau(k*u) = (\tau k)*u = k*(\tau u)$.

### Filtering and Fourier Transforms

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

### Calculus of Variations

We will often use variational methods in image processing; the Calculus of Variations is a field of analysis that uses small perturbations in functions to find their extrema. A common type of function to analyze with this field is called a functional, which we take to be a map from some set of functions to the real numbers (and often in the form of a definite integral).

The typical problem is to find a $u$ that yields:

$$\inf_{u \in V} F(u)$$

Where $F(u) = \int_\Omega f(x,u(x),\nabla u(x))dx$, $\Omega \subset \mathbb{R}^N$ is bounded, $V$ is some specified set of admissible functions (often a Banach space), and $f(x,u,\xi)$ is given.

### Gradient Descent

If we want to minimize some differentiable function $F$, we can employ gradient descent to try to approximate the stationary points of $F$ (points where $\nabla F = 0$). If $F$ is convex, then any stationary point is a global minimizer.

If we are minimizing $F(u)$ over the vector space $u \in X$ then we can introduce a paramaterization. Let $u_0 \in X$ and let $u(t) \in X$ for $t \geq 0$ satisfy the initial value problem:

$$u(0) = u_0$$

$$\frac{du}{dt} = -\nabla F(u(t)), \text{ for } t > 0$$

In other words, $u(t)$ starts at some initial point in the input space $u_0$, and then travels in the direction of fastest decrease for $F$. As $t$ increases, $F(u(t))$ will decrease, as:

\begin{align*}
\frac{d}{dt}F(u(t)) &= \nabla F(u(t)) \frac{du}{dt}\\
&= -\|\nabla F(u(t))\|^2\\
&\leq 0
\end{align*}

In the discrete approximation we define a time step $\Delta t > 0$ and define $u^n = u(n\Delta t)$ for $n = 0,1,2,...$

Applying the forward Euler method to our initial value problem starts with $u^0 = u_0$ and evolves like:

$$u^{n+1} = u^n - \Delta t \nabla F(u^n)$$

Since this is only an approximation, we should check to see that unless a steady state is reached or $\nabla F(u^n) = 0$ that the inequality:

$$F(u^{n+1}) < F(u^n)$$

Is satisfied, and if it's not we need to decrease $\Delta t$ and take smaller steps.

It should be noted that if $f$ is convex and $\nabla F$ is Lipschitz continuous with Lipschitz constant $L$ then with $k$ iterations with the fixed step size $\Delta t < \frac{1}{L}$ then:

$$F(u^k) - F(u^*) \leq \frac{\|x^0-x^*\|^2}{2k\Delta t }$$

Where $u^* = \arg \min_u F(u)$; i.e. under these conditions gradient descent is guaranteed to converge and it converges at a rate of $O(1/k)$.

### Finite Differences

Finite difference methods are ways to approximate functions and their derivatives to solve differential equations. Such formulas are derived using Taylor's formula.

If $u$ is a one dimensional function with $u:[a,b] \to \mathbb{R}$ then we can discretize $[a,b]$ into a sequence of $M$ points: $x_0 = a$, $x_1 = x_0 + \Delta x$, $x_2 = x_0 + 2\Delta x, \cdots, x_0 + M\Delta x, x_0 + (M+1)\Delta x = b$ so that $\Delta x = \frac{b-a}{M+1}$.

For ease of notation, denote $u_j$ by $u(x_j)$. If $u \in C^2[a,b]$ on the interval $[a,b]$ then we can approximate its derivative at $x_j$ by:

$$u'(x_j) \approx \frac{u_{j+1}-u_j}{\Delta x}, j = 0,...,M$$

This is simply the slope of the line joining the points $(x_j,u_j)$ and $(x_{j+1},u_{j+1})$ and serves as a linear approximation to the derivative of $f$. This approximation has error $O(\Delta x) = -\frac{\Delta x}{2}u''(\xi)$ for some $x_j \leq \xi \leq x_{j+1}$.

We can also approximate backwards:

$$u'(x_j) \approx \frac{u_{j}-u_{j-1}}{\Delta x}, j = 1,...,M+1 $$

If $u \in C^3[a,b]$ then the second order central finite difference is given by:

$$u'(x_j) \approx \frac{u_{j+1}-u_{j-1}}{2\Delta x}, j = 1,...,M$$

With error $O(\Delta x^2) = -\frac{\Delta x^2}{6}u''(\xi)$ for some $x_{j-1} \leq \xi \leq x_{j+1}$.

If $u \in C^4[a,b]$ we can approximate its second derivative by the second order central finite difference:

$$u''(x_j) \approx \frac{u_{j+1}-2u_j+u_{j-1}}{\Delta x^2}, j = 1,...,M$$

With error $O(\Delta x^2) = - \frac{\Delta x^2}{12}u^{(4)}(\xi)$, $x_{j-1} \leq \xi \leq x_{j+1}$.

In two dimensions, the formulation is similar. Let $u:[a,b]\times[c,d]\to \mathbb{R}$ and partition each interval by step sizes $\Delta x = \frac{b-a}{n}$ and $\Delta y = \frac{d-c}{m}$. If we draw vertical (for $x$) and horizontal (for $y$) lines at $x_i = a + i\Delta x$ and $y_j = c + j\Delta y$ for $i=0,...,n$ and $j = 0,...,m$ then we have a grid over our domain. The points where two grid lines intersect are called the mesh points of the grid.

By the above, we have for second order central difference approximations:

\begin{align*}
\frac{\partial F}{\partial x}(x_i,y_j) \approx \frac{F(x_{i+1},y_j)-F(x_{i-1},y_j)}{2\Delta x}\\
\frac{\partial F}{\partial y}(x_i,y_j) \approx \frac{F(x_{i},y_{j+1})-F(x_{i},y_{j-1})}{2\Delta y}
\end{align*}

\begin{align*}
\frac{\partial^2F}{\partial x^2}(x_i,y_j) &\approx \frac{F(x_{i+1},y_j)-2F(x_i,y_j)+F(x_{i-1},y_j)}{\Delta x^2}\\
\frac{\partial^2F}{\partial y^2}(x_i,y_j) &\approx \frac{F(x_i,y_{j+1})-2F(x_i,y_j)+F(x_i,y_{j-1})}{\Delta y^2}
\end{align*}

With the obvious errors from the 1-d case.

### Reducing the smoothing effect

We have thus developed a theory that allows us to find a denoised image by minimizing:

$$F(u) = \|Ku-f\| + \|\Gamma u\|^2$$

But which norms shall we use? Our functions presumably come from Sobolev space:

$$W^{1,2}(\Omega) := \{u \in L^2(\Omega), \nabla u \in L^2(\Omega)^2\}$$

Since our images often have sharp borders and thus would have discontinuous gradients. It would then be natural to use the gradient of $u$ and the L^2 norm and minimize:

$$F(u) = \int_\Omega |ku-f|dx + \lambda \int_\Omega |\nabla u|^2 dx$$

The above does work, and we can often find unique solutions characterized by the Euler-Lagrange equation:

$$K^*Ku-K^*f-\lambda \Delta u=0$$

But in practice the Laplacian operator $\Delta$ is simply too strong in the sense that it over smoothes images. The $L^2$ norm of the gradient destroys edges too much. Instead, we may then opt to use the $L^1$ norm of the gradient of $u$ which is  called the total variation. We will now then refer to the "Energy" of the system as:

$$E(u) = \frac{1}{2}\int_\Omega |Ku-f|^2dx + \lambda \int_\Omega \phi(|\nabla u|)dx$$

Where $\phi$ is a strictly convex, nondecreasing function with $\phi(0) = 0$ and $\lim_{s \to \infty} \phi(s) = \infty$ (at linear speed). (These conditions allow us to use the direct method of the calculus of variations). This formulation preserves discontinuities in the sense that the solution to our minimization problem will be a piecewise constant image made up of homogeneous regions separated by sharp edges. In other words, it allows for edge-preserving smoothing. The specific choice of $\phi$ determines the smoothness.

If $K$ is then presumed to be a linear continuous operator we can guarantee a unique solution in a weak sense.

## Part 1: Removing Normally Distributed Noise

Suppose that we are given a noisy image $f$ that is the result of adding normally distributed noise $\eta$ to some original image $u$. That is, $f = u + \eta$ where $\eta \sim \mathcal{N}(0,\sigma)$.

We can approximate $u$ by finding the minimizer of the simple energy functional:

$$E(u) = \|u-f\| + \lambda \|\nabla u\|$$

Which we will employ gradient descent to do. Pseudocode to restore the image is below; recall that to implement it on a real image you will need to import the file and convert it to a matrix of intensity values. Similarly, to display the restored image you will need to convert the matrix of intensities back into an image. To do this, we recommend PIL - the Python Imaging Library.

```{code-cell}
:tags: [hide-cell]

from math_583 import denoise

im = denoise.Image()
d = denoise.Denoise(image=im, sigma=0.4)
imported_original_image = d.u_exact
noisy_image = d.u_noise

def norm(u):
    return np.sqrt((u**2).sum())
```
```{code-cell}
u = imported_original_image #mxn matrix
# eta = generated_gaussian_noise #mxn matrix
# f = u + eta
f = noisy_image
x0 = f
lam = 1 #change this as you like

def E(u):
    return norm(u-f)+lam*norm(grad(u))

def dE(u):
  return second_order_central_difference(u,E)

u_restored = grad_descent(x0,E,dE)
```

Part 2: PQ Denoising

A generalization of our energy functional allows us to take powers of our norms.

$$E(u) = \frac{1}{p}\|\nabla u\|^p + \lambda \frac{1}{q}\|u-f\|^q$$

But the case where $p = q = 1$ is exceptionally interesting for its connection to something called the **flat norm**, which we will build up to now.

The flat norm provides a sense of distance in the space of currents. Currents are the generalized surfaces of geoemetric measure theory. Currents may be realtively tame, or they may be extremely wild objects. We will construct relatively tame objects of currents to build intuition for what they are.

Assume that we have some kind of 2-dimensional surface $M$ in $\mathbb{R}^3$. We can assign a two-vector $\xi(x)$ at each point $x$ of the surface so that each pair spans a small parallelogram tangent to the surface at the point where they are anchored.

If we are then handed a 2-form, which is a function $\omega$ that takes in said 2-vector parallelogram $\xi(x)$ at the point $x$ and produces a real number $\omega(\xi(x))$ (for instance, possibly by projecting the parallelogram down onto the xy-plane and yielding the area there) we have a scalar valued function defined on the surface and can thus integrate it with the appropriate dimension (Hausdorff) measure $\int_M \omega(\xi(x))d\mathcal{H}^2(x)$ to get a number.

For a fixed $k$-dimensional surface $M$ we may view this integral above as a linear map $T_M$ which assigns $k$-forms $\omega \mapsto \int_M \omega(\xi(x))d\mathcal{H}^k(x) \in \R$ to real numbers (it inherits linearity from the integral). This object is an example of a current, and a particularly nice one. Most generally, a current may be an arbitrary element of the dual space to differential forms.

The flat norm works by decomposing a current into two pieces, and then each piece is measured differently. The flat norm then finds the minimum over all such decompositions.

Recall that if $S$ is a $k+1$ dimensional object, then its boundary is $k$ dimensional. i.e the boundary of a 2-d disk is a 1-d circle, the boundary of a 3-d ball is a 2-d sphere. If $T$ is a $k$-dimesional current and $S$ is a $k+1$ dimensional current, then we can write:

$$T = (T-\partial S) + \partial S$$

In measuring the "mass" of the above object, we expect to simply get the mass of $T$ which is not particularly useful. However we can measure $T-\partial S$ as normal $M(T-\partial S)$ but then instead of adding on the mass of the boundary of $S$, we can measure $S$ itself. The definition of the flat norm is:

$$F(T) = \min_{S} (M(T-\partial S) + M(S))$$

We can add some control to the above by adding in the parameter $\lambda = \frac{1}{r}$ where $r$ is the minimum radius of curvature of level sets in the minimizer. We then get:

$$F_\lambda(T) = \min_{S} (M(T-\partial S) + \lambda M(S))$$

*image of flat norm decomposition*

We may view the $L^1TV$ functional (the case with $p = q = 1$) as a special case of the flat norm for currents that are boundaries of $n$-dimensional sets in $\mathbb{R}^n$. 

$$f(u) = \int_R |\nabla u|dx + \lambda \int_R |u-d|dx$$

Suppose that $d = \chi_\Omega$ is an indicator (characteristic) function defined by:

$$\chi_\Omega(x) = \begin{cases} 1 & x \in \Omega\\ 0 & x \not \in \Omega\end{cases}$$

It is then a theorem that in this case $f(u)$ has a minimizer $u = \chi_\Sigma$ which is also a characteristic function. Hence we restrict to minimizing over characteristic functions when the data $d$ is a characteristic function.

Suppose then $u = \chi_\Sigma$ and $d = \chi_\Omega$. Then $f(u) = \int_R |\nabla u| dx + \lambda \int_R |u-d|dx = M(\partial \Sigma) + \lambda M(\Sigma \Delta \Omega)$ where $\Sigma \Delta \Omega$ is the symmetric difference between $\Sigma$ and $\Omega$ and is the set of points that are in either $\Sigma$ or $\Omega$ but not both.

If we give $\Sigma$ a multiplicity of $-1$ and $\Omega$ a multiplicity of $1$ and add them we get a current $S_{\Sigma \Delta \Omega}$ that turns $T_\Omega$ into $T_\Sigma$, i.e.

$$T_\Sigma = T_\Omega - S_{\Sigma \Delta \Omega}$$

Which implies

$$T_{\partial \Sigma} = T_{\partial \Omega} - S_{\partial(\Sigma \Delta \Omega)}$$

Since $S_{\partial(\Sigma \Delta \Omega)} = \partial S_{\Sigma \Delta \Omega}$ we can write:

$$T_{\partial \Omega} = (T_{\partial \Omega} - \partial S_{\Sigma \Delta \Omega}) + \partial S_{\Sigma \Delta \Omega}$$

$$T_{\partial \Omega} = T_{\partial \Sigma} + \partial S_{\Sigma \Delta \Omega}$$

So the flat norm $\mathbb{F}_\lambda$ is given by:

\begin{align*}
\mathbb{F}_\lambda(T_{\partial \Omega}) &= \min_{S_{\Sigma \Delta \Omega}}(M(T_{\partial \Sigma}) + \lambda M (S_{\Sigma \Delta \Omega}))\\
&= \min_{\Sigma \Delta \Omega}(\text{perimeter}(\Sigma) + \lambda |\Sigma \Delta \Omega |)\\
&= \min_\Sigma \int_R |\nabla \chi_\Sigma|dx + \lambda \int_R |\chi_\Sigma - \chi_\Omega|dx\\
&= \min_u \int_R |\nabla u|dx + \lambda \int_R |u-\chi_\Omega | dx
\end{align*}