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

Recall that if $S$ is a sufficiently smooth (its boundary is not a fractal) $k+1$ dimensional object, then its boundary is $k$ dimensional. i.e the boundary of a 2-d disk is a 1-d circle, the boundary of a 3-d ball is a 2-d sphere. If $T$ is a $k$-dimesional current and $S$ is a $k+1$ dimensional current, then we can write:

$$T = (T-\partial S) + \partial S$$

In measuring the "mass" of the above object, we expect to simply get the mass of $T$ which is not particularly useful. However we can measure $T-\partial S$ as normal $M(T-\partial S)$ but then instead of adding on the mass of the boundary of $S$, we can measure $S$ itself. The definition of the flat norm is:

$$F(T) = \min_{S} (M(T-\partial S) + M(S))$$

We can add some control to the above by adding in the parameter $\lambda = \frac{1}{r}$ where $r$ is the minimum radius of curvature of level sets in the minimizer. We then get:

$$F_\lambda(T) = \min_{S} (M(T-\partial S) + \lambda M(S))$$

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

### Graph-Based Methods

![](figures/image_heat_graph_flow.png)

We can think of an image as a graph by taking the pixels to be vertices whose value is the intensity of that pixel. Color channels may analyzed separately or together depending on the application. A typical representation then connects each vertex to its 4 nearest neighbors defined by the original image to generate a gridlike structure.

If we like, for a particular image $u$ we may solve the heat equation, which is in general tantamount to minimizing $\int |\nabla u|^2$. Notice however that there is no data fidelity term here, and so when you put a $u$ in and try to minimize the solutions will simply flow downhill to a constant image and do not need to stay close to the original in any way. If you recall from PDEs, letting the solution run is equivalent to repeatedly convolving $u$ with a heat kernel, which brings a particular point closer to the average value of its neighbors. 

On a graph we can also solve the heat equation by calculating for a particular vertex a new value based on the average between it and its 4 neighbors. This will smooth the original image and hence get rid of noise, but the difficulty in this of course is that the strong smoothing properties will end up smoothing over and hence destroying edges. We can however devise workarounds that preserve the edges in our image.

Suppose in the following graph that the red vertices have a value that is different than the blue vertices. Then the edges marked with green represent a difference between neighbors. One possible technique is to strongly weight those edges connecting like neighbors to preserve data there. Another is to only connect vertices to like vertices in the first place, so that very different neighboring pixels are not communicating in the graph.

![](figures/weighted_image_graph.png)

In particular we may choose to start from the ground up and make sparse connections only to those vertcies that should be talking, or we could choose to start from a complete graph and prune those edges that we don't want. 

First we'll try working from the ground up. Suppose we have our pixels as above, each is connected to its 4 nearest neighbors by an edge and then we weight the edge $e_{ji}$ which connects pixel $p_i$ and $p_j$ by $w_{ji} = |d(p_i)-d(p_j)|$ where $d(p)$ indicates the intensity of the pixel $p$. 

Now we find the minimal spanning tree for our graph; that is we find a path through the graph that touches every vertex so that the sum of the weights on the edges we go through is as small as possible. We then run our averaging algorithm on the minimal spanning tree. Because the minimal spanning tree will not generally make very expensive connections (i.e. connect two pixels with very different values) when we average over neighboring values, our algorithm will not smooth over boundaries (except perhaps in a single spot, since the tree will be connected).

![](figures/minimal_spanning_tree_smoothing.png)

There is one small problem however: the minimal spanning tree will not preserve much information about neighbors in general. Pixels that are similar and spatially close on the image may be considered to be far apart in the graph because of the minimal structure. There are things we may do about this: we could for instance go back after finding the minimal spanning tree and add in edges that cost below some given threshold - this will make our smoothing a bit more effective.

For nonlocal means the technique is quite different. First, we plot our pixels as vertices just like normal, but then instead of connecting each pixel to its 4 nearest neighbors we instead connect each pixel to *every* other pixel to form a complete graph. We then assign weights to each edge based on the similarity of pixel neighborhoods and smooth the resulting graph.

Specifically, for each pixel $p_i$ we let its $(2k+1)\times (2k+1)$ neighborhood of pixel values represent $p_i$ in $\mathbb{R}^{(2k+1)^2}$. I.e. if $k = 3$ then every $p_i$ generates a point $n_i$ in 49 dimensional Euclidean space.

We then compute $w_{ij} = \| n_i - n_j \|$ for all $i,j$ pairs.

For each $p_i$ we then select only those edges $e_{ij}$ such that $w_{ij}$ are the smallest measure of the possible $w_{ij}$'s.

We then prune all edges from the complete graph that have not been selected and smooth over this resulting graph.

### Discrete Laplacian

In working with graphs, it's often useful to employ a meaningful notion of a Laplacian on the graph.

Recall the definition of the second-order central finite difference method:

$$f''(x) \approx \frac{f(x+h)-2f(x)+f(x-h)}{g^2}$$

Where $h$ is the distance between $x$ and two points that are sampled to do the calculation. If we consider a discrete segment of the real line and let $h = 1$ then by the above the second derivative at a point is given by:

$$f''(x) \approx \text{value at previous point} -2 \cdot \text{value at current point} + \text{value at next point}$$

The above definition works fine for graphs equivalent to a line of points. If we have a grid of points in a graph then the second derivative becomes the Laplacian and if $p_1, p_2, p_3, p_4$ are the nearest neighbors of the point $p$ on the graph $u$ then:

$$\Delta u(p) = d(p_1) + d(p_2) + d(p_3) + d(p_4) - 4 d(p)$$

In other words, the average difference between the point and its 4 nearest neighbors on the grid.

### Min-cut max-flow and the $L^1TV$

Suppose that we have a 1-d image (a signal) which we can represent as the graph of a function from $\mathbb{R}\to \mathbb{R}$. We may choose then to view this function as a set $\Omega$ of points in the domain, where the function takes a nonzero value that point is in the set and where the value is zero the point is not in the set.

![](figures/mincut_1.png)

In the above figure the points with a green bar above them are in the set and the points with an orange bar below them are not in the set. Hence we have converted $\Omega$ to a characteristic function, and so we also expect our solution to be a characteristic function.

Recall that for sets that $L^1TV$ we have:

$$F(\chi_\Sigma) = \text{per}(\Sigma) + \lambda|\Sigma \Delta \Omega|$$

One way to calculate this functional is by drawing a source and a sink node and connecting each point in our set to the source and all the points not in our set to the sink, which will form an initial input.

![](figures/mincut_2.png)

We then add weights $\lambda$ to those connections and a value of $1$ to each horizontal connection between the points.

Now we create a cut between the source and the sink which defines a candidate $\Sigma$. Whenever we go up or down we are charged 1 for the jump as it adds 1 to the perimeter (total variation), and we are charged $\lambda$ where we disagree with the original $\Omega$ (exactly like $\lambda|\Sigma \Delta \Omega|$). For the illustration below, we are including in $\Sigma$ all the points above the pink cut and disregarding points below the pink cut.

![](figures/mincut_3.png)

In two dimensions we may cover our image (as a characteristic function) with lines and then correctly add up all of the crossings to get a rough approximation of Crofton's formula, which says that if we fix a line at the origin and rotate it through space, projecting the number of times said line intersects a curve $\varphi$ at a given angle $\theta$ then integrate that function $h(\theta)$ we will get a number proportional to the length of the curve.

$len(\varphi) = c \int_0^{2\pi}h(\theta)d\theta$

![](figures/croftons.png)

So to approximate this we take a grid:

![](figures/croftons_approximation.png)

And at each crossing we get an edge with the value of the difference in height of the characteristic function, 1. We then need to compute weights for these crossings based on the spacing of our lines. We can then form a graph with a source and a sink as before and find the minimum cut.

Alternatively to the Crofton's approximation method (which may not always be so great since there are not enough directions) we could directly compute weights based on the directions in a $5\times 5$ grid.

![](figures/mincut_4_neighbors.png)

Normally there would be 16 directions to consider, but due to symmetry we only need to calculate 3 weights. We can calculate these weights directly by determining what would minimize the error for any unit vector input.

I.e. imagine a plane with slope one in any possible direction and making it so the integrated errors over all possible directions is minimized - this yields better weights than the Crofton method. Essentially the algorithm approximates the norm $|\nabla u|$ in the total variation term with a polygonal approximation

![](figures/mincut_5_norm.png)

### Graph Laplacian and Geometric Diffusion

While we have already defined the discrete Laplacian that works on a square grid, we would like our construction to be more general so that we can apply it to a broader range of graphs and have it make sense.

So imagine that we have a graph and label the verticies $a_1,...,a_n$.

![](figures/graph.png)

Next we can write down a $n \times n$ adjacency matrix $A$, wherein if node $a_i$ is connected to node $a_j$ we put a 1 in that row $i$ and column $j$, and if they're not we put a 0. Note that the matrix will be symmetric since if $a_i$ is connected to $a_j$ then $a_j$ is also obviously connected to $a_i$. For our example above we have:

\begin{align*}
A &= \begin{bmatrix} 0 & 1 & 1 & 1\\
1 & 0 & 0 & 0\\
1 & 0 & 0 & 1\\
1 & 0 & 1 & 0\end{bmatrix}
\end{align*}

Often in image analysis we deal with simple graphs wherein there are no loops and no parallel edges (a node cannot be connected to itself and there is at most one edge connecting any two nodes) this implies for us that $A$ will have all $0$'s down the main diagonal and this matrix encodes all of the connections in the graph.

Furthermore we can form the $n \times n$ diagonal degree matrix, wherein the $i$th row and $i$th column we record the degree of the $i$th node of the graph. For our example we would have:

\begin{align*}
D &= \begin{bmatrix} 3 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 2 & 0\\
0 & 0 & 0 & 2\end{bmatrix}
\end{align*}

We may then choose to define our graph Laplacian $L$ by:

\begin{align*}
L := A-D
\end{align*}

With the note that those who do spectral graph theory typically reverse the order and write it as $D-A$ so that they have positive eigenvalues. When we are comparing $L$ to the normal Laplacian however, it makes sense to use the $A-D$ formulation.

Note that this construction indeed behaves like the Laplacian, and we can verify some properties we expect it to have. For example, we know that the Laplacian of a constant is $0$. We can then check that when $L$ acts on a constant vector we should get the zero vector; and indeed we do. This is because the rows and columns sum of $A$ sum to the degree of the respective nodes, and taking away $D$ cancels them out when we do our matrix product. For our example let $C$ be a constant vector, then:

\begin{align*}
LC &= \begin{bmatrix} -3 & 1 & 1 & 1\\
1 & -1 & 0 & 0\\
1 & 0 & -2 & 1\\
1 & 0 & 1 & -2\end{bmatrix}\begin{bmatrix}c\\c\\c\\c\end{bmatrix}\\
&= \begin{bmatrix}
0\\
0\\
0\\
0
\end{bmatrix}
\end{align*}

Another interesting matrix we can form is $M$ defined by:

\begin{align*}
M := AD^{-1}
\end{align*}

Where the diagonal entries of $D^{-1}$ are the multiplicative reciprocals of the degrees from $D$. For our example we would have:

\begin{align*}
M &= \begin{bmatrix} 0 & 1 & 1 & 1\\
1 & 0 & 0 & 0\\
1 & 0 & 0 & 1\\
1 & 0 & 1 & 0\end{bmatrix}\begin{bmatrix} \frac{1}{3} & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & \frac{1}{2} & 0\\
0 & 0 & 0 & \frac{1}{2}\end{bmatrix}\\
&= \begin{bmatrix}
0 & 1 & \frac{1}{2} & \frac{1}{2}\\
\frac{1}{3} & 0 & 0 & 0\\
\frac{1}{3} & 0 & 0 & \frac{1}{2}\\
\frac{1}{3} & 0 & \frac{1}{2} & 0
\end{bmatrix}
\end{align*}

By construction the columns of $M$ will always sum to 1, making it a left stochastic (Markov) matrix. If $p$ is a probability distribution, i.e. a (stochastic) vector whose entries all sum to 1, notice that defining $P_0 = p$ and $P_{t+1} = Mp_t$  for $t \geq 1$ describes a sequence of probability vectors. In this formulation we may think of the verticies as states and the columns of $Mp_t$ are conditional probabilities - the entry of $Mp_t$ at row $i$ and column $j$ is the probability that the system at time $t + 1$ is in the $i$th state given that the system is currently in state $j$.

If we would prefer $L$ to have $1$'s along the diagonal, we may define the matrix $D^{-\frac{1}{2}}$ which is the same matrix as$D^{-1}$ but with the square roots of every entry on the diagonal. For our example:

\begin{align*}
D^{-\frac{1}{2}} &= \begin{bmatrix} \frac{1}{\sqrt{3}} & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & \frac{1}{\sqrt{2}} & 0\\
0 & 0 & 0 & \frac{1}{\sqrt{2}}\end{bmatrix}
\end{align*}

We can then calculate an alternative formulation $\hat{L}$:

\begin{align*}
\hat{L} &:= D^{-\frac{1}{2}}LD^{-\frac{1}{2}}\\
&= D^{-\frac{1}{2}} A D^{-\frac{1}{2}} - I
\end{align*}

Which we may call the symmetrized graph Laplacian. A trade off for getting 1's on the diagonal is that constant vectors are no longer null vectors - they don't map to zero. However, constant multiples of $D^{-\frac{1}{2}}\mathbb{1}$ where $\mathbb{1}$ is a vector of $1$'s will be. We may then note that this tells us that $0$ is an eigenvalue.

$M$ and $\hat{L}$ have the same spectra (eigenvalues) since $D^{-\frac{1}{2}}MD^{\frac{1}{2}} = D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ defines a similarty transform. Furthermore, if $A$ is simply positive, symmetric, and positive semi-definite then the corresponding spectra for both $M$ and $\hat{L}$ are both in the unit interval $[0,1]$. The reason for this is that $AD^{-1}$ is a matrix whose operator norm $\|M\|_1 = 1$ is equal to one by the fact that the columns all sum to 1 and there are no negative entries, where the operator norm is defined by $\sup_{x \neq 0} \frac{\|Mx\|_1}{\|x\|_1}$. Said norm is obviously a bound on the spectra.

All of this together leads to the diffusion map by Ronald Coifman and Stephane Lafon. To summarize so far:

Let $A$ be any symmetric, non-negative, positive semidefinitie matrix (i.e. the kind we get as adjaceny matricies). In other words, $a_{ij} = a_{ji}$, $a_{ij} \geq 0$, and $v^Tav \geq 0$ for all $v$. Then $M = AD^{-1}$ is a matrkov matrix and $\hat{A} = D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ is a symmetrized Markov matrix that we know is still symmetric and positive semidefinite. Futher more the spectral radius $\rho(\hat{A})$ is $\leq 1$ and if $\phi, \lambda$ are an eigenvector eigenvalue pair of $\hat{A}$ then $D^{-\frac{1}{2}}\phi,\lambda$ are an eigenvector eigenvalue pair of $M$. The spectra $\sigma(\hat{A})$ is $1 = \lambda_1 \geq \lambda 2 \geq \cdots \geq \lambda_n \geq 0$.

Now consider a random walk on a graph. Notice that the number of edges between two points may not make sense as a distance between them, since the probability of reaching one may be very different than the probability of reaching another the same number of edges away.

Because $\hat{A}$ is symmetric and positive semidefinite there exists an eigendecomposition. If $\phi_i$ is the $i$th eigvenvector with eigenvalue $\lambda_i$ then $\hat{A} = \sum_{i=1}^n \lambda_i \phi_i \phi_i^T$. That is, taking outer products of eigenvectors and multiplying them by their respective eigenvaleus yields rank 1 matricies that, when added together recover the matrix. Because our eigenvalues are positive we can write them as $\mu_i^2 = \lambda_i$. We then define the $m$th iterate of $A$, $A^M = \underbrace{AA\cdots A}_m$ (this is analogus to the probability distribution after $m$ steps):

\begin{align*}
\hat{A^m} &= \sum_{i=1}^n \mu_i^{2m} \phi_i\phi_i^T
\end{align*}

Denote now $\phi_i(k)$ to be the $k$th element of the $i$th eigenvector. We now define the diffusion mapping for the $k$th node of the graph by:

\begin{align*}
\Phi_m(k) &= \begin{bmatrix} \mu_1^m \phi_1(k)\\
\mu_2^m \phi_2(k)\\
\vdots\\
\mu_n^m\phi_n(k)
\end{bmatrix}
\end{align*}

This allows us to then map the $k$th node of a graph to a point in $n$ dimensions. Note that quite interestingly, the representation we obtain of the graph generally remains topologically faithful even if we drop all but the first 3 coordinates. This means then that we can then embed each node into the familiar 3 dimensional space and work with it there.

Now that we've embedded our graph in $\mathbb{R}^n$ we can easily define the distance between points in the usual way. The so-called diffusion distance between nodes $k$ and $j$ of the graph is given by $D_m^{\hat{A}}(k,j) := \|\Phi_m(k)-\Phi_m(j) \|$.

### Distributions and Samples

Suppose that you have a single sample as the result of some probabalistic process you're studying. You believe that it's distributed according to some imagined probability distribution. How can you tell, given this sample, whether or not the distribution you have in mind is the correct one?

Most people tend to answer the above by saying one sample isn't enough to tell how likely a given distribution is, we need many samples. Suppose then you get many samples, perhaps a thousand, and you are asked the same question. Have you made any headway?

Consider that even with your thousand samples $x_1,x_2,...,x_{1000}$ the question whether or not a given distribution is the true one underlying the data is equivalent to asking if the one observation $(x_1,x_2,...,x_{1000})$ in a 1000 dimensional product space tells us how likely the product distribution is. We are back to square one, asking if a single sample can tell us how likely a given distribution is.

We can never really be sure whether or not a given observation came from a particular distribution. However, what we can do is say that given some distribution, how likely are we to get this set of data given some number of samples.

Let's devise an experiment to generate some data. Suppose we have a 1 dimensional "camera" with 201 pixels that measures the value of a smooth step function $u(x) = aH(x-x_0) + \eta$ for $x \in [-1,1]$, where $H(x) = 2\tanh(x)-1$ with normally distributed noise $\eta \sim \mathcal{N}(0,\sigma)$.

If we take $\sigma = 1$ and assume $a \in (0,10]$ with $x_0 \in [-1,1]$ how might we find what $x_0$ is?

One option might be to try to fit a curve to the data set using least squares. This is a great option if we know a good function like $H(x)$ that models our data. Sometimes however if there is a lot of noise, then the method will fail.

Another approach is template matching, in which case we take a function that models our data and calculate the inner product of our data with the model function as we shift along the $x$ axis. The inner product will have a maximum value where the functions are most aligned, which can yield an estimate for $x_0$.

If we plot the overlaps (the inner product) of the two as a function of the shifts we will generate a curve. However, if we plot this curve and look at where its maximum is relative to $x_0$ we will notice that there is an inherent bias to its estimates. We can attempt to remove this bias in the following way:

Assume that our estimate for $x_0$ is the true value of $x_0$. Then we can go back and generate data using that $x_0$ and our model. We now then run our algorithm for finding $x_0$ again on our generated data. If we do this a thousand times we will generate a thousand estimates for $x_0$ and we can plot them on a histogram and we will hopefully get some nice distribution which we can estimate $x_0$ with. the distance between our original gues for $x_0$ and this estimate yields a bias, which we can then add back into the model, and use this to generate a thousand more estimates, and if we plot them on a histogram as before we will see that our bias does get better, but is not completely removed by this process. The reason for this is that our bias in general depends on $x_0$. We can then add in that bias and iterate this process some amount of times until we see our guess for $x_0$ sitting in the proper place on the distribution.

We may generalize the above process. Suppose we have data generated by a shifted function $U = f(x-x_0) + \eta$ plus some random noise and we want to estimate that shift. We develop an estimator $A$ that estimates said shift. In our case above, $A = $template$(U)$ was the template method. For each value of $x_0$ we can run our model to generate data and use the estimator to try to figure out $x_0$. We will get a distribution of guesses for that $x_0$ and if we take the median of the distribution we will have corrected for some amount of bias. Plotting the median of the distribution generated in this way for each value of $x_0$ will yield what is called a "calibration curve". If our estimator is unbiased, this curve will be the identity $A = id(x)$. If it is not, then we can use this curve to correct the bias in our estimator.

![](figures/calibration_curve.png)

### Kernel Density Estimation

Suppose we've collected some samples $\{x_i\}_{i=1}^n$ and want to try to derive a probability distribution from them. What we can do is take a bump function $k(x)$ centered on the origin whose area under the curve is 1. We can center one of these functions on each of our data points and horizontally scale it by a factor of $h$ by $k_{x_i}^h(x) := \frac{1}{h}k(\frac{x-x_i}{h})$, wherein we divide by $h$ to preserve the area under the curve. Note that if $h$ is small then the function becomes more narrow and rapidly tends towards zero away from the point on which it is centered.

We then take as our distribution:

$$P_h(x) = \frac{1}{n}\sum_{i=1}^n k_{x_i}^h(x)$$

$h$ is then a parameter that adjusts the distribution. If $h$ is too small the resulting function will look like a sum of Dirac delta functions and if $h$ is too big very little detail will show. Often $h \approx \left(\frac{1}{n}\right)^{\frac{1}{5}}$ is chosen. However, adaptive scaling may be used to put more weight near clumps of points.

![](figures/kernel_density_estimation.png)

### Classifying objects

Suppose that we have some input space $X$ of objects that we would like to classify. Often $X \subseteq \mathbb{R}^d$ consists of feature vectors that describe some object with $d$ real numbers, and a classifying function will then be a map $f: X \to \{0,1\}$. We will assume that every $x \in X$ has correct classification, either 0 or 1.

Consider the special case of $X = [a,b] \subset \mathbb{R}$ and suppose we take $m$ samples from $X$ to get a collection of points $\{x_n, n = 1,..,m\}$ or equivalently partition $[a,b]$ into $m$ subintervals. Then there are $2^m$ possible binary functions that we could define with our samples as its domain or equivalently as histograms whose buckets are our subintervals. For now we will take the latter view, and call this collection of candidate histograms $\mathcal{F}$. We would like to select based on this data a classifying function $\hat{f}$ that will best match not only our sample but the whole interval $X$ as well.

Analogusly, perhaps we have a collection of pictures and we would like to determine if each picture has a duck in it or not. This is our domain $X$. So we take out $m$ pictures and classify each by hand as having a duck in them or not, and then from this we construct a function that can hopefully do a good job at classifying the rest of the pictures for us.

Since we have access to the true classification of our samples, we know that any good function should at least correctly classify them.

Let $f$ be the true classification of each point in $X$. Then we define the risk $R$ of a classifier $\hat{f}_n$ to be:

$$R(\hat{f}_n) = \int_X |\hat{f}_n-f|d\rho$$

And with $\hat{f}_n$ and $f$ being probability distributions, the risk is the likelihood of making an error given a random point on the interval. The hat notation and subscript $n$ is to make the dependence on our sample explicit.

Ideally we would like to find a function that minimizies the risk, $\min_{f \in \mathcal{F}} R(f)$ and in fact with our setup there exists a classifier with zero probability of error - our truth function. Define now the empirical risk by:

$$\hat{R}_n(f) = \frac{1}{n}\sum_{i}|f(x_i)-y_i|$$

Which calculates our error with respect to training data. We are then interested in:

$$P(R(\hat{f}_n) > \varepsilon)$$

Which is the probability of a classifier having a risk of more than $\varepsilon$. Essentially this may be computed by taking all of the possible classifying functions, putting all of those with a risk more than $\varepsilon$ into a big pile and then calculating which fraction that represents. Note however that this depends on the samples chosen. If $\hat{f}_n = \argmin_{f \in \mathcal{F}}\hat{R}_n(f)$ we can assert the following bound:

$$P(R(\hat{f}_n) > \varepsilon) \leq |\mathcal{F}| e^{-m\varepsilon}$$

Which for a fixed sample we can make as small as we like by taking more and more samples or dividing into smaller and smaller bins. 
