Math 583 - Learning from Signals
================================
[![Documentation Status](https://readthedocs.org/projects/iscimath-583-learning-from-signals/badge/?version=latest)](https://iscimath-583-learning-from-signals.io/en/latest/?badge=latest)
[![gitlab pipeline status](https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/badges/main/pipeline.svg)](https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/-/commits/main)
[![gitlab coverage report](https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/badges/main/coverage.svg)](https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/-/commits/main)

This is the main project for the [WSU Mathematics] course
**Math 583: iSciMath - Learning from Images and Signals: Theory and Computation** first offered in [Spring 2023](https://schedules.wsu.edu/List/Pullman/20231/Math/583/01).

Physics has a successful track record of providing effective solutions to complex
problems outside its specific domain. This course will focus on using efficient
numerical techniques inspired by physics to solve challenging problems in a wide variety
of applications.  Techniques will be chosen from physics applications, but also applied
to problems outside of the physics domain including economics, biology, sociology, etc.
Students will be introduced to powerful numerical toolkits based on the
[SciPy](https://www.scipy.org/) and [NumFocus](https://numfocus.org) ecosystem. Using
the [CoCalc](https://cocalc.com/) platform will enable rapid development and prototyping
with an explicit path to stable, tested, and performant codes capable of supporting
research, or industry applications.

## TL;DR

Clone this repository somewhere with [Anaconda Project] and [Conda] installed, then run

```bash
make init
```

This will generate a [Conda] environment you can use to work with the project.  Once
this is done you can make the documentation, tests, etc. with commands like:

```bash
make test
make html
make doc-server
...
```

The latter will host the documentation on https://localhost:8000 and auto-update when you
make changes.

These will use `anaconda-project run` to perform the various actions.  If you want to
manually interact with the environment, then you can run:

```bash
anaconda-project run shell
```

In this shell, you can directly run `pytest` for example.

## Developer Notes

This file can be included in the documentation by adding the following to
`Docs/index.md`:

````markdown
```{include} ../README.md
```
````

See `Docs/index.md` for more details.



To use this repository:

1. *(Optional)* 
   * Create accounts on [CoCalc] and [GitLab], a project on [CoCalc], and a
   repo on [GitLab].  Send your [GitLab] account name to your instructor.
   * Create [Create SSH
   keys](https://doc.cocalc.com/project-settings.html#ssh-keys), and add them [to your
   CoCalc account](https://doc.cocalc.com/account/ssh.html) and [to your GitLab
   account](https://docs.gitlab.com/ee/ssh/).
   * SSH into your [CoCalc] project for the remaining steps.
2. Clone this repo and initialize it:

   ```bash
   git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git math-583-learning-from-signals
   cd math-583-learning-from-signals
   make init
   ```
   
   This will create a [Conda] environment you can activate `conda activate envs/math-583`,
   and a Jupyter kernel called `math-583` that you can select from notebooks.
   
3. *(Optional)* 
   * Add your [GitLab] repo as a remote, and push.
   * Create a [GitHub] mirror with automatic push from your [GitLab] repo so you can
     take advantage of their CI.

4. *(Optional)*
   * Add an appropriate Git Username etc. by defining `LC_GIT_USERNAME` etc. in your
     project or send these in your `~/.ssh/config` file.  In your project you can add
     something like this in your project **Settings > Custom environmental variables**
     section: 
     
     ```json
     {
       "LC_GIT_USERNAME": "Your Full Name", 
       "LC_GIT_USEREMAIL": "your.name@example.com",
       "LC_HG_USERNAME": "Your Full Name <your.name@example.com>",
       "LC_EDITOR": "vi",
     }
     ```

Overview
--------

One of the things I would like to do is to present each technique from at least two
perspectives:

From scratch
: I feel it is extremely valuable to understand what is at the core
  of an algorithm, and to be able to quickly implement a simplistic "brute force"
  version.  These have the following advantage:
  * Predictable convergence properties.  While often not as accurate or fast as more
    specialized or adaptive routines, simple brute-force versions of an algorithm can
    often be understood completely in terms of convergence properties and/or stability
    with respect to round-off errors etc.  I highly recommend always checking your results
    with a simple brute-force approach to make sure you are not making any mistakes:
  * Occasionally one will need to implement an algorithm from scratch on a specialize
    platform.  For example, to efficiently port code to a GPUs, one must try to remove
    conditionals (`if` statements).  This makes high-precision adaptive routines very
    hard to program, and one  can often gets better performance from a more
    straight-forward brute-force approach. 
  * Related: sometimes high precision is not needed.  In these cases, a simple
    low-accuracy but highly optimized brute-force approach might be fastest.  *(This is
    common in video graphics for example where results need only be accurate to the
    pixel level.)*
  * Having a basic understanding what is happening under "under the hood" of library
    routines can help when those routines fail. 

As fast as possible
: When you need accurate results for research, however, it is also very useful to be
  familiar with library techniques so you can quickly get high-precision results. 
  * If the appropriate technique is implement in a well-tested library, then one can
    often use it in a few lines of code and implement it in a few minutes of time --
    most of which is spent understanding the arguments.  This allows you to get results
    quickly. 
  * These techniques tend to be adaptive, spending more time where the problem is hard
    to achieve desired tolerance objectives.  If you need performance, you can reduce
    the tolerance. 
   * Unfortunately, adaptive routines can skip over the hard stuff, giving incorrect
     results that may seem reasonable at first.  There is no substitute for
     understanding your problem. 
  * It can be hard to understand exactly what black-box library routines are doing, and
    hence hard to understand their convergence properties.  It is essential to check your
    results with different techniques, or, at a minimum, with different resolutions. 
      > **Definition 16 (IDIOT)** *Anyone who publishes a calculation without checking it
      > against an identical computation with smaller $N$ OR without evaluating the 
      > residual of the pseudospectral approximation via finite differences is an IDIOT.*
      *(J. P. Boyd: Chebyshev and Fourier Spectral Methods {cite:p}`Boyd:1989`)* 

## Software Carpentry

Another objective of the course is to provide students with good software carpentry
skills.  This includes using version control, documenting code, testing, code-coverage,
and continuous integration (CI).  In particular, students are expected to fork this
repository and maintain their own [GitLab] repository with fully tested code.
Assignments will be distributed in the form of tests which the students must provide
functions which pass these tests.

* Students will be expected to maintain their code under version control.
* Code must be tested with unit tests providing at least 85% code coverage.
* Code must meet certain quality metrics, including documenting behaviour,
  inputs/outputs, specifying interfaces etc.

As part of the course, I will provide a detailed explanation of how to use tools like
[pytest](https://docs.pytest.org), [Coverage.py](https://coverage.readthedocs.io),
[Flake8](https://flake8.pycqa.org) and new tools like [LGTM](https://lgtm.com/) that
provide security analyses of code for Python-based projects to satisfying all of these
objectives.  With continuous integration techniques, these tests can be run whenever
code is committed, helping maintain functioning, well-tested code.

This repository provides a skeleton satisfying these requirements, and demonstrating how
to write proper tests, use [GitLab]s continuous integration, and to generate
documentation for Python.  *(Students wishing to use other languages will need to learn
how to use similar tools on their own.)*

Justification
: * Thinking about how to test code can significantly help in understanding the
    techniques and the problems. A significant portion of the course will address this
    issue.  For example: How can one find non-trivial problems with analytic solutions
    for testing?  What if such problems cannot be found? How should the algorithms
    converge?  Is appropriate convergence being achieved? *(Answering this later
    question quantitatively can provide for very useful test cases.)*
  * These skills will definitely be of benefit to anyone looking later for a career in
    industry, but will also help in maintaining code in a research setting.
  * The repository of code developed for this course can serve as a future portfolio.
  * Working tests serve as a demonstration of how to use the code, thereby functioning
    in some sense as documentation examples that are checked.

### GitLab Fork

1. Create an account on [GitLab].
2. Fork the [Official Course Repository] (I suggest making this private since your grade
   is associated with the tests, but you are welcome to make it public whenever you are
   comfortable.)
3. Add your instructor `@mforbes` as a **Developer** for the project:

   * **Project Information > Members**
   
4. Clone this to your [CoCalc] project and/or your computer.  Do your work etc. and push
   your changes.
5. Trigger the CI pipeline if it was not triggered by your push.

   * **CI/CD > Pipelines > Run pipeline**

6. Add the badges *(I don't know how to automate this or store this in a file
   yet... could maybe use [the Badges API](https://docs.gitlab.com/ee/api/project_badges.html))*:

   * **Settings > General > Badges**

    The following list the required fields:
    
    ```
    Name
    Link
    Badge image URL
    ```
    
    ```
    Docs
    https://wsu-phys-581-fall-2021.readthedocs.io/en/latest/?badge=latest
    https://readthedocs.org/projects/wsu-phys-581-fall-2021/badge/?version=latest
    ```
    ```
    Pipeline
    https://gitlab.com/%{project_path}
    https://gitlab.com/%{project_path}/badges/%{default_branch}/pipeline.svg
    ```
    ```
    Tests
    https://gitlab.com/%{project_path}
    https://gitlab.com/%{project_path}/-/jobs/artifacts/%{default_branch}/raw/_artifacts/test-badge.svg?job=test
    ```
    ```
    Coverage
    https://gitlab.com/%{project_path}
    https://gitlab.com/%{project_path}/-/jobs/artifacts/%{default_branch}/raw/_artifacts/coverage-badge.svg?job=test
    ```
    ```
    Assignment-0
   https://gitlab.com/%{project_path}
    https://gitlab.com/%{project_path}/-/jobs/artifacts/%{default_branch}/raw/_artifacts/test-0-badge.svg?job=test-0
    ```
    ```
    Assignment-1
    https://gitlab.com/%{project_path}
    https://gitlab.com/%{project_path}/-/jobs/artifacts/%{default_branch}/raw/_artifacts/test-1-badge.svg?job=test-1
    ```

    etc.

### Optional: SSH Keys

Typing your password every time you want to pull or push quickly gets tiring.  A better
option is to use [SSH] to authenticate, connect, and to forward your agent so you don't
need to re-authenticate.  The basic ideas are explained in [connecting to CoCalc with
SSH](https://doc.cocalc.com/project-settings.html#ssh-keys).

### Optional: GitHub Mirror

You can create a mirror on [GitHub] of your [GitLab] project which is updated whenever
you commit to your `main` branch.  Maintaining a [GitHub] mirror like this allows you to
use the [GitHub CI] tools, which differ somewhat from those on [GitLab].  In particular,
[LGTM] integration is quite interesting.

3. *(Optional)* Create an account on [GitHub].


## References

* [Python Tutorial](https://docs.python.org/3/tutorial/): This is the definative tutorial for the python language.  If you have not read this and plan to use python, then you should.
* [NumPy Tutorial](https://numpy.org/numpy-tutorials/): Growing repository of tutorials for using NumPy.  Being able to "think" in terms of arrays *(vectorization)* can greatly simplify your understanding of algorithms, while simultaneously improving your code, both from a performance and a reliability standpoint.  Not every problem benefits from this approach, but many of those in physics do.  *(We should try to [contribute](https://github.com/numpy/numpy-tutorials) to these.)*
* [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/): Deals with issues about packaging, testing, etc.  I plan to follow this (with some modifications discussed in [Hypothes.is annotations](https://hypothes.is/groups/z7AoNvZ1/computing) to setup the coding framework.



## Maintainer Notes
Try to keep this upper-level project as clean as possible, matching the layout expected
for the students.  This will be turned into a skeleton at some point.


## Tools

### Cookie Cutter
* https://github.com/cookiecutter/cookiecutter
* https://cookiecutter-hypermodern-python.readthedocs.io/

### Anaconda Project

```bash
anaconda-project init
anaconda-project add-packages python=3.9 scipy matplotlib sphinx
anaconda-project add-packages conda-forge::uncertainties
anaconda-project add-packages conda-forge::sphinx-panels conda-forge::sphinx-book-theme conda-forge::myst-nb
anaconda-project add-packages --pip sphinxcontrib-zopeext sphinxcontrib-bibtex mmf-setup
```

To clean:

```bash
anaconda-project clean
```

## Repository Setup

Can use GitHub, GitLab, or Heptapod.  With automatic pushing to GitHub, one and run the
following CI's:
* LGTM.com

* One course repo.  Students clone their own fork and pull changes.  Assignments
  distributed as tests.
* How to grade?  Student's can keep projects private (but probably will not have access
  to badges.)  Run tests on Student's CoCalc servers or with CI?
  

## Best Practices

* Use Jupytext and version control the associated python files.  Only commit the full
  notebooks (with output) when you want to archive documentation.
  * Maybe do this on an "output" branch or something so the main repo does not get
    cluttered?


## Docs

To build the documents interactively:

```bash
make doc-server
```

This will run [`sphinx-autobuild`](https://github.com/executablebooks/sphinx-autobuild)
which will launch a webserver on http://127.0.0.1:8000 and rebuild the docs whenever you
save a change.

Here is the play-by-play for setting up the documentation.

```bash
cd Docs
sphinx-quickstart
wget https://brand.wsu.edu/wp-content/themes/brand/images/pages/logos/wsu-signature-vertical.svg -O _static/wsu-logo.svg 
cp -r ../envs/default/lib/python3.9/site-packages/sphinx_book_theme/_templates/* _templates
```

I then edited the `conf.py`

```bash
hg add local.bib _static/ _templates/
```

## CoCalc Setup


* [Purchase a license](https://cocalc.com/settings/licenses) with 2 projects to allow
  the course and [WSU Courses CoCalc project] and [Shared CoCalc Project] to run.  This
  approach requires the students to pay $14 for access four the term (4 months).  They
  can optionally use any license they already have instead.
   
  Optionally, one might opt to purchase a license for $n+2$ projects where $n$ is the
  number of students, if there is central funding available.  See [Course Upgrading
  Students](https://doc.cocalc.com/teaching-upgrade-course.html#course-upgrading-students)
  for more details.
  
* Next, [create a course](https://doc.cocalc.com/teaching-create-course.html).  I do
  this in my [WSU Courses CoCalc project]..



* Create a [Shared CoCalc Project] and activate the license for this project so that it
  can run.  I then add the SSH key to may `.ssh/config` files so I can quickly login.

* Clone the repos into the shared project and initialize the project.  Optional, but
  highly recommend -- use my [`mmf-setup`] project to provide some useful features

  ```bash
  ssh smc<project name>       # My alias in .ssh/config
  python3 -m pip install mmf_setup
  mmf_setup cocalc
  ```
  
  This provides some instructions on how to use the CoCalc configuration.  The most
  important is to forward your user agent and set your `hg` and `git` usernames:
  
  ```bash
  ~$ mmf_setup cocalc
  ...
  If you use version control, then to get the most of the configuration,
  please make sure that you set the following variables on your personal
  computer, and forward them when you ssh to the project:

      # ~/.bashrc or similar
      LC_HG_USERNAME=Your Full Name <your.email.address+hg@gmail.com>
      LC_GIT_USEREMAIL=your.email.address+git@gmail.com
      LC_GIT_USERNAME=Your Full Name

  To forward these, your SSH config file (~/.ssh/config) might look like:

      # ~/.ssh/config
      Host cc_math-583-learning-from-signals
        User 5111388d081149c49cb9223049d52da7
    
      Host cc_math-583-learning-from-signals
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  ```
  
  Logout and log back in so we have the forwarded credentials, and now clone the repos.
  
  ```bash
  git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git math-583-learning-from-signals
  cd math-583-learning-from-signals
  make
  ```
  
  The last step runs `git clone git@gitlab.com:wsu-courses/iscimath-583-learning-from-images-and-signals_resources.git _ext/Resources` which puts the resources folder in `_ext/Resources`.

* Create an environment:

  ```bash
  ssh cc_math-583-learning-from-signals
  cd math-583-learning-from-signals
  anaconda2021
  anaconda-project prepare
  conda activate envs/math-583
  python -m ipykernel install --user --name "math-583" --display-name "Python 3 (math-583)"
  ```

  This will create a Conda environment as specified in `anaconda-project.yml` in `envs/math-583`.


# Funding

<a href="https://www.nsf.gov"><img width="10%"
src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png" />
</a>
<br>

Some of the material presented here is based upon work supported by the National Science
Foundation under [Grant Number 2012190](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2012190). Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.


<!-- Links -->
[Anaconda Project]: <https://github.com/Anaconda-Platform/anaconda-project> "Anaconda Project"
[CoCalc]: <https://cocalc.com> "CoCalc: Collaborative Calculation and Data Science"
[Conda]: <https://docs.conda.io/en/latest/> "Conda: Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more."
[GitHub CI]: <https://docs.github.com/en/actions/guides/about-continuous-integration> "GitHub CI"
[GitHub]: <https://github.com> "GitHub"
[GitLab]: <https://gitlab.com> "GitLab"
[Git]: <https://git-scm.com> "Git"
[Heptapod]: <https://heptapod.net> "Heptapod: is a community driven effort to bring Mercurial SCM support to GitLab"
[Jupyter]: <https://jupyter.org> "Jupyter"
[Jupytext]: <https://jupytext.readthedocs.io> "Jupyter Notebooks as Markdown Documents, Julia, Python or R Scripts"
[LGTM]: <https://lgtm.com/> "Continuous security analysis: A code analysis platform for finding zero-days and preventing critical vulnerabilities"
[Mercurial]: <https://www.mercurial-scm.org> "Mercurial"
[Miniconda]: <https://docs.conda.io/en/latest/miniconda.html> "Miniconda is a free minimal installer for conda."
[MyST]: <https://myst-parser.readthedocs.io/en/latest/> "MyST - Markedly Structured Text"
[Read the Docs]: <https://readthedocs.org> "Read the Docs homepage"
[WSU Physics]: <https://physics.wsu.edu> "WSU Department of Physics and Astronomy"
[WSU Mathematics]: <https://www.math.wsu.edu/> "WSU Department of Mathematics and Statistics"
[`anaconda-project`]: <https://anaconda-project.readthedocs.io> "Anaconda Project: Tool for encapsulating, running, and reproducing data science projects."
[`anybadge`]: <https://github.com/jongracecox/anybadge> "Python project for generating badges for your projects"
[`conda-forge`]: <https://conda-forge.org/> "A community-led collection of recipes, build infrastructure and distributions for the conda package manager."
[`genbadge`]: <https://smarie.github.io/python-genbadge/> "Generate badges for tools that do not provide one."
[`mmf-setup`]: <https://pypi.org/project/mmf-setup/> "PyPI mmf-setup page"
[`pytest`]: <https://docs.pytest.org> "pytest: helps you write better programs"
[hg-git]: <https://hg-git.github.io> "The Hg-Git mercurial plugin"
[GitLab test coverage visualization]: <https://docs.gitlab.com/ee/user/project/merge_requests/test_coverage_visualization.html>



[Official Course Repository]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals> "Official Course Repository hosted on GitLab"
[Shared CoCalc Project]: <https://cocalc.com/5111388d-0811-49c4-9cb9-223049d52da7/> "Shared CoCalc Project"
[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>


[GitHub Mirror]: <https://github.com/WSU-Physics-Courses/iscimath-583-learning-from-images-and-signals> "GitHub mirror"
[GitLab public repo]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals> "GitLab public repository."
[Gitlab private resources repo]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals_resources> "Private resources repository."
[file an issue]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/-/issues> "Issues on the GitLab project."

<!-- End Links -->
