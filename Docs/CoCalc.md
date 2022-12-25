# [CoCalc]

[CoCalc] provides an online platform for performing computations without the need to
install any software on your computer.  It allows you to immediately start exploring in
a variety of languages, with several unique features.

## TL;DR

One of the main benefits of [CoCalc] is that you can just fire it up and get to work.
Later in this document we will discuss how to setup a reproducible programming
environment, but for now, you can do the following to just get started.

1. Create an account and project on [CoCalc].
   *If you are taking this as a course, then you should use the project you are invited
   to join as part of the course.*

2. [Purchase a license](https://cocalc.com/settings/licenses).  A [site
   license](https://cocalc.com/store/site-license) is required to access the internet
   from projects, e.g. to install software,and will provide for much better performance.
   While you can run your project without a license, this is really only appropriate for
   testing things.
   *Students will be prompted for a term license from the course, but may use their own
   license if they wish to purchase something more long-term.*

3. Open your project and start exploring.
   *Students should see two projects that are automatically created for the course:*
   
   * *A [shared CoCalc project] for the course, which is where everyone can
     interact with each other and collaborate on documents provided by the instructor.
     Remember -- this is shared with everyone, so be respectful and careful with your
     modifications. If you want to make extensive changes, it might be better to copy
     material to your private course project (see below).*
   * *A private course project where you will complete your homework. Your instructors
     have access to this by default as they need to collect your assignments for
     grading.  You have complete access to this project and can add or remove
     collaborators.  It will remain yours after the course concludes.*


## Features

The following features are particularly relevant:

Collaborative Editing
: [CoCalc] is one of the only platform I know of at the time that allows simultaneous editing
  of [Jupyter] notebooks.  This means that the instructors can directly connect to
  student's projects, seeing the exact same code, and interactively debugging it.  This
  is done with a custom implementation of the notebook server, with the downside that
  not all features of [Jupyter] notebooks are supported yet.  If you need to, you can
  launch a [Plain Jupyter Server and JupyterLab
  Server](https://doc.cocalc.com/jupyter.html#alternatives-plain-jupyter-server-and-jupyterlab-server)
  to regain all functionality (but will lose the collaborative editing ability while you
  do so).

[Extensive Software Preinstalled](https://cocalc.com/doc/software.htm)
: [CoCalc] comes with a large amount of [useful
  software](https://cocalc.com/doc/software.html), including a rather complete
  `Anaconda` environment.  This allows you to immediately start working.  Simply create
  a new [Jupyter] notebook, choose the `Anaconda2021` kernel, and start coding with the
  full SciPy software stack at your disposal. (Once you get things working, I
  **strongly** advocate migrating your code to a well tested repository like the one
  described here, but don't let this stop you from exploring.)
  
[VS Code] Editor:
: [CoCalc] now supports editing files in your browser with [VS Code].  While I
  personally use [Emacs], [VS Code] seems to be a very good tool for beginners.  I
  strongly recommend that you learn a good editor with powerful search and replace
  features, syntax highlighting and language support: it will ultimately save you lots
  of time.  *(Note: this is a fairly
  [new feature](https://github.com/sagemathinc/cocalc/issues/4479), however, and I have
  not explored it much, but it looks good.)*

I find a couple of other features important:

[Open Source](https://github.com/sagemathinc/cocalc)
: [CoCalc] itself is [open source](https://github.com/sagemathinc/cocalc) and can be
  installed from a [Docker image](https://doc.cocalc.com/docker-image.html).  This means
  that you can run [CoCalc] on your own hardware, with complete control of your data,
  even if they go out of business.  The make their profits by selling their service.  I
  completely support this type of business model which puts you in control of your data.

[Time Travel](https://doc.cocalc.com/time-travel.html)
: [CoCalc] implements an amazing backup system they call [Time
  Travel](https://cocalc.com/doc/software.htm) that allows you to roll back almost any
  file minutes, hours, days, weeks, or more.  I am blown away by how well this feature
  is implemented: it has saved me several times and along is worth the license costs.

Responsive Support
: The [CoCalc] company is small enough that they can still be responsive to feature and
  support requests.  When I have issues, the often make changes within an hour, and
  virtually never take more than a day.  This is in stark contrast to large companies
  where you submit a request to their community forums only to have it ignore for
  years.  Of course, the team being small means that they do not have the resources to
  implement everything, but can be motivated by money if you really need something
  done.  Nevertheless, they have always taken care of any core issues I have found
  promptly, and are really nice people too!

[Remote File Systems](https://doc.cocalc.com/project-settings.html#ssh-remote-files)
: You can [mount remote file systems with
  `sshfs`](https://doc.cocalc.com/project-settings.html#ssh-remote-files).  This allows
  you to use [CoCalc] as a tool to analyze off-site data (although performance will be
  slow because the data needs to be transferred over the network).

For a more completed exploration, look at the [list of features](https://cocalc.com/doc/).

## Setup for Software Development

While one of the main benefits of [CoCalc] is that you can just fire it up and get to
work, for the purposes of this course, establishing a reproducible computing environment
is important.  After exploring several tools, I have landed on [`anaconda-project`]
which allows you to manage a [Conda] environment in a somewhat reasonable way.

Using this effectively on `CoCalc` is a bit challenging out of the box because the
default `anaconda2021` environment they have setup has a `/ext/anaconda2021.11/.condarc`
file with a whole slew of channels -- so many in fact that even a simple `conda search
uncertainties` almost runs out of memory.  One option is to use `mamba` which can be
done by setting `CONDA_EXE=mamba`.

Another potential option is to use a custom miniconda environment, but even this takes
too much memory.  Until `anaconda-project` has a [way of ignored the
channels](https://github.com/Anaconda-Platform/anaconda-project/issues/336), it seems
like the best option is to simply install our own version of Miniconda and use this as a
base:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -qO miniconda.sh
echo "1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f  miniconda.sh" > miniconda.shasum
shasum -a 256 -c miniconda.shasum && bash miniconda.sh -b -p ~/.miniconda
rm miniconda.sh*
. ~/.miniconda/bin/activate
conda install anaconda-project
conda clean --all -y
du -sh ~/.miniconda   # 136M	/home/user/.miniconda
echo "export COCALC_MINICONDA=~/.miniconda" >> ~/.bashrc
```

These are complete instructions for students to get started working with this project
for use in the course and include managing a [GitLab] repository, with automated testing
etc.

## Instructions

Here are the general instructions.  Some steps may require additional information if you
are not familiar with the associated concepts (i.e. setting up SSH keys).

1. *(Optional)* Create an account and project on [CoCalc].  If you are taking the
   course, then you should use the project you are invited to join as part of the
   course.  The instructions will generally assume you are working in the [CoCalc]
   project, but things will probably work out-of-the-box on Linux machines or Mac OS
   X. (No guarantees with Windows or other platforms.)  If you encounter any problems,
   please [file an issue].
   
   If you plan to use [CoCalc], then either complete the remaining steps using an [Online
   Linux Terminal](https://cocalc.com/doc/terminal.html) in your project, or by
   [connecting to CoCalc with
   SSH](https://doc.cocalc.com/project-settings.html#ssh-keys).
   
   If you are running this on another platform, you must make sure that you have a
   [Conda] environment setup with `anaconda-project >= 0.8.4`.  You can do this easily
   by installing [Miniconda], then either updating the `base` environment:
   
   ```console
   (base) $ conda install anaconda-project
   (base) $ anaconda-project ...
   ```

   or creating a new environment with `anaconda-project`:
   
   ```console
   (base) $ conda create -n myenv anaconda-project
   (base) $ conda activate myenv
   (myenv) $ anaconda-project ...
   ```

   
2. Clone this repo and change directories to enter the project:

   ```bash
   git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git math-583-learning-from-signals
   cd math-583-learning-from-signals
   ```
   
   *Note: I actually prefer to use [Mercurial] and usually use the [hg-git] plugin, but
   don't recommend this unless you are very familiar with [Mercurial].*
   
   ```bash
   hg clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git math-583-learning-from-signals
   cd math-583-learning-from-signals
   ```
   
3. Use [`anaconda-project`] to provision the environment, setup the kernel, clone the
   resources, etc.  This can all be done with:

   ```bash
   make init
   ```
   
   Which will do the following:
   
   1. Clone the [GitLab private resources repo] into `_ext/Resources`.  Note: this is
      private and requires that your instructor grant you access.
   2. Run `anaconda-project prepare` which uses `conda` or `mamba` to create an
      environment in `envs/math-583` as specified in the `anaconda-project.yaml`
      file. *(This may take some time when you first run it, and consumes about 2GB of
      disk space.  You can clean up some space after by running `make clean`.)*
   3. Installs this environment as an [IPython
      kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
      called `math-583` for use with [Jupyter].  This is done by running
      `anaconda-project run init` which runs [`ipykernel
      install`](https://ipython.readthedocs.io/en/stable/install/kernel_install.html):
      see the `init` target in `anaconda-project.yaml` for details.
   4. If you are on [CoCalc] *(technically, if `ANACONDA2020` is defined)*, then `make
      init` will also install the `mmf-setup` package, update some files, create a
      `~/.bash_aliases` file, and insert a line to activate the environment when logging
      in.  This also then runs `make sync` which uses [Jupytext] to populate the
      notebooks so you can use them with [CoCalc].

4. *(Optional)* After installing, you might like to clean up the downloaded files.  This
   is especially important on [CoCalc] where disk space is at a premium:
   
   ```console
   conda clean --all -y
   ```
   

At this point you can start using the project, viewing the notebooks, running and
editing code, etc.  If you need additional packages, you should add them with
`anaconda-project`.  I recommend the following strategy. See if the package is available
from the default conda repos, and install from there if it is available.  If it is only
available from [`conda-forge`] then explicitly install it from there, otherwise use `pip`:
   
```bash
conda search --override-channels -c defaults sphinx
conda search --override-channels -c conda-forge uncertainties
anaconda-project add-packages sphinx
anaconda-project add-packages conda-forge::uncertainties  # Not available in defaults
anaconda-project add-packages --pip mmf-setup             # Only available through pip
```

Note: I am explicitly using `--override-channels`: this is crucial on CoCalc for now as
the default `/ext/anaconda2020.02/.condarc` file has so many channels that `conda` will
run out of memory.

There are a few more things you should do if you are registered in the course:

5. [Create SSH keys](https://doc.cocalc.com/project-settings.html#ssh-keys), and add
   them [to your CoCalc account](https://doc.cocalc.com/account/ssh.html) and [to your
   GitLab account](https://docs.gitlab.com/ee/ssh/).
   your project with SSH, forwarding your SSH agent.
4. Create a [GitLab] account and send the username to your instructor so that they can
   give you access to the [GitLab private resources repo].
5. Create a [GitLab] repository for this course, and add this as a remote so that you
   can push your work to it.  *You may make this project public or private as you prefer,
   but note that private projects may have more limited access to CI resources.  See
   [GitLab pricing](https://about.gitlab.com/pricing/) for details.*


## CoCalc Setup

There are a few more things that one should do if using the [CoCalc] platform.

### License

[Purchase a license](https://cocalc.com/settings/licenses).  A license is required to
access the internet from projects, for example, to clone from [GitLab].  Students will
be prompted for a term license from the course, but may use their own license.

### SSH Keys

SSH keys are needed for two tasks: connecting to [GitLab] or other external servers
**from** [CoCalc] to pull or push changes, and *(optional)* to directly connect **to**
[CoCalc] from your computer without relying on the browser interface.

To connect **from** [CoCalc], you need:

1. to be authenticated to a key on [CoCalc], and
2. to share the associated **public key** with the external resource (e.g. [GitLab]).

If you want to work with the web interface, then you need to generate a key on [CoCalc],
and then share the public key to [GitLab] etc.  Do this in a new `Linux terminal` (use
the New button on the top-left of the [CoCalc] interface):

```bash
# This will generated a private key in ~/.ssh/id_ed25519 and a
# public key in .ssh/id_ed25519.pub.  The ed25518 type of key is
# short and secure, but some older sites may need RSA or DSA keys
# which you can use by changing the argument to -t
ssh-keygen -t ed25519   # or -t rsa or -t dsa

# Copy the public key and use this on GitLab etc.
cat ~/.id_ed25519.pub
```

When you want to use this key, i.e. to clone or push to a private [GitLab] repo, you
will need to first authenticate.  This often happens automatically, but requires you
enter your password multiple times.  To avoid this, you can start a terminal (`bash`) with
[`ssh-agent`](https://en.wikipedia.org/wiki/Ssh-agent) and add then `ssh-add` the key.
The agent will then manage the key for you and you can push an pull multiple times *from
that terminal* without re-authenticating.

```bash
ssh-agent bash  # Runs a new bash shell with ssh-agent
ssh-add         # May need to specify the key if you have several

# No more passwords needed in this shell
git fetch ...
git push ...
etc.
```

To connect **to** [CoCalc] from your computer with SSH, you will need to create a
similar key *on your computer*, then copy the public version to CoCalc as described in
[connecting to CoCalc with SSH](https://doc.cocalc.com/project-settings.html#ssh-keys).
Also copy this public key to [GitLab] etc.  If you opt to do this, you can then forward
your authenticated credentials to [CoCalc] when you connect, and use those forwarded
credentials instead of running `ssh-agent` as above.  This can all be expressed in your
`~.ssh/config` file *on your computer*.  I recommend something like the following:

```
# ~/.ssh/config file on your computer
...
Host cc_math-583-learning-from-signals
  User 5111388d081149c49cb9223049d52da7
Host cc_*
  HostName ssh.cocalc.com
  ForwardAgent yes
  SetEnv LC_HG_USERNAME=Your Full Name <your.name@example.com>
  SetEnv LC_GIT_USERNAME=Your Full Name
  SetEnv LC_GIT_USEREMAIL=your.name@example.com
Host *
  AddressFamily inet
  # Force IPv4
  # https://www.electricmonk.nl/log/2014/09/24/
  #         ssh-port-forwarding-bind-cannot-assign-requested-address/
```

The first `Host cc_math-583-learning-from-signals` is an [SSH
alias](https://man.openbsd.org/ssh_config) that specifies the project-specific username.
Find the correct value under your [CoCalc] project's
[`Settings`](https://doc.cocalc.com/project-settings.html#ssh-keys) tab.  The second
entry is a wildcard that applies to every host that starts with `cc_`.  The last
one is a general setting that applies for all hosts.  Note that general settings must
come after the more specific aliases.

Alternatively, if you use the environmental variables `LC_HG_USERNAME` etc. *on your
computer*, then you can just send the current values:

```
# ~/.ssh/config file on your computer
...
Host cc_*
  HostName ssh.cocalc.com
  ForwardAgent yes
  SendEnv LC_HG_USERNAME=
  SendEnv LC_GIT_USERNAME
  SendEnv LC_GIT_USEREMAIL
...
```

Once this is done, I can connect directly to a terminal on [CoCalc] by:

1. Starting the project from the web if it is not running (I don't think there is a
   workaround for this step.  If the project is not running you will get an error like
   `f6432a...@ssh.cocalc.com: Permission denied (publickey).`.

2. Connect from a terminal *on your computer*:

   ```bash
   ssh cc_math-583-learning-from-signals
   ```

   Because of the `~/.ssh/config` file, this is equivalent to directly typing something
   like
   
   ```bash
   ssh -A -o SendEnv=LC_HG_USERNAME ... f6432a...@ssh.cocalc.com
   ```
   
   The `ForwardAgent yes` config (`-A` on the command line) will forward you your key to
   [CoCalc] which can then forward it to [GitLab] eschewing the need to enter your
   password.  In this case you should add the public key *on your computer* to [GitLab]
   etc.
   
   *Note: on most laptops, your login manager will authenticate you to your SSH key
   when you login.  If this does not happen, then you might need to first run
   `ssh-agent bash` and then `ssh-add` on your computer as we did above on CoCalc.
   If you find yourself needing to do this, search online to see if you can figure out
   how to have your login manager act as the SSH agent.  For example, `Keychain
   Access.app` does this on my Mac OS X, `KWallet` does this on linux with KDE, etc.* 
   
### Git Username

In order to commit to your version control, you need to specify your username and email
to [Git] or [Mercurial]. Typically this is done by adding appropriate entries in
`~/.gitconfig` or `~/.hgrc`:

```bash
# ~/.gitconfig file
[user]
        email = your.name@example.com
        name = Your Full Name
...
```

```bash
# ~/.hgrc file
[ui]
username = Your Full Name <your.name@example.com>
...
```

However, if you collaborate on [CoCalc], then this will not work well because all of
your collaborators will be committing as you!  Instead, we rely on the value of
`LC_GIT_USERNAME` etc. which each user can set when using SSH to access the project.

If you do not use SSH, you can set these directly in your project's **Settings > Custom
environmental variables** section:
     
```json
{
  "LC_GIT_USERNAME": "Your Full Name", 
  "LC_GIT_USEREMAIL": "your.name@example.com",
  "LC_HG_USERNAME": "Your Full Name <your.name@example.com>",
  "LC_EDITOR": "vi",
}
```

but this will have the same issue if you collaborate -- all committers who use the web
interface will commit with this username/email.  (Collaborators who use SSH to forward
`LC_*` values will override these.)  Restart your project for these to take effect.

:::{Note}
The reason for the `LC_` prefix is that SSH allows these variables to
be sent by default -- others will be blocked and we do not have root access on [CoCalc]
to change this behavior.  We must then set the more appropriate variables
`GIT_AUTHOR_NAME="${LC_GIT_USERNAME}"` etc. in `~/.bash_aliases`.

Specifically, `make init` does the following (and more):

```
# ~/.gitconfig
[user]
# These are NOT set here...
#   email =
#   name = 
...
```

```
# ~/.hgrc
[ui]
username = $LC_HG_USERNAME
...
```

```
# ~/.bash_aliases
...
export GIT_AUTHOR_NAME="${LC_GIT_USERNAME}"
export GIT_AUTHOR_EMAIL="${LC_GIT_USEREMAIL}"
export GIT_COMMITTER_NAME="${LC_GIT_USERNAME}"
export GIT_COMMITTER_EMAIL="${LC_GIT_USEREMAIL}"
...
```
:::

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

<!-- global markdown links -->
[CoCalc]: <https://cocalc.com> "CoCalc: Collaborative Calculation and Data Science"
[Conda]: <https://docs.conda.io/en/latest/> "Conda: Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more."
[Emacs]: <https://www.gnu.org/software/emacs/> "GNU Emacs: An extensible, customizable, free/libre text editor - and more."
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
[SSH]: <https://en.wikipedia.org/wiki/Secure_Shell> "SSH on Wikipedia"
[`anaconda-project`]: <https://anaconda-project.readthedocs.io> "Anaconda Project: Tool for encapsulating, running, and reproducing data science projects."
[`anybadge`]: <https://github.com/jongracecox/anybadge> "Python project for generating badges for your projects"
[`conda-forge`]: <https://conda-forge.org/> "A community-led collection of recipes, build infrastructure and distributions for the conda package manager."
[`genbadge`]: <https://smarie.github.io/python-genbadge/> "Generate badges for tools that do not provide one."
[`mmf-setup`]: <https://pypi.org/project/mmf-setup/> "PyPI mmf-setup page"
[`pytest`]: <https://docs.pytest.org> "pytest: helps you write better programs"
[hg-git]: <https://hg-git.github.io> "The Hg-Git mercurial plugin"
[WSU Physics]: <https://physics.wsu.edu> "WSU Physics Department"
[VS Code]: <https://code.visualstudio.com> "Visual Studio Code"




[Official Course Repository]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals> "Official Course Repository hosted on GitLab"
[Shared CoCalc Project]: <https://cocalc.com/5111388d-0811-49c4-9cb9-223049d52da7/> "Shared CoCalc Project"
[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>


[GitHub Mirror]: <https://github.com/WSU-Physics-Courses/iscimath-583-learning-from-images-and-signals> "GitHub mirror"
[GitLab public repo]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals> "GitLab public repository."
[Gitlab private resources repo]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals_resources> "Private resources repository."
[file an issue]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals/-/issues> "Issues on the GitLab project."
<!-- global markdown links end -->


