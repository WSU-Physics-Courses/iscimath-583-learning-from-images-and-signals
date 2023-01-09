Instructor Notes
================

I use the following on my laptop to sync the assignments:

```bash
cd ~/current/courses/iSciMath583/Assignments
ASSIGNMENT="../Course/Docs/Assignments/Denoising"
ls "${ASSIGNMENT}"
jupytext --sync ${ASSIGNMENT}/*.md
cp -Hr "${ASSIGNMENT}" .
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
find . -name "*.md" -delete
rsync -avz . smcwsu:iSciMath583/Assignments
```

I would like to do this on the student's accounts, but it needs internet access.

```bash
mkdir ~/.repositories
cd ~/.repositories && git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git
```


On the WSU Course project, I install the course repo. so it can be pushed to students
(using HTTP so that students can clone without authenticating).

```bash
ssh cc_wsu
mkdir -p ~/.repositories
cd ~/.repositories
git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git
git clone https://github.com/mforbes/mmf-setup-fork.git

```

I then add this as a handout, but edit the `.course` file by hand to insert this into
`~/.repositories/iscimath-583-learning-from-images-and-signals` on their projects. We
can then push updates, or refer to these as needed.
