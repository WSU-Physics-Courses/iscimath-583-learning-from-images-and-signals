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
find . -type d -name ".ipynb_checkoints" -exec rm -rf {} +
find . -name "*.md" -delete
rsync -avz . smcwsu:iSciMath583/Assignments
```

I would like to do this on the student's accounts, but it needs internet access.

```bash
mkdir ~/.repositories
cd ~/.repositories && git clone https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals.git
```
