<!-- Math 583 - Learning from Signals
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
-->


<!-- Include ../README.md
     If you would like to use the contents of your top-level README.md file here, then
     you can literally include it here with the following:

```{include} ../README.md
``` 

    Note that this may will break `sphinx-autobuild` (`make doc-server`) which will not rebuild
    this index file when ../README.md changes.  See the note at the bottom of the file
    if you want to do this while using sphinx-autobuild.
--> 


Welcome to Math 583 - Learning from Signals!  This is the main documentation page for the
course.  For more class information, please see the {ref}`sec:sylabus`.


# TL;DR

* [Read the Docs][]: The official online documentation.  Might be slightly out of date
  during the course, but ultimately, this is where the final online documentation will
  be hosted.
* [Generated Documentation on CoCalc][]: This is the generated documentation on CoCalc.
  This is likely to be a bit more up-to-date than [Read the Docs][], but will disappear
  after the course (and is only accessible by class participants).
* [CoCalc Shared Project][]: All participants should have access to this project.  This
  is where the instructors will demonstrate [code in class][class notebooks], post
  handouts, [lecture notes][], etc.  Feel free to play with the material here, but don't
  significantly change or modify the material.  If you want to significantly change a
  notebook, please copy it to a folder with your name in the [Workspaces][] folder, or
  copy the notebook to your student project.
* [GitLab Course Project][]: Course repository for the official course documentation and
  source code.
  
[Generated Documentation on CoCalc]: <https://cocalc.com/5111388d-0811-49c4-9cb9-223049d52da7/raw/iSciMath583/Docs/_build/html/index.html>
[Read the Docs]: <https://iscimath-583-learning-from-signals.readthedocs.io/en/latest/>
[CoCalc Shared Project]: <https://cocalc.com/projects/5111388d-0811-49c4-9cb9-223049d52da7/>
[lecture notes]: <https://cocalc.com/projects/5111388d-0811-49c4-9cb9-223049d52da7/files/Lectures/>
[class notebooks]: <https://cocalc.com/projects/5111388d-0811-49c4-9cb9-223049d52da7/files/ClassNotebooks/>
[Workspaces]: <https://cocalc.com/projects/5111388d-0811-49c4-9cb9-223049d52da7/files/Workspaces/>
[GitLab Course Project]: <https://gitlab.com/wsu-courses/iscimath-583-learning-from-images-and-signals>

# Getting Started

You should have an email invitation from CoCalc inviting you to create an account and
join the course.  Once you create your account (please use the email you were invited
on, or contact an instructor), then under <https://cocalc.com/projects> you should see
the [CoCalc Shared Project][] and a corresponding private student project.

Open your private student project, and purchase a license.  There should be an option
for a $14 student license for the term: this is probably the most cost effective unless
you want to purchase a license for the year.  If you already have a license, you are
free to use that.

```{toctree}
---
maxdepth: 2
caption: "Contents:"
titlesonly:
hidden:
glob:
---
Syllabus
Code
Assignments
Notes/*


References
```

```{toctree}
---
maxdepth: 2
caption: "Miscellaneous:"
hidden:
---
Demonstration
CoCalc
ClassLog
../InstructorNotes

README.md <../README>
```

<!-- If you opt to literally include files like ../README.md and would like to be able
     to take advantage of `sphinx-autobuild` (`make doc-server`), then you must make
     sure that you pass the name of any of these files to `sphinx-autobuild` in the
     `Makefile` so that those files will be regenerated.  We do this already for
     `index.md` but leave this note in case you want to do this elsewhere.
     
     Alternatively, you can include them separately and view these directly when editing.
     We do not include this extra toc when we build on RTD or on CoCalc.  We do this
     using the `sphinx.ext.ifconfig extension`:
     
     https://www.sphinx-doc.org/en/master/usage/extensions/ifconfig.html

```{eval-rst}
.. ifconfig:: not on_rtd and not on_cocalc

   .. toctree::
      :maxdepth: 0
      :caption: Top-level Files:
      :titlesonly:
      :hidden:

      README.md <../README>
      InstructorNotes.md <../InstructorNotes>
```
-->
