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

Instructors: the information presented here at the start of the course documentation is
contained in the `Docs/index.md` file, which you should edit to provide an overview of
the course.

One reasonable option might be to replace this by a literal include of the top-level
`README.md` file with the following code:

````markdown
```{include} ../README.md
``` 
````

```{toctree}
---
maxdepth: 2
caption: "Contents:"
titlesonly:
hidden:
glob:
---
Syllabus
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
