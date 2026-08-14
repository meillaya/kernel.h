# latexmk configuration for the kernel.h LaTeX technical report.
# `latexmk -pdf docs/report.tex` (run from the repo root, as documented in
# .omo/evidence/task-39-latex-kernelh.txt) must write report.pdf + aux files
# into docs/ — latexmk would otherwise drop them in the current directory.
$out_dir = 'docs';
