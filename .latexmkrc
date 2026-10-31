# latexmk configuration for the kernel.h LaTeX technical report.
# `latexmk docs/report.tex` selects LuaLaTeX (tikz-flora needs it) and writes
# report.pdf plus aux files into docs/. The TikZ Flora package is
# vendored under docs/flora; TEXINPUTS points there.
$pdf_mode = 4;
$out_dir = 'docs';
$ENV{'TEXINPUTS'} = 'docs/flora:' . ($ENV{'TEXINPUTS'} || '');
