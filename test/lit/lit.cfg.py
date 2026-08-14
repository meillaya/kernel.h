"""Lit configuration for the kernel.h test suite.

Two entry points are supported (see lit.site.cfg.py.in):

1. Build dir (canonical; used by ctest and the check-kernelh target): lit
   loads the CMake-generated <build>/test/lit/lit.site.cfg.py, which sets the
   tool paths and then loads this file.

2. Source dir (plain `lit test/lit`, also from a subdir such as
   `lit test/lit/passes` since lit searches parent directories): lit loads
   this file directly, and it discovers the generated site config in the
   pinned build dir <repo>/build (fixed by devenv scripts.build.exec, which
   always configures with `-B build`).
"""

import os

import lit.formats

# Guard against double execution: the source-dir entry point loads the site
# config, which in turn loads this file again (lit has no recursion guard).
if not getattr(config, "_kernelh_cfg_loaded", False):
    config._kernelh_cfg_loaded = True

    config.name = "kernel.h"
    config.test_format = lit.formats.ShTest(True)
    config.suffixes = [".mlir", ".test"]

    if not getattr(config, "kernelh_opt_bin", None):
        # Loaded directly from the source tree: pull the tool paths from the
        # CMake-generated site config in the pinned build dir. Lit sets
        # config.test_source_root only AFTER the config script runs, so
        # derive this file's directory from __file__ (lit injects it into
        # the config globals).
        _src_root = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.normpath(
            os.path.join(_src_root, os.pardir, os.pardir))
        _site_cfg = os.path.join(_repo_root, "build", "test", "lit",
                                 "lit.site.cfg.py")
        if not os.path.exists(_site_cfg):
            lit_config.fatal(
                "kernel.h lit: generated site config %r not found; run the "
                "devenv `build` command first (cmake configure creates it)."
                % _site_cfg)
        lit_config.load_config(config, _site_cfg)

    if not os.path.isfile(config.kernelh_opt_bin):
        lit_config.fatal(
            "kernel.h lit: kernelh-opt binary %r not found; run the devenv "
            "`build` command first." % config.kernelh_opt_bin)
    if not os.path.isfile(config.kernelh_bin):
        lit_config.fatal(
            "kernel.h lit: kernelh CLI binary %r not found; run the devenv "
            "`build` command first." % config.kernelh_bin)

    config.substitutions.append(("kernelh-opt", config.kernelh_opt_bin))
    # Boundary rule: "kernelh " (with trailing space) never matches inside
    # "kernelh-opt" - neither in RUN lines nor in the kernelh-opt binary path
    # substituted above. A bare "kernelh" rule would re-write both.
    config.substitutions.append(("kernelh ", config.kernelh_bin + " "))
    config.substitutions.append(("FileCheck", config.filecheck_bin))
