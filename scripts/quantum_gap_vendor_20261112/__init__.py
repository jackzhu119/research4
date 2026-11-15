"""Byte-preserved scientific helpers from two pinned research4 commits.

The upstream modules use absolute imports by their original basenames. Callers
must put this directory on sys.path and use those basenames consistently so IR
and publication calculations share one configured electronic-core module.
No upstream runner, budget loop, publisher, or calculation is invoked here.
"""
