"""Candidate submissions for P01.

Putting the problem directory on sys.path here means a candidate module can
simply `from pbwt_core import pbwt_step` (and import numpy) without any path
boilerplate of its own. Candidates must not import the oracle.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
