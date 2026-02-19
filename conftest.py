"""
conftest.py — placed in Self-Trainer/ (the rootdir).
Adds the project root to sys.path so that imports like
`from core.engine import ...` work on both Windows and Linux.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))