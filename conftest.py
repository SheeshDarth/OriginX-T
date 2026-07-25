"""Pytest rootdir marker.

Its mere presence makes pytest add the repository root to ``sys.path`` so
tests can ``import src.ingestion`` without any install step.
"""
