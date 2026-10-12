"""Make the package and tests importable when running pytest from this dir.

pytest (prepend import mode) inserts the directory containing this conftest.py
onto sys.path, so ``import cross_model_kv`` and ``import tests.synthetic`` both
resolve without an editable install.
"""
