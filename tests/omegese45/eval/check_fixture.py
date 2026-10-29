#!/usr/bin/env python3
"""Test-only invocation: substitute public test keys before checking synthetic cases.

Production check-demo has no override mechanism. This test shim loads its real
entry point (including mutated copies), replaces only the key constants with
public test keys, and runs main. Never use this shim for real demo receipts.
"""
import importlib.machinery
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(sys.argv[1]))
loader = importlib.machinery.SourceFileLoader("fixture_checker", sys.argv[1])
spec = importlib.util.spec_from_loader(loader.name, loader)
checker = importlib.util.module_from_spec(spec)
loader.exec_module(checker)
import mkcase
checker.EXPECTED_KEYS = {k: mkcase.pub(v) for k, v in mkcase.TEST_KEYS.items()}
sys.exit(checker.main([sys.argv[1], sys.argv[2]]))
