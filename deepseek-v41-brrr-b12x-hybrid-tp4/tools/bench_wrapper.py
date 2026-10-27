#!/usr/bin/env python3
"""Run the pinned benchmark, changing only its interactive self-update callback."""
import importlib.util
import sys

from config import PINS
from prepare_bench import verify


def main():
    source=verify()
    spec=importlib.util.spec_from_file_location('recipe_pinned_llm_bench',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if module.VERSION!=PINS['benchmark_version']:raise RuntimeError('Unexpected benchmark version')
    module.check_for_update=lambda console:False
    sys.argv[0]=str(source)
    try:module.main()
    finally:module._restore_terminal()


if __name__=='__main__':main()
