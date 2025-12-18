#!/usr/bin/env python3
"""Test if aggregate_semantics can be imported"""
import sys
sys.path.insert(0, '.')

try:
    from agent5.bottom_up_aggregator import aggregate_semantics
    print("SUCCESS: aggregate_semantics imported!")
    print("Function: {}".format(aggregate_semantics.__name__))
    print("Module: {}".format(aggregate_semantics.__module__))
except ImportError as e:
    print("IMPORT ERROR: {}".format(e))
    import traceback
    traceback.print_exc()
except Exception as e:
    print("ERROR: {}".format(e))
    import traceback
    traceback.print_exc()

