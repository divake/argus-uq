"""argus_uq.models.closed_source - vendor-isolated agents.

Each vendor module (openai.py, anthropic.py, google.py) imports only from
this subpackage's `base` module and Python stdlib + the vendor's SDK. No
cross-vendor imports. The CI test
``tests/test_vendor_isolation.py`` enforces this rule.
"""
