"""Live smoke tests - real model loading and real API calls.

Gated behind ``@pytest.mark.slow`` and ``@pytest.mark.needs_gpu`` /
``@pytest.mark.needs_api``. Skipped on CI; run manually on a GPU machine.

Invocation:
    # All open-source live smoke tests (~50 minutes on one 48 GB GPU):
    pytest tests/live/test_smoke_open_source.py -m "slow and needs_gpu" -v

    # All closed-source live smoke tests (~$3-15 in API costs):
    pytest tests/live/test_smoke_closed_source.py -m "slow and needs_api" -v

    # Both:
    pytest tests/live/ -m slow -v
"""
