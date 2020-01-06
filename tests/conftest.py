import pytest

# pytest configuration file
# the configuration allow 3 kind of tests:
#   * unmarked tests run on all pytest execution
#   * tests with large datasets\long testing time are marked as "slow" and have to be run with pytest run --runslow
#   * tests with inconclusive result are marked as "inconclusive" have to be run with pytest run --runinconclusive
#   * tests can be both slow and inconclusive and have to be run with pytest run --runslow --runinconclusive

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runinconclusive", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(items, config):
    if not config.getoption("--runslow"):    
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runinconclusive"):    
        skip_slow = pytest.mark.skip(reason="need --runinconclusive option to run")
        for item in items:
            if "inconclusive" in item.keywords:
                item.add_marker(skip_slow)

