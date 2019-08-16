def pytest_configure(config):
    from glue_astronomy import setup
    setup()
