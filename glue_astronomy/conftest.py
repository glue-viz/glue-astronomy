def pytest_configure(config):
    from glue_astronomy.translators import (setup_ccddata,
                                            setup_regions,
                                            setup_spectral_cube,
                                            setup_spectrum1d,
                                            setup_trace)

    setup_ccddata()
    setup_regions()
    setup_spectral_cube()
    setup_spectrum1d()
    setup_trace()
