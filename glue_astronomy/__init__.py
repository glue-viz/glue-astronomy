from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


def setup():
    from glue_astronomy.translators import spectrum1d
