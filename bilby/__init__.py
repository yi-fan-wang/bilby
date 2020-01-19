"""
Bilby
=====

Bilby: a user-friendly Bayesian inference library.

The aim of bilby is to provide user friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code, and many examples are hosted at https://git.ligo.org/lscsoft/bilby.
For installation instructions see
https://lscsoft.docs.ligo.org/bilby/installation.html.

"""


from __future__ import absolute_import
import sys

from . import core, gw, hyper

from .core import utils, likelihood, prior, result, sampler
from .core.sampler import run_sampler
from .core.likelihood import Likelihood

__version__ = utils.get_version_information()


if sys.version_info < (3,):
    raise ImportError(
"""You are running bilby 0.6.4 on Python 2

Bilby 0.6.4 and above are no longer compatible with Python 2, and you still
ended up with this version installed. That's unfortunate; sorry about that.
It should not have happened. Make sure you have pip >= 9.0 to avoid this kind
of issue, as well as setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Your choices:

- Upgrade to Python 3.

- Install an older version of bilby:

 $ pip install 'bilby<0.6.4'

""")
