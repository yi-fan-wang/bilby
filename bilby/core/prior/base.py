from importlib import import_module
import json
import os
import re

import numpy as np
import scipy.stats
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from bilby.core.utils import infer_args_from_method, BilbyJsonEncoder, decode_bilby_json, logger


class Prior(object):
    _default_latex_labels = {}

    def __init__(self, name=None, latex_label=None, unit=None, minimum=-np.inf,
                 maximum=np.inf, check_range_nonzero=True, boundary=None):
        """ Implements a Prior object

        Parameters
        ----------
        name: str, optional
            Name associated with prior.
        latex_label: str, optional
            Latex label associated with prior, used for plotting.
        unit: str, optional
            If given, a Latex string describing the units of the parameter.
        minimum: float, optional
            Minimum of the domain, default=-np.inf
        maximum: float, optional
            Maximum of the domain, default=np.inf
        check_range_nonzero: boolean, optional
            If True, checks that the prior range is non-zero
        boundary: str, optional
            The boundary condition of the prior, can be 'periodic', 'reflective'
            Currently implemented in cpnest, dynesty and pymultinest.
        """
        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                "maximum {} <= minimum {} for {} prior on {}".format(
                    maximum, minimum, type(self).__name__, name
                )
            )
        self.name = name
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.check_range_nonzero = check_range_nonzero
        self.least_recently_sampled = None
        self.boundary = boundary
        self._is_fixed = False

    def __call__(self):
        """Overrides the __call__ special method. Calls the sample method.

        Returns
        -------
        float: The return value of the sample method.
        """
        return self.sample()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if type(self.__dict__[key]) is np.ndarray:
                if not np.array_equal(self.__dict__[key], other.__dict__[key]):
                    return False
            elif isinstance(self.__dict__[key], type(scipy.stats.beta(1., 1.))):
                continue
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True

    def sample(self, size=None):
        """Draw a sample from the prior

        Parameters
        ----------
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        -------
        float: A random number between 0 and 1, rescaled to match the distribution of this Prior

        """
        self.least_recently_sampled = self.rescale(np.random.uniform(0, 1, size))
        return self.least_recently_sampled

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This should be overwritten by each subclass.

        Parameters
        ----------
        val: Union[float, int, array_like]
            A random number between 0 and 1

        Returns
        -------
        None

        """
        return None

    def prob(self, val):
        """Return the prior probability of val, this should be overwritten

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        np.nan

        """
        return np.nan

    def cdf(self, val):
        """ Generic method to calculate CDF, can be overwritten in subclass """
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with"
                "infinite support")
        x = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(x)
        cdf = cumtrapz(pdf, x, initial=0)
        interp = interp1d(x, cdf, assume_sorted=True, bounds_error=False,
                          fill_value=(0, 1))
        return interp(val)

    def ln_prob(self, val):
        """Return the prior ln probability of val, this should be overwritten

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        np.nan

        """
        return np.log(self.prob(val))

    def is_in_prior_range(self, val):
        """Returns True if val is in the prior boundaries, zero otherwise

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        np.nan

        """
        return (val >= self.minimum) & (val <= self.maximum)

    @staticmethod
    def test_valid_for_rescaling(val):
        """Test if 0 < val < 1

        Parameters
        ----------
        val: Union[float, int, array_like]

        Raises
        -------
        ValueError: If val is not between 0 and 1
        """
        valarray = np.atleast_1d(val)
        tests = (valarray < 0) + (valarray > 1)
        if np.any(tests):
            raise ValueError("Number to be rescaled should be in [0, 1]")

    def __repr__(self):
        """Overrides the special method __repr__.

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        -------
        str: A string representation of this instance

        """
        prior_name = self.__class__.__name__
        instantiation_dict = self.get_instantiation_dict()
        args = ', '.join(['{}={}'.format(key, repr(instantiation_dict[key]))
                          for key in instantiation_dict])
        return "{}({})".format(prior_name, args)

    @property
    def _repr_dict(self):
        """
        Get a dictionary containing the arguments needed to reproduce this object.
        """
        property_names = {p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)}
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = self.__dict__.copy()
        for key in property_names.intersection(subclass_args):
            dict_with_properties[key] = getattr(self, key)
        return {key: dict_with_properties[key] for key in subclass_args}

    @property
    def is_fixed(self):
        """
        Returns True if the prior is fixed and should not be used in the sampler. Does this by checking if this instance
        is an instance of DeltaFunction.


        Returns
        -------
        bool: Whether it's fixed or not!

        """
        return self._is_fixed

    @property
    def latex_label(self):
        """Latex label that can be used for plots.

        Draws from a set of default labels if no label is given

        Returns
        -------
        str: A latex representation for this prior

        """
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):
        """ If a unit is specified, returns a string of the latex label and unit """
        if self.unit is not None:
            return "{} [{}]".format(self.latex_label, self.unit)
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    def get_instantiation_dict(self):
        subclass_args = infer_args_from_method(self.__init__)
        property_names = [p for p in dir(self.__class__)
                          if isinstance(getattr(self.__class__, p), property)]
        dict_with_properties = self.__dict__.copy()
        for key in property_names:
            dict_with_properties[key] = getattr(self, key)
        instantiation_dict = dict()
        for key in subclass_args:
            instantiation_dict[key] = dict_with_properties[key]
        return instantiation_dict

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary not in ['periodic', 'reflective', None]:
            raise ValueError('{} is not a valid setting for prior boundaries'.format(boundary))
        self._boundary = boundary

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    def to_json(self):
        return json.dumps(self, cls=BilbyJsonEncoder)

    @classmethod
    def from_json(cls, dct):
        return decode_bilby_json(dct)

    @classmethod
    def from_repr(cls, string):
        """Generate the prior from it's __repr__"""
        return cls._from_repr(string)

    @classmethod
    def _from_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)

        string = string.replace(' ', '')
        kwargs = cls._split_repr(string)
        for key in kwargs:
            val = kwargs[key]
            if key not in subclass_args and not hasattr(cls, "reference_params"):
                raise AttributeError('Unknown argument {} for class {}'.format(
                    key, cls.__name__))
            else:
                kwargs[key] = cls._parse_argument_string(val)
            if key in ["condition_func", "conversion_function"] and isinstance(kwargs[key], str):
                if "." in kwargs[key]:
                    module = '.'.join(kwargs[key].split('.')[:-1])
                    name = kwargs[key].split('.')[-1]
                else:
                    module = __name__
                    name = kwargs[key]
                kwargs[key] = getattr(import_module(module), name)
        return cls(**kwargs)

    @classmethod
    def _split_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)
        args = string.split(',')
        remove = list()
        for ii, key in enumerate(args):
            if '(' in key:
                jj = ii
                while ')' not in args[jj]:
                    jj += 1
                    args[ii] = ','.join([args[ii], args[jj]]).strip()
                    remove.append(jj)
        remove.reverse()
        for ii in remove:
            del args[ii]
        kwargs = dict()
        for ii, arg in enumerate(args):
            if '=' not in arg:
                logger.debug(
                    'Reading priors with non-keyword arguments is dangerous!')
                key = subclass_args[ii]
                val = arg
            else:
                split_arg = arg.split('=')
                key = split_arg[0]
                val = '='.join(split_arg[1:])
            kwargs[key] = val
        return kwargs

    @classmethod
    def _parse_argument_string(cls, val):
        """
        Parse a string into the appropriate type for prior reading.

        Four tests are applied in the following order:

        - If the string is 'None':
            `None` is returned.
        - Else If the string is a raw string, e.g., r'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains ', e.g., 'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains an open parenthesis, (:
            The string is interpreted as a call to instantiate another prior
            class, Bilby will attempt to recursively construct that prior,
            e.g., Uniform(minimum=0, maximum=1), my.custom.PriorClass(**kwargs).
        - Else:
            Try to evaluate the string using `eval`. Only built-in functions
            and numpy methods can be used, e.g., np.pi / 2, 1.57.


        Parameters
        ----------
        val: str
            The string version of the agument

        Returns
        -------
        val: object
            The parsed version of the argument.

        Raises
        ------
        TypeError:
            If val cannot be parsed as described above.
        """
        if val == 'None':
            val = None
        elif re.sub(r'\'.*\'', '', val) in ['r', 'u']:
            val = val[2:-1]
        elif "'" in val:
            val = val.strip("'")
        elif '(' in val:
            other_cls = val.split('(')[0]
            vals = '('.join(val.split('(')[1:])[:-1]
            if "." in other_cls:
                module = '.'.join(other_cls.split('.')[:-1])
                other_cls = other_cls.split('.')[-1]
            else:
                module = __name__.replace('.' + os.path.basename(__file__).replace('.py', ''), '')
            other_cls = getattr(import_module(module), other_cls)
            val = other_cls.from_repr(vals)
        else:
            try:
                val = eval(val, dict(), dict(np=np))
            except NameError:
                raise TypeError(
                    "Cannot evaluate prior, "
                    "failed to parse argument {}".format(val)
                )
        return val


class Constraint(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None):
        super(Constraint, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                         latex_label=latex_label, unit=unit)
        self._is_fixed = True

    def prob(self, val):
        return (val > self.minimum) & (val < self.maximum)

    def ln_prob(self, val):
        return np.log((val > self.minimum) & (val < self.maximum))


class PriorException(Exception):
    """ General base class for all prior exceptions """
