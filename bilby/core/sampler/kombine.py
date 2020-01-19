from __future__ import absolute_import, print_function
from ..utils import logger, get_progress_bar
import numpy as np
import os
from .emcee import Emcee
from .base_sampler import SamplerError


class Kombine(Emcee):
    """bilby wrapper kombine (https://github.com/bfarr/kombine)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `kombine.Sampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nwalkers: int, (500)
        The number of walkers
    iterations: int, (100)
        The number of iterations
    auto_burnin: bool (False)
        Use `kombine`'s automatic burnin (at your own risk)
    nburn: int (None)
        If given, the fixed number of steps to discard as burn-in. These will
        be discarded from the total number of steps set by `nsteps` and
        therefore the value must be greater than `nsteps`. Else, nburn is
        estimated from the autocorrelation time
    burn_in_fraction: float, (0.25)
        The fraction of steps to discard as burn-in in the event that the
        autocorrelation time cannot be calculated
    burn_in_act: float (3.)
        The number of autocorrelation times to discard as burn-in


    """

    default_kwargs = dict(nwalkers=500, args=[], pool=None, transd=False,
                          lnpost0=None, blob0=None, iterations=500, storechain=True, processes=1, update_interval=None,
                          kde=None, kde_size=None, spaces=None, freeze_transd=False, test_steps=16, critical_pval=0.05,
                          max_steps=None, burnin_verbose=False)

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 pos0=None, nburn=None, burn_in_fraction=0.25, resume=True,
                 burn_in_act=3, autoburnin=False, **kwargs):
        super(Kombine, self).__init__(likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                                      use_ratio=use_ratio, plot=plot, skip_import_verification=skip_import_verification,
                                      pos0=pos0, nburn=nburn, burn_in_fraction=burn_in_fraction,
                                      burn_in_act=burn_in_act, resume=resume, **kwargs)

        if self.kwargs['nwalkers'] > self.kwargs['iterations']:
            raise ValueError("Kombine Sampler requires Iterations be > nWalkers")
        self.autoburnin = autoburnin

    def _check_version(self):
        # set prerelease to False to prevent checks for newer emcee versions in parent class
        self.prerelease = False

    def _translate_kwargs(self, kwargs):
        if 'nwalkers' not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nwalkers'] = kwargs.pop(equiv)
        if 'iterations' not in kwargs:
            if 'nsteps' in kwargs:
                kwargs['iterations'] = kwargs.pop('nsteps')
        # make sure processes kwarg is 1
        if 'processes' in kwargs:
            if kwargs['processes'] != 1:
                logger.warning("The 'processes' argument cannot be used for "
                               "parallelisation. This run will proceed "
                               "without parallelisation, but consider the use "
                               "of an appropriate Pool object passed to the "
                               "'pool' keyword.")
                kwargs['processes'] = 1

    @property
    def sampler_function_kwargs(self):
        keys = ['lnpost0', 'blob0', 'iterations', 'storechain', 'lnprop0', 'update_interval', 'kde',
                'kde_size', 'spaces', 'freeze_transd']
        function_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}
        function_kwargs['p0'] = self.pos0
        return function_kwargs

    @property
    def sampler_burnin_kwargs(self):
        extra_keys = ['test_steps', 'critical_pval', 'max_steps', 'burnin_verbose']
        removal_keys = ['iterations', 'spaces', 'freeze_transd']
        burnin_kwargs = self.sampler_function_kwargs.copy()
        for key in extra_keys:
            if key in self.kwargs:
                burnin_kwargs[key] = self.kwargs[key]
        if 'burnin_verbose' in burnin_kwargs.keys():
            burnin_kwargs['verbose'] = burnin_kwargs.pop('burnin_verbose')
        for key in removal_keys:
            if key in burnin_kwargs.keys():
                burnin_kwargs.pop(key)
        return burnin_kwargs

    @property
    def sampler_init_kwargs(self):
        init_kwargs = {key: value
                       for key, value in self.kwargs.items()
                       if key not in self.sampler_function_kwargs and key not in self.sampler_burnin_kwargs}
        init_kwargs.pop("burnin_verbose")
        init_kwargs['lnpostfn'] = self.lnpostfn
        init_kwargs['ndim'] = self.ndim

        # have to make sure pool is None so sampler will be pickleable
        init_kwargs['pool'] = None
        return init_kwargs

    def _initialise_sampler(self):
        import kombine
        self._sampler = kombine.Sampler(**self.sampler_init_kwargs)
        self._init_chain_file()

    def _set_pos0_for_resume(self):
        # take last iteration
        self.pos0 = self.sampler.chain[-1, :, :]

    @property
    def sampler_chain(self):
        # remove last iterations when resuming
        nsteps = self._previous_iterations
        return self.sampler.chain[:nsteps, :, :]

    def check_resume(self):
        return self.resume and os.path.isfile(self.checkpoint_info.sampler_file)

    def run_sampler(self):
        if self.autoburnin:
            if self.check_resume():
                logger.info("Resuming with autoburnin=True skips burnin process:")
            else:
                logger.info("Running kombine sampler's automatic burnin process")
                self.sampler.burnin(**self.sampler_burnin_kwargs)
                self.kwargs["iterations"] += self._previous_iterations
                self.nburn = self._previous_iterations
                logger.info("Kombine auto-burnin complete. Removing {} samples from chains".format(self.nburn))
                self._set_pos0_for_resume()

        tqdm = get_progress_bar()
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')
        iterations -= self._previous_iterations
        sampler_function_kwargs['p0'] = self.pos0
        for sample in tqdm(
                self.sampler.sample(iterations=iterations, **sampler_function_kwargs),
                total=iterations):
            self.write_chains_to_file(sample)
        self.checkpoint()
        self.result.sampler_output = np.nan
        if not self.autoburnin:
            tmp_chain = self.sampler.chain.copy()
            self.calculate_autocorrelation(tmp_chain.reshape((-1, self.ndim)))
            self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps`. Try increasing the number of steps.")
        tmp_chain = self.sampler.chain[self.nburn:, :, :].copy()
        self.result.samples = tmp_chain.reshape((-1, self.ndim))
        blobs = np.array(self.sampler.blobs)
        blobs_trimmed = blobs[self.nburn:, :, :].reshape((-1, 2))
        self.calc_likelihood_count()
        log_likelihoods, log_priors = blobs_trimmed.T
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.log_prior_evaluations = log_priors
        self.result.walkers = self.sampler.chain.reshape((self.nwalkers, self.nsteps, self.ndim))
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result
