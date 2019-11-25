from __future__ import division, print_function
import numpy as np
import bilby

duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'ResultParity'
label = '2019-11-25-log10lambda-4params'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(19930309)

#Injection
injection_parameters = dict(
    mass_1=30, 
    mass_2=30,
    luminosity_distance=400,
    theta_jn=0, 
    psi=2.659,
    ra=1.375, dec=-1.2108, 
    a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, 
    phase=1.3, geocent_time=1126259642.413, 
    dchi_2=0.0,dchi_3=0.0,mg_lambda=0.0,dalpha_2=0.0,parity_lambdatilt=0.0, parity_log10lambdatilt = 0.0, parity_alpha=0)

priors = bilby.gw.prior.BBHPriorDict()

priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
            'dec', 'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]

#Testing GR parameters
priors['mg_lambda'] = 0.0
priors['dchi_2'] = 0.0
priors['dchi_3'] = 0.0
priors['dalpha_2'] = 0.0
priors['parity_lambdatilt'] = 0.0
priors['parity_log10lambdatilt'] = bilby.core.prior.Uniform(
    minimum= 5,
    maximum=  15,
    name='parity_log10lambdatilt', latex_label=r'$\log_10\tilde\Lambda$')
priors['parity_alpha'] = 1


# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_nonGR_binary_black_hole,
    waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1','V1'])

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)

ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)


likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
    priors=priors  
)#, phase_marginalization=True,
    #time_marginalization=True)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

result.plot_corner()