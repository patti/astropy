.. _convolve-models:

Model Convolution
=================

.. versionadded:: 1.1

Models can be convolved using the :func:`~astropy.convolution.convolve_model` function. The result is a CompoundModel that performs the convolution on evaluation. For example a source model may be convolved with a instrument response function or point spread function. The convolved model can then be fit to data.

It is the user's responsibility to make sure the kernel is both centered and normalized as no errors or warnings will be raised. Usage and fitting are illustrated in the following examples

An Absorption Line Convolved with a Gaussian Instrument Response
----------------------------------------------------------------
The absorption line and Instrument Response Function (IRF) are each modeled by `astropy.modeling.functional_models.Gaussian1D`. In this example, the IRF parameters are fixed and the convolved compound model is fit to simulated data to recover the source model parameters.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plot
    from astropy.modeling import models, fitting
    from astropy.convolution.convolve import convolve_model
    from scipy.signal import fftconvolve

    np.random.seed(100)

    # Absorption Line Model
    mean = 0
    amp = -2
    stddev = .01
    src = models.Gaussian1D(amp, mean, stddev, name='Source')

    # Normalized Gaussian PSF Model
    irf_width = .3
    amplitude = 1 / np.sqrt(2 * np.pi * irf_width ** 2)
    irf = models.Gaussian1D(mean=mean, stddev=irf_width,
                amplitude=amplitude, name='IRF')

    # Fix the IRF parameters
    for param in irf.param_names: irf.fixed[param] = True

    # Create convolved CompoundModel object
    cmod = convolve_model(src, irf).rename('Convolved')

    # Simulate data image
    # Note that the inputs are odd-shaped and centered on the kernel mean
    x = np.linspace(-10, 10, 1001)
    expected = cmod(x)
    noise = np.random.normal(0, 1, (expected.shape))
    cim = expected + noise

    # Offset model parameters to test fitting
    cmod.mean_0 = .1
    cmod.stddev_0 = .02
    cmod.amplitude_0 = -2.5

    # Fit
    fitter = fitting.LevMarLSQFitter()
    fit_mod = fitter(cmod, x, cim)
    fit = fit_mod(x)

    # Plot
    plt.figure(figsize=(16,12))
    for i,(t,im) in enumerate([('Source', src(x)), 
                    ('IRF', irf(x)), ('Data', cim), 
                    ('Fit', fit), ('Residual', cim-fit), 
                    ('Actual - Fit', expected-fit)]):
        plt.subplot(3, 2, i+1)
        plt.plot(x, im)
        if i>3:
            plt.ylim(-.4, .4)
        plt.title(t)
        plt.xlabel('x')
        plt.ylabel('Amplitude')

An Elliptical Galaxy convolved with a Gaussian PSF
--------------------------------------------------
The galaxy model is given by the de Vancouleur's profile using `astropy.modeling.functional_models.Sersic2D` with sersic index, ``n=4``. The Point Spread Function (PSF) is a `astropy.modeling.functional_models.Gaussian2D` model instance with fixed parameters. The convolved compound model is fit to simulated data to recover the source model parameters.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plot
    from astropy.modeling import models, fitting
    from astropy.convolution.convolve import convolve_model
    from scipy.signal import fftconvolve

    np.random.seed(100)

    # DeVanCoulier's Galaxy Source Model
    x0, y0 = 50, 50
    r_eff = 5000
    n = 4
    b0 = .1
    ellip = .5
    theta = np.pi / 3
    src = models.Sersic2D(b0, r_eff, n, x0, y0, ellip, theta, name='Source')

    # Normalized Gaussian PSF Model
    psf_width = 5
    amplitude = 1 / (2 * np.pi * psf_width ** 2)
    psf = models.Gaussian2D(x_mean = x0, y_mean=y0,
                     x_stddev=psf_width, y_stddev=psf_width,
                     amplitude=amplitude,
                     name='PSF')
    for param in psf.param_names: psf.fixed[param] = True

    # Create convolved CompoundModel object
    cmod = convolve_model(src, psf).rename('Convolved')

    # Simulate data image
    # Note that the inputs are odd-shaped and centered on the kernel mean
    y,x = np.indices((101, 101))
    expected = cmod(x, y)
    noise = np.random.normal(0, .1, (expected.shape))
    cim = expected + noise

    # Offset model parameters to test fitting
    cmod.amplitude_0 = .15
    cmod.r_eff_0 = 5500
    cmod.n_0 = 3.6
    cmod.ellip_0 = .64
    cmod.theta_0 = np.pi/4
    cmod.x_0 = 45
    cmod.y_0=55

    # Fit
    fitter = fitting.LevMarLSQFitter()
    fit_mod = fitter(cmod, x, y, cim)
    fit = fit_mod(x, y)

    # Plot
    plt.figure(figsize=(16,8))
    for i,(t,im) in enumerate([('Source',src(x, y)), 
                    ('PSF', psf(x, y)), ('Data', cim), 
                    ('Fit', fit), ('Residual', cim-fit), 
                    ('Actual - Fit', expected-fit)]):
        plt.subplot(2, 3, i+1)
        if i in [0, 2, 3]: 
            vmin, vmax = 0, 60
        elif i==1: 
            vmin, vmax = 0, psf.amplitude.value
        else: 
            vmin, vmax = -5, 5
        plt.imshow(im, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(t)
        plt.xlabel('x')
        plt.ylabel('y')
