# Function definitions in order to calculate the various components
# of the effective diffusivity of a tracer (Nakamura 1996). This script
# is made especially to use 2D passive tracer data from the Gulf Stream
# jet extension. This script was creating building on a similar script
# produce by Ryan Abernthey, University of Columbia - this can be
# found at github.com/rabernat/effdiff.

import numpy as np

# calculate spatial gradients of tracer for each week
def calcTracerGradients( jetTracer, dx, dy ) :

    # pre-allocate for tracer gradients
    dpdx = np.zeros( jetTracer.shape )
    dpdy = np.zeros( jetTracer.shape )

    # calculate spatial tracer gradients week-by-week
    for week in range(1137) :

        weekly2DMap = jetTracer[ week, :, : ]
        weeklyGradients = np.gradient( weekly2DMap )
        dpdy[ week, :, : ] = weeklyGradients[0]
        dpdx[ week, :, : ] = weeklyGradients[1]

    dpdx = dpdx/dx
    dpdy = dpdy/dy

    return dpdx, dpdy


# make K_eff a function of latitude instead of tracer values
def interpFromTracerToLat( effDiff, meanTracer, linearTracer ) :

    meanTracer = meanTracer[::-1]

    # interpolate K_eff values week-by-week
    for week in range(1137) :

        weeklyEffDiff = effDiff[ week, : ]
        weeklyEffDiffInterp = np.interp( meanTracer, linearTracer, weeklyEffDiff )
        effDiff[ week, : ] = weeklyEffDiffInterp

    return effDiff


# define a function to calculate the effective diffusivity for each week
def calcEffDiff(jetTracer):

    # specify (prescribed) model diffusivity (m^2s^-1)
    kappa = 100

    # define grid spacing in x and y (in m)
    dx = 111.3 * np.cos(40 * np.pi / 180) * (10 ** 3) / 10
    dy = 111 * (10 ** 3) / 10

    # first average in time and along x axis, to obtain a mapping
    # between tracers values and y
    meanTracer = np.mean(jetTracer, axis=0)
    meanTracer = np.mean(meanTracer, axis=1)

    # choose how many tracer values to use
    N = list( meanTracer.shape )[0]

    # produce linearly spaced tracer values
    linearTracer = np.linspace( meanTracer.min(), meanTracer.max(), N )

    # calculate spatial gradients of tracer
    dpdx, dpdy = calcTracerGradients( jetTracer, dx, dy )

    # calculate gradient squared
    gradSquared = np.power( dpdx, 2 ) + np.power( dpdy, 2 )

    # pre-allocate area and integral arrays
    areaBelowP = np.zeros( (1137,N) )
    integralOfGradSquared = np.zeros( (1137,N) )

    # loop through each of the linearly space tracer values, and calculate
    # the area BELOW each tracer value, as well as the integral of the
    # gradient squared at those positions
    for n, p in enumerate( linearTracer ) :

        # loop through each week
        for week in range(1137) :

            # for this week, find locations where tracer <= p
            weekly2DMap = jetTracer[ week, : , : ]
            indices = weekly2DMap <= p

            # calculate area with tracer <= p
            areaBelowP[ week, n ] = np.count_nonzero( indices )*dx*dy

            # integrate gradient squared over area with tracer <= p
            weeklyGradSquared = gradSquared[ week, :, : ]
            integralOfGradSquared[ week, n ] = np.sum( weeklyGradSquared[ indices ] )*dx*dy

    # pre-allocate for derivatives
    dpdA = np.zeros( (1137,N) )
    dXdA = np.zeros( (1137,N) )

    # for convenience, let the integral of the gradient squared equal X. Now
    # calculate the derivative of both the tracer p and X, with respect
    # to the area A, i.e. dpdA and dXdA
    for week in range(1137) :

        dpdA[ week, : ] = np.divide( np.gradient( linearTracer ), np.gradient( areaBelowP[ week, :] ) )
        dXdA[ week, : ] = np.divide( np.gradient( integralOfGradSquared[ week, :] ), np.gradient( areaBelowP[week, : ] ) )


    # calculate the equivalent length squared ( Le^2 = dXdA / ( dpdA^2 ) )
    Le2 = np.divide( dXdA, np.power( dpdA, 2 ) )


    # the minimum length of a tracer contour is the length of the domain
    # in the x direction
    Lmin = jetTracer.shape[2]*dx


    # calculate effective diffusivity
    effDiff = kappa*Le2/(Lmin*Lmin)

    # the effective diffusivity gives us a value of K_eff at each week, at
    # each tracer value. It's better to have K_eff as a function of latitude
    # instead, so interpolate to linearly spaced latitutde values
    effDiff = interpFromTracerToLat( effDiff, meanTracer, linearTracer )

    return effDiff

# calculate the sub-annual variation of the effective diffusivity
def calcAnnualComposite( effDiff ) :

    # pre-allocate
    effDiffComposite = np.zeros( (52, effDiff.shape[1] ) )

    # loop through each year
    for year in range(21) :

        # loop through each week
        for week in range(52) :

            # loop through each latitude value
            for lat in range( effDiff.shape[1] ) :

                effDiffComposite[ week, lat ] += effDiff[ 1 + year*52 + week, lat ]

    # need to divide by number of years to get the average
    effDiffComposite = effDiffComposite/21

    return  effDiffComposite

# calculate annual means of effective diffusivity
def calcAnnualMeans( effDiff ) :

    # pre-allocate
    effDiffAnnual = np.zeros( ( 21, effDiff.shape[1] ) )
    effDiffStdDev = np.zeros( (1, effDiff.shape[1] ) )

    # loop through each year
    for year in range(21) :

        effDiffAnnual[ year, : ] = np.mean( effDiff[ 1 + year*52 : (year + 1)*52, :], 0 )

    # calculate the inter-annual standard deviation
    effDiffStdDev= np.std( effDiffAnnual, 0 )

    return effDiffAnnual, effDiffStdDev
