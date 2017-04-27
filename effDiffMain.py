# This is a python script to take 2D (in space) passive tracer
# data and calculate the time mean effective diffusivity. The
# effective diffusivity is described in more detail in
# Nakamura (1996), Shuckburgh and Haynes (2003), and Abernathey
# and Marshall (2013).


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import scipy.ndimage.interpolation as imrotate
from effDiffFunctions import *


# import passive tracer data from .mat file
dict = sio.loadmat( 'p_gulf.mat' )
pGulf = dict['p_gulf']

# reverse directions of x and y axes
pGulf = pGulf[:,::-1,:]
pGulf = pGulf[:,:,::-1]

# make a passive tracer snapshot from week 100
snapshot = pGulf[100,:,:]
snapshot[ snapshot == 0 ] = 0.2
snapshotRot = imrotate.rotate( snapshot, -9 )

# pre-allocate for rotated tracer values
dim =  snapshotRot.shape
pGulfRot = np.zeros( ( 1137, dim[0], dim[1] ) )

# rotate passive tracer data week-by-week
for week in range(1137) :

    weeklyTracer = pGulf[week, :, :]
    pGulfRot[week,:,:] = imrotate.rotate( weeklyTracer, -9 )

# define region and extract values from jet extension
X0 = [28,250]
Y0 = [50,137]
pGulfRot = pGulfRot[ :, Y0[0]:Y0[1]+1, X0[0]:X0[1]+1 ]

# calculate the effective diffusivity for each week
effDiff = calcEffDiff( pGulfRot )
meanEffDiff = np.mean( effDiff, 0 )

# calculate sub-annual variations
effDiffComposite = calcAnnualComposite( effDiff )
effDiffComposite = np.transpose( effDiffComposite )

# calculate annual means and standard deviations
effDiffAnnual, effDiffStdDev = calcAnnualMeans( effDiff )

###### PLOTTING #####
dy = 111 / 10
y = np.linspace( 0, meanEffDiff.shape[0], meanEffDiff.shape[0] )*dy

fig, axArray = plt.subplots( 2, 2, figsize=(13,8) )

ax0 = axArray[0,0]
ax1 = axArray[1,0]
ax2 = axArray[0,1]
ax3 = axArray[1,1]

# plot tracer snapshot and illustrate region
ax0.pcolormesh( snapshotRot )
ax0.add_patch( patches.Rectangle( (X0[0],Y0[0] ), X0[1]-X0[0], Y0[1]-Y0[0], fill=False, edgecolor='white' ) )
ax0.invert_xaxis()
ax0.axis('off')
ax0.set_title('Illustration of Region Sampled')


# plot time-mean effective diffusivity, with spread
ax1.fill_between( y, ( meanEffDiff - effDiffStdDev )/1000, ( meanEffDiff + effDiffStdDev )/1000 )
ax1.plot( y, meanEffDiff/1000, color='black' )
ax1.set_xlabel('Distance, Perpendicular to Jet (km)')
ax1.set_ylabel('Effective Diffusivity (1000 m2s-1)')
ax1.set_ylim( [0,8.5] )
ax1.set_xlim( [y[0],y[-1]] )
ax1.set_title('Time-Mean Effective Diffusivity 1993-2014')


# plot weeky-by-week effective diffusivity as 2D colourmap
im2 = ax2.pcolormesh( np.linspace(1,52,52), y, effDiffComposite )
fig.colorbar( im2, ax=ax2)
ax2.set_ylabel('Distance (km)')
ax2.set_xlabel('Week', labelpad=0 )
ax2.set_title('Effective Diffusivity Weekly-Composite')
ax2.invert_yaxis()


# plot inter-annual variability
ax3.plot( y, np.transpose( effDiffAnnual/1000 ) )
ax3.set_xlabel('Distance, Perpendicular to Jet (km)')
ax3.set_ylabel('Effective Diffusivity (1000 m2s-1)')
ax3.set_xlim( [y[0],y[-1]] )
ax3.set_ylim( [0,8.5] )
ax3.set_title('Annual Means of Effective Diffusivity')


plt.show()
fig.savefig('gulfStreamPassiveTracer.png', bbox_inches='tight', dpi=600)










