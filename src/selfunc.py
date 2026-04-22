import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import KDTree
from matplotlib import cm
from scipy.interpolate import LinearNDInterpolator
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import os

def read_stemo21_selfunc():
    """read in stemo,et al+2021 selection function"""
    # read plot image
    im = mpimg.imread("/home/insepien/dftime/data/c.png")
    im = np.flip(im,axis=0)
    # get the RGB color arrays, im[:,:,3] is transparency
    rgb_image = im[:,:,:3]

    # make a lut -- look up table for viridis cmap
    viridis = cm.get_cmap('viridis', 256)
    lut = viridis(np.linspace(0, 1, 256))[:, :3]

    # reshape image for query
    pixels = rgb_image.reshape(-1, 3)
    # kd-tree for quick nearest-neighbor lookup, return position in lut array
    tree = KDTree(lut)
    _, indices = tree.query(pixels)
    # reshape back to 2D
    data_2d = indices.reshape(rgb_image.shape[0], rgb_image.shape[1])
    # Scale to the actual units (0.0 to 0.8 according to paper colorbar)
    final_data = (data_2d / 255.0) * 0.8
    return im,final_data

def sample_stemo_selfunc(final_data):
    """sample stemo plot to avoid grid lines"""
    nq,npix = final_data.shape
    # mesh with sampling indices
    pcoords = np.arange(0,npix,25)[1:]-1
    qcoords = np.arange(0,nq,20)[1:]-3
    pcoords,qcoords = [np.insert(arr,0,0) for arr in [pcoords,qcoords]]
    pcoords,qcoords = [np.insert(arr,len(arr),n-1) for arr,n in zip([pcoords,qcoords],[npix,nq])]
    PIX,Q = np.meshgrid(pcoords,qcoords)
    # sample data
    csamp = np.array([final_data[y,x] for x,y in zip(PIX,Q)])
    return PIX+1,Q+1,csamp,nq,npix

def get_interp_selfunc():
    """main function to read stemo plot, sample, and interpolate"""
    # read stemo data
    _,final_data = read_stemo21_selfunc()
    # sample that plot
    PIX,Q,csamp,nq,npix = sample_stemo_selfunc(final_data)
    # conversion from indices to paper ticks
    ipix = npix/200
    iq = nq/100
    # interpolate
    pflat = PIX.ravel()
    qflat = Q.ravel()
    interp = LinearNDInterpolator(list(zip(qflat/iq,pflat/ipix)),csamp.ravel())
    return interp

def kpc_to_pix(sep_kpc,plate_scale=0.05,z=0.2):
    # convert physical sep to pix sep
    fwhm = plate_scale*2.5 *u.arcsec #arcsec/pix, WFC3/UVIS
    sep_theta = ((sep_kpc*u.kpc/cosmo.angular_diameter_distance(z)).to("")*u.rad).to(u.arcsec)
    sep_pix = sep_theta/fwhm
    return sep_pix

def psel(sep_kpc,z,q,plate_scale=0.05):
    """return p(obs|sep in kpc, mass ratio q)"""
    # get interpolator and conversion factor from index to plot values
    interp = get_interp_selfunc()
    sep_pix = kpc_to_pix(sep_kpc,plate_scale,z)
    return interp(q,sep_pix)