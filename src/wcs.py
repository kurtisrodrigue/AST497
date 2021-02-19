from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from astropy import coordinates
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import ICRS, Galactic, FK4, FK5

lon = '8h51m49.36s'
lat = '+11d53m39.45s'
#lon = '20h10m44.93s'
#lat = '+04d11m41.2s'
file = '../data/fits/M67-B.fts'

Coord = SkyCoord(ra= lon, dec = lat, frame = FK5)
print(Coord)

f = fits.open(file)
w = WCS(f[0].header)
x,y = skycoord_to_pixel(Coord, w, mode='all')
radec = pixel_to_skycoord(1425,3960, w, mode='all', cls=SkyCoord)
radec2 = pixel_to_skycoord(0,0, w, mode='all', cls=SkyCoord)

print(radec)
print(radec2)
print (x,y)
#should be 1425 3960