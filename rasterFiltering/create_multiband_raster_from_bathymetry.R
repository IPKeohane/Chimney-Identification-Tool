library(terra)
source("bpi_filter.R")
source("locally_normalized_bathy_filter.R")

####
fp_in  = "../data/cit_test_bathy_gsc_ll_1m.tif"
fp_out = "../data/cit_test_mulitbandRaster_gsc_1m.tif"

####
bath_rast = rast(fp_in)
###

outRas0 = locNormBathy(bath_rast, ws=31, scale_fact=50) # subtract min bathy within window size in pixels (ws=31) and divide by scale factor (50)
outRas1 = terrain(bath_rast, v="slope", unit="degrees", neighbors=8) # local slope
outRas2 = bpi(bath_rast, ws_i=3, ws_o=11) # bathymetric position index, inner radius = 3 pixels, outer radius = 11 pixels

###
outRas = c(outRas0, outRas1/90, outRas2/20)
names(outRas) = c("norm_bathy", "slope", "BPI_0311")

writeRaster(outRas, fp_out, overwrite=TRUE)



bath_rast = project(bath_rast, "epsg:4326")
