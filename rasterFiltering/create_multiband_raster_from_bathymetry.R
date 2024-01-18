library(terra)
source("bpi_filter.R")
source("locally_normalized_bathy_filter.R")

####
fp_in  = "../data/gsc_ps2_AUVBath_filled_utm.tif"
fp_out = "../data/test_mulitband_out.tif"

####
bath_rast = rast(fp_in)
###

outRas0 = locNormBathy(bath_rast, ws=31, scale_fact=50) # subtract min within ws and divide by scale fact
outRas1 = terrain(bath_rast, v="slope", unit="degrees", neighbors=8) # local slope
outRas2 = bpi(bath_rast, ws_i=3, ws_o=11) # inner radius = 3 pixels, outer radius = 11 pixels

###
outRas = c(outRas0, outRas1/90, outRas2/20)
names(outRas) = c("norm_bathy", "slope", "BPI_0311")

writeRaster(outRas, fp_out, overwrite=TRUE)

