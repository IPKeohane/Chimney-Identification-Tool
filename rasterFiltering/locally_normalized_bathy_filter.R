library(terra)

##########################

locNormBathy <- function(my_ras, ws, scale_fact=50){
  
temp_in  = focal(my_ras, w=matrix(1,ws,ws), fun=min, na.rm=TRUE) 
temp_out = (my_ras - temp_in)/scale_fact

return(temp_out)

}



