library(terra)

##########################

matrixHole <- function(ws_i, ws_o){
ro = (ws_o-1)/2
ri = (ws_i-1)/2

mtx = matrix(1/(ws_o**2-ws_i**2),ws_o,ws_o)

mtx[((ro+1)-ri):((ro+1)+ri),((ro+1)-ri):((ro+1)+ri)]=0

return(mtx)
}

###########################

bpi <- function(my_ras, ws_i, ws_o){
  
temp_in  = focal(my_ras, w=matrix(1,ws_i,ws_i), fun=mean, na.rm=FALSE) 
temp_out = focal(my_ras, w=matrixHole(ws_i, ws_o), fun=sum, na.rm=FALSE) 

return(temp_in - temp_out)

}



