#
#Convert RData to csv, because the package pyreadr does not
#support S3 lists in R
#

library(rstudioapi)
setwd(dirname(getSourceEditorContext()$path))
load("icon_eps_t_2m.RData")
write.csv(data_icon_eps, "icon_eps_t_2m.csv", row.names = FALSE)
load("icon_eps_wind_10m.RData")
write.csv(data_icon_eps, "icon_eps_wind_10m.csv", row.names = FALSE)

