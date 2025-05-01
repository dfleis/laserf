####################################################################################################
# pkg-load-build-cycle.R
# Run the different stages of the package build cycle.
#
####################################################################################################
library(Rcpp)
library(devtools)
library(usethis)
# library(testthat)
library(bench)

rm(list = ls()); gc()

#----- (Optional) Update version numbering in DESCRIPTION
# usethis::use_version("dev")
# usethis::use_version("patch")
# usethis::use_version("minor")
# usethis::use_version("major")

#----- (Optional) Different ways to clean if we're worried about conflicts
# devtools::unload() # Unload from R session (usually sufficient if we're having problems)
# devtools::uninstall(unload = T) # Combines unloading and removal (dev-friendly version of remove.packages)

#----- Development cycle
tm_load <- bench::system_time(devtools::load_all()) # Load the current version (and checks to recompile)
# devtools::load_all(compile = FALSE) 
tm_doc <- bench::system_time(devtools::document()) # Update documentation
# devtools::test()   # Run testthat tests

#----- Build cycle
tm_check <- bench::system_time(devtools::check()) # Catch errors before build

devtools::clean_dll()
tm_build <- bench::system_time(devtools::build())

# devtools::install() # If we want to test an actual installation

#----- (Optional) Test build from github
# bench::system_time(
#   devtools::install_github("dfleis/laserf", subdir = "r-package/laserf")
# )
# bench::system_time(
#   devtools::install_github("dfleis/laserf", subdir = "r-package/laserf", ref = "dev")
# )




