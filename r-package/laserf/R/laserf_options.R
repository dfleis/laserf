#' laserf package options
#'
#' laserf package options can be set using R's \code{\link{options}} command.
#' The current available options are:
#' \itemize{
#'  \item `laserf.legacy.seed`: controls whether laserf's random seed behavior depends on
#'  the number of CPU threads used to train the forest. The default value is `FALSE`.
#'  Set to `TRUE` to recover random seed behaviour inherited from grf versions prior to 2.4.0.
#' }
#'
#' @return Prints the current laserf package options.
#'
#' @examples
#' \donttest{
#' # Use random seed behavior prior to version 2.4.0.
#' options(laserf.legacy.seed = TRUE)
#'
#' # Print current package options.
#' laserf_options()
#'
#' # Use random seed independent of num.threads (default).
#' options(laserf.legacy.seed = FALSE)
#' }
#'
#' @export
laserf_options <- function() {
    print(c(
        laserf.legacy.seed = get_legacy_seed()
    ))
}
