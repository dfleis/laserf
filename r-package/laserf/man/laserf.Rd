% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/laserf.R
\name{laserf}
\alias{laserf}
\title{Locally Adaptive Subspace Estimation Random Forest}
\usage{
laserf(
  X,
  Y,
  num.trees = 2000,
  sample.weights = NULL,
  clusters = NULL,
  equalize.cluster.weights = FALSE,
  sample.fraction = 0.5,
  mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
  min.node.size = 5,
  honesty = TRUE,
  honesty.fraction = 0.5,
  honesty.prune.leaves = TRUE,
  alpha = 0.05,
  imbalance.penalty = 0,
  compute.oob.predictions = TRUE,
  num.threads = NULL,
  seed = runif(1, 0, .Machine$integer.max),
  .env = NULL
)
}
\arguments{
\item{X}{TODO... auxiliary covariates}

\item{Y}{TODO... primary covariates}

\item{num.trees}{TODO...}

\item{sample.weights}{TODO...}

\item{clusters}{TODO...}

\item{equalize.cluster.weights}{TODO...}

\item{sample.fraction}{TODO...}

\item{mtry}{TODO... <<HIGH PRIORITY>>}

\item{min.node.size}{TODO... <<HIGH PRIORITY>>}

\item{honesty}{TODO...}

\item{honesty.fraction}{TODO...}

\item{honesty.prune.leaves}{TODO...}

\item{alpha}{TODO... <<HIGH PRIORITY>>}

\item{imbalance.penalty}{TODO... <<HIGH PRIORITY>>}

\item{compute.oob.predictions}{TODO...}

\item{num.threads}{TODO...}

\item{seed}{TODO...}

\item{.env}{Environment. Used during development as a container to help with debugging.}
}
\description{
TODO:
 * Which input parameters are still relevant.
 * How to think about `sample.weights` and `clusters` parameters.
 * How do deal with `NA` data.
 * Clean up validation of primary covariates `Y` (n by p dimensional)
 * Clean up validation of auxiliary covariates `X` (n by d dimension)
 *
}
\examples{
\donttest{
# TODO...
}

}
