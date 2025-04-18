% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/regression_forest.R
\name{predict.regression_forest}
\alias{predict.regression_forest}
\title{Predict with a regression forest}
\usage{
\method{predict}{regression_forest}(
  object,
  newdata = NULL,
  num.threads = NULL,
  estimate.variance = FALSE,
  ...
)
}
\arguments{
\item{object}{The trained forest.}

\item{newdata}{Points at which predictions should be made. If NULL, makes out-of-bag
predictions on the training set instead (i.e., provides predictions at
Xi using only trees that did not use the i-th training example). Note
that this matrix should have the number of columns as the training
matrix, and that the columns must appear in the same order.}

\item{num.threads}{Number of threads used in prediction. If set to NULL, the software
automatically selects an appropriate amount.}

\item{estimate.variance}{Whether variance estimates for \eqn{\hat\tau(x)} are desired
(for confidence intervals).}

\item{...}{Additional arguments (currently ignored).}
}
\value{
Vector of predictions, along with estimates of the error and
        (optionally) its variance estimates. Column 'predictions' contains
        estimates of E[Y|X=x]. The square-root of column 'variance.estimates' is the standard error
        the test mean-squared error. Column 'excess.error' contains
        jackknife estimates of the Monte-carlo error. The sum of 'debiased.error'
        and 'excess.error' is the raw error attained by the current forest, and
        'debiased.error' alone is an estimate of the error attained by a forest with
        an infinite number of trees. We recommend that users grow
        enough forests to make the 'excess.error' negligible.
}
\description{
Gets estimates of E[Y|X=x] using a trained regression forest.
}
\examples{
\donttest{
# Train a standard regression forest.
n <- 50
p <- 10
X <- matrix(rnorm(n * p), n, p)
Y <- X[, 1] * rnorm(n)
r.forest <- regression_forest(X, Y)

# Predict using the forest.
X.test <- matrix(0, 101, p)
X.test[, 1] <- seq(-2, 2, length.out = 101)
r.pred <- predict(r.forest, X.test)

# Predict on out-of-bag training samples.
r.pred <- predict(r.forest)

# Predict with confidence intervals; growing more trees is now recommended.
r.forest <- regression_forest(X, Y, num.trees = 100)
r.pred <- predict(r.forest, X.test, estimate.variance = TRUE)
}

}
