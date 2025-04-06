validate_X <- function(X, allow.na = FALSE) {
  valid.classes <- c("matrix", "data.frame")

  if (!inherits(X, valid.classes)) {
    stop(paste(
      "Currently the only supported data input types are:",
      "`matrix`, `data.frame`"
    ))
  }
  if (any(0 %in% dim(X))) {
    stop("Feature matrix X must have non-zero dimensions.")
  }

  if (!is.numeric(as.matrix(X))) {
    stop(paste(
      "The feature matrix X must be numeric. LASERF does not",
      "currently support non-numeric features. If factor variables",
      "are required, we recommend one of the following: Either",
      "represent the factor with a 1-vs-all expansion,",
      "(e.g., using model.matrix(~. , data=X)), or then encode the factor",
      "as a numeric via any natural ordering (e.g., if the factor is a month)."
    ))
  }

  has.missing.values <- anyNA(X)

  if (!allow.na && has.missing.values) {
    stop("The feature matrix X contains at least one NA.")
  }

  has.missing.values
}

validate_observations <- function(V, X, allow.matrix = FALSE) {
  if (!allow.matrix) {
    if (is.matrix(V) && ncol(V) == 1) {
      V <- as.vector(V)
    } else if (!is.vector(V)) {
      stop("Observations (W, Y, Z or D) must be vectors.")
    }
  } else {
    if (is.matrix(V) || is.data.frame(V) || is.vector(V)) {
      V <- as.matrix(V)
    } else {
      stop("Observations Y must be either a vector/matrix/data.frame.")
    }
  }

  if (!is.numeric(V) && !is.logical(V)) {
    stop(paste(
      "Observations (W, Y, Z or D) must be numeric. LASERF does not ",
      "currently support non-numeric observations."
    ))
  }

  if (anyNA(V)) {
    stop("The vector of observations (W, Y, Z or D) contains at least one NA.")
  }

  if (NROW(V) != nrow(X)) {
    stop("length of observation (W, Y, Z or D) does not equal nrow(X).")
  }
  V
}

validate_num_threads <- function(num.threads) {
  if (is.null(num.threads)) {
    num.threads <- 0
  } else if (!is.numeric(num.threads) | num.threads < 0) {
    stop("Error: Invalid value for num.threads")
  }
  num.threads
}

validate_clusters <- function(clusters, X) {
  if (is.null(clusters) || length(clusters) == 0) {
    return(vector(mode = "numeric", length = 0))
  }
  if (mode(clusters) != "numeric") {
    stop("Clusters must be able to be coerced to a numeric vector.")
  }
  clusters <- as.numeric(clusters)
  if (!all(clusters == floor(clusters))) {
    stop("Clusters vector cannot contain floating point values.")
  } else if (length(clusters) != nrow(X)) {
    stop("Clusters vector has incorrect length.")
  } else {
    # convert to integers between 0 and n clusters
    clusters <- as.numeric(as.factor(clusters)) - 1
  }
  clusters
}

validate_equalize_cluster_weights <- function(equalize.cluster.weights, clusters, sample.weights) {
  if (is.null(clusters) || length(clusters) == 0) {
    return(0)
  }
  cluster_size_counts <- table(clusters)
  if (equalize.cluster.weights == TRUE) {
    samples.per.cluster <- min(cluster_size_counts)
    if (!is.null(sample.weights)) {
      stop("If equalize.cluster.weights is TRUE, sample.weights must be NULL.")
    }
  } else if (equalize.cluster.weights == FALSE) {
    samples.per.cluster <- max(cluster_size_counts)
  } else {
    stop("equalize.cluster.weights must be either TRUE or FALSE.")
  }

  samples.per.cluster
}

validate_newdata <- function(newdata, X, allow.na = FALSE) {
  validate_X(newdata, allow.na = allow.na)
  if (ncol(newdata) != ncol(X)) {
    stop("newdata must have the same number of columns as the training matrix.")
  }
}

validate_sample_weights <- function(sample.weights, X) {
  if (!is.null(sample.weights)) {
    if (length(sample.weights) != nrow(X)) {
      stop("sample.weights has incorrect length")
    }
    if (anyNA(sample.weights) || any(sample.weights < 0) || any(is.infinite(sample.weights))) {
      stop("sample.weights must be nonnegative and without missing values")
    }
  }
}

validate_target_rank <- function(target.rank) {
  if (!isTRUE(target.rank >= 1)) { # using isTRUE handles cases like NULL, NA, NA_real_
    stop("target.rank must be a positive integer >= 1.")
  } else if (isFALSE(is.numeric(target.rank)) || length(target.rank) != 1) {
    stop("target.rank must be a positive integer.")
  }
  as.integer(target.rank) # rounds down (up to a precision of .Machine$double.eps)
}

# Indices are offset by 1 for C++.
create_train_matrices <- function(X,
                                  outcome = NULL,
                                  treatment = NULL,
                                  sample.weights = FALSE) {
  out <- list()
  offset <- ncol(X) - 1
  if (!is.null(outcome)) {
    out[["outcome.index"]] <- (offset + 1):(offset + NCOL(outcome))
    offset <- offset + NCOL(outcome)
  }
  if (!is.null(treatment)) {
    out[["treatment.index"]] <- (offset + 1):(offset + NCOL(treatment))
    offset <- offset + NCOL(treatment)
  }
  # Forest bindings without sample weights: sample.weights = FALSE
  # Forest bindings with sample weights:
  # -sample.weights = NULL if no weights passed
  # -sample.weights = numeric vector if passed
  if (is.logical(sample.weights)) {
    sample.weights <- NULL
  } else {
    out[["sample.weight.index"]] <- offset + 1
    if (is.null(sample.weights)) {
      out[["use.sample.weights"]] <- FALSE
    } else {
      out[["use.sample.weights"]] <- TRUE
    }
  }

  X <- as.matrix(X)
  out[["train.matrix"]] <- as.matrix(cbind(X, outcome, treatment, sample.weights))

  out
}

create_test_matrices <- function(X) {
  out <- list()
  out[["test.matrix"]] <- as.matrix(X)

  out
}

# Call the laserf Rcpp bindings (argument_names) with R argument.names
#
# All the bindings argument names (C++) have underscores: sample_weights, train_matrix, etc.
# On the R side each variable name is written as sample.weights, train.matrix, etc.
# This function simply replaces the underscores in the passed argument names with dots.
do.call.rcpp = function(what, args, quote = FALSE, envir = parent.frame()) {
  names(args) = gsub("\\.", "_", names(args))
  do.call(what, args, quote, envir)
}

get_legacy_seed <- function() {
  opt <- getOption("laserf.legacy.seed", default = FALSE)
  if (!is.logical(opt) || length(opt) != 1) {
    stop("laserf option `laserf.legacy.seed` should be either TRUE or FALSE.")
  }

  opt
}
