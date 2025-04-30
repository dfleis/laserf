#' Locally Adaptive Subspace Estimation Random Forest
#' 
#' Fit a subspace forest
#' 
#' TODO:
#'  * Which input parameters are still relevant?
#'  * How to think about `sample.weights` and `clusters` parameters?
#'  * How do deal with `NA` data?
#'  * Clean up validation of primary covariates `Y` (n by p dimensional).
#'  * Clean up validation of auxiliary covariates `X` (n by d dimension).
#'  * `split.rank` (rank r). Would be nice for this to be optional like in `prcomp` and `princomp`.
#'  * `split.rank` (rank r). Can we allow the option to tune r similar to the ridge penalty parameter in `grf::ll_regression_forest`?
#'  * `subspace_forest_train` C++ implementation
#'  * predict.subspace_forest S3 method (via `subspace_forest_predict` and `subspace_forest_predict_oob`)
#' 
#' 
#' @param X TODO... auxiliary covariates (n by d)
#' @param Y TODO... primary covariates (n by p)
#' @param split.rank TODO... target subspace dimension during splitting (rank r) 
#' @param num.trees TODO...
#' @param sample.weights TODO...
#' @param clusters TODO...
#' @param equalize.cluster.weights TODO...
#' @param sample.fraction TODO...
#' @param mtry TODO... HIGH PRIORITY
#' @param min.node.size TODO... HIGH PRIORITY
#' @param honesty TODO...
#' @param honesty.fraction TODO...
#' @param honesty.prune.leaves TODO...
#' @param alpha TODO... HIGH PRIORITY
#' @param imbalance.penalty TODO... HIGH PRIORITY
#' @param compute.oob.predictions TODO...
#' @param num.threads TODO...
#' @param seed TODO...
#' @param .env Environment. Used during development as a container to help with debugging.
#'
#' @examples
#' \donttest{
#' # TODO...
#' }
#'
#' @export
laserf <- function(X, Y, split.rank,  
                   num.trees = 2000, # TODO: HIGH PRIORITY
                   sample.weights = NULL,
                   clusters = NULL,
                   equalize.cluster.weights = FALSE,
                   sample.fraction = 0.5,
                   mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)), 
                   min.node.size = 5, # TODO: HIGH PRIORITY
                   honesty = TRUE,
                   honesty.fraction = 0.5,
                   honesty.prune.leaves = TRUE,
                   alpha = 0.05,
                   imbalance.penalty = 0,
                   compute.oob.predictions = FALSE,
                   num.threads = NULL,
                   seed = runif(1, 0, .Machine$integer.max),
                   .env = NULL) {
  has.missing.values <- validate_X(X, allow.na = FALSE)
  validate_sample_weights(sample.weights, X)
  validate_Y(Y, allow.na = FALSE)
  Y <- validate_observations(Y, X, allow.matrix = TRUE)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)
  
  split.rank <- validate_rank(r = split.rank, num.features = NCOL(Y)) 
  
  data <- create_train_matrices(X, outcome = Y, sample.weights = sample.weights)
  args <- list(split.rank = split.rank,
               num.trees = num.trees,
               clusters = clusters,
               samples.per.cluster = samples.per.cluster,
               sample.fraction = sample.fraction,
               mtry = mtry,
               min.node.size = min.node.size,
               honesty = honesty,
               honesty.fraction = honesty.fraction,
               honesty.prune.leaves = honesty.prune.leaves,
               alpha = alpha,
               imbalance.penalty = imbalance.penalty,
               compute.oob.predictions = compute.oob.predictions,
               num.threads = num.threads,
               seed = seed,
               legacy.seed = get_legacy_seed())
  
  forest <- do.call.rcpp(subspace_forest_train, c(data, args))
  class(forest) <- c("subspace_forest", "laserf")
  forest[["seed"]] <- seed
  forest[["num.threads"]] <- num.threads
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["sample.weights"]] <- sample.weights
  forest[["clusters"]] <- clusters
  forest[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest[["has.missing.values"]] <- has.missing.values
  forest[["split.rank"]] <- split.rank
  
  if (is.environment(.env)) {
    # TODO Drop this in the final version
    message("DEVELOPMENT: Saving `laserf::laserf` args, data, and forest in user-supplied environment.")
    .env$args <- args
    .env$data <- data
    .env$forest <- forest
  }
  
  return (forest)
}

#' Predict with a subspace forest
#'
#' TODO
#'
#' @param object The trained forest.
#' @param newX TODO... C++ just takes the test auxiliary covariates X, but in R we can accept a test Y as well
#' @param newY TODO... See `newX`
#' @param rank TODO... Optional, defaults to the trained `split.rank`
#' @param compute.scores TODO... compute z = Vt y 
#' @param compute.projections TODO... compute yhat = Vz
#' @param num.threads Number of threads used in prediction. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A list containing `predictions`: a matrix of predictions for each outcome.
#'
#' @examples
#' \donttest{
#' # TODO ...
#' }
#'
#' @method predict laserf
#' @export
predict.laserf <- function(object, 
                           newX = NULL, 
                           newY = NULL,
                           rank = NULL,
                           compute.scores = FALSE,
                           compute.projections = FALSE,
                           num.threads = NULL,
                           ...) {
  num.threads <- validate_num_threads(num.threads)
  if (is.null(rank)) rank <- object[["split.rank"]]
  num.features <- NCOL(object[["Y.orig"]])
  rank <- validate_rank(r = rank, num.features = num.features)
  
  forest.short <- object[-which(names(object) == "X.orig")]
  train.data <- create_train_matrices(X = object[["X.orig"]], outcome = object[["Y.orig"]])

  args <- list(
    forest.object = forest.short,
    num.threads = num.threads,
    rank = rank
  )
  
  compute.oob.preds <- TRUE
  if (!is.null(newX)) {
    validate_newX(newX = newX, X = object[["X.orig"]], allow.na = FALSE)
    test.data <- create_test_matrices(newX)
    
    if (is.null(newY)) {
      compute.scores <- FALSE
      compute.projections <- FALSE
    } else {
      validate_newY(newY = newY, Y = object[["Y.orig"]], allow.na = FALSE)
    }
    
    ret <- do.call.rcpp(subspace_forest_predict, c(train.data, test.data, args))
    
    compute.oob.preds <- FALSE
  } else {
    if (!is.null(newY)) {
      stop("Cannot make predictions at new features `newY` without ",
           "supplying a corresponding set of auxiliary covariates `newX`.")
    } 
    
    ret <- do.call.rcpp(subspace_forest_predict_oob, c(train.data, args))
    compute.oob.preds <- TRUE
  }

  restructure_subspace_predictions(
    preds.raw    = ret$predictions, 
    num.features = num.features,
    rank         = rank,
    Y            = if (compute.oob.preds) object[["Y.orig"]] else newY, 
    compute.scores      = compute.scores,
    compute.projections = compute.projections
  )
} 

#' Restructure the vectorized subspace prediction vectors
#' @param preds.raw TODO...
#' @param num.features TODO... dimension of Y input feature space
#' @param rank TODO... rank
#' @param Y TODO...
#' @param compute.scores TODO... low-dimensional representations... Ignored if `compute.projections = TRUE`
#' @param compute.projections TODO... projections
#' @return TODO...
#' @keywords internal
restructure_subspace_predictions <- function(preds.raw, 
                                             num.features,
                                             rank,
                                             Y = NULL,
                                             compute.scores = FALSE,
                                             compute.projections = FALSE) {
  V.size <- rank * num.features
  idx.eigvecs <- 1:V.size
  idx.eigvals <- (V.size + 1):(V.size + rank)
  idx.means <- (V.size + rank + 1):(V.size + rank + num.features)
  
  format_preds <- function(pred, y) {
    V <- matrix(pred[idx.eigvecs], nrow = num.features, ncol = rank)
    eigvals <- pred[idx.eigvals]
    y.means <- pred[idx.means]
    
    z <- if (isTRUE(compute.scores) || isTRUE(compute.projections)) {
      as.vector(crossprod(V, y - y.means))
    } else {
      NULL
    }
      
    y.proj <- if (isTRUE(compute.projections) && !is.null(z)) {
      y.means + as.vector(V %*% z)
    } else {
      NULL
    }
    
    list(V = V, 
         eigvals = eigvals, 
         y.means = y.means, 
         z = z, 
         y.proj = y.proj)
  }
  
  out <- sapply(1:nrow(preds.raw), function(i) {
    format_preds(
      pred = preds.raw[i,], 
      y = if (!is.null(Y)) Y[i,] else NULL
    )
  })
  
  list(
    "eigenvectors"  = out["V",],
    "eigenvalues"   = t(simplify2array(out["eigvals",], except = 0L)),
    "feature.means" = t(simplify2array(out["y.means",])),
    "scores"        = if (compute.scores) t(simplify2array(out["z",], except = 0L)) else NULL,
    "projections"   = if (compute.projections) t(simplify2array(out["y.proj",])) else NULL
  )
}













