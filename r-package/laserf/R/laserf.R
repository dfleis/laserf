#' Locally Adaptive Subspace Estimation Random Forest
#' 
#' TODO:
#'  * Which input parameters are still relevant?
#'  * How to think about `sample.weights` and `clusters` parameters?
#'  * How do deal with `NA` data?
#'  * Clean up validation of primary covariates `Y` (n by p dimensional).
#'  * Clean up validation of auxiliary covariates `X` (n by d dimension).
#'  * `target.rank` (rank r). Would be nice for this to be optional like in `prcomp` and `princomp`.
#'  * `target.rank` (rank r). Can we allow the option to tune r similar to the ridge penalty parameter in `grf::ll_regression_forest`?
#'  * `subspace_forest_train` C++ implementation
#'  * predict.subspace_forest S3 method (via `subspace_forest_predict` and `subspace_forest_predict_oob`)
#' 
#' 
#' @param X TODO... auxiliary covariates (n by d)
#' @param Y TODO... primary covariates (n by p)
#' @param target.rank TODO... target subspace dimension (rank r) 
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
laserf <- function(X, Y, target.rank,  
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
                   compute.oob.predictions = TRUE,
                   num.threads = NULL,
                   seed = runif(1, 0, .Machine$integer.max),
                   .env = NULL) {
  has.missing.values <- validate_X(X, allow.na = FALSE) # TODO: TRUE or FALSE?
  validate_sample_weights(sample.weights, X)            # TODO: How to think about weights and clusters
  Y <- validate_observations(Y, X, allow.matrix = TRUE) # TODO: The Y data must be a matrix (n-by-p)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)
  
  target.rank <- validate_target_rank(target.rank)
  
  data <- create_train_matrices(X, outcome = Y, sample.weights = sample.weights)
  args <- list(target.rank = target.rank,
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
  forest[["target.rank"]] <- target.rank
  
  
  # TODO
  # TODO For debugging during development
  # TODO
  if (is.environment(.env)) {
    .env$args <- args
    .env$data <- data
    .env$forest <- forest
  }
  
  return (forest)
}