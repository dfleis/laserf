#' Print a LASERF forest object.
#' @param x The tree to print.
#' @param decay.exponent A tuning parameter that controls the importance of split depth.
#' @param max.depth The maximum depth of splits to consider.
#' @param ... Additional arguments (currently ignored).
#'
#' @method print laserf
#' @export
print.laserf <- function(x, decay.exponent = 2, max.depth = 4, ...) {
  var.importance <- variable_importance(x, decay.exponent, max.depth)
  var.importance <- c(round(var.importance, 3))
  names(var.importance) <- 1:length(var.importance)
  
  main.class <- class(x)[1]
  num.samples <- nrow(x$X.orig)
  
  cat("LASERF forest object of type", main.class, "\n")
  cat("Number of trees:", x[["_num_trees"]], "\n")
  cat("Number of training samples:", num.samples, "\n")
  
  cat("Variable importance:", "\n")
  print(var.importance)
}
