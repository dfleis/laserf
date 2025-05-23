// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// compute_split_frequencies
Rcpp::NumericMatrix compute_split_frequencies(const Rcpp::List& forest_object, size_t max_depth);
RcppExport SEXP _laserf_compute_split_frequencies(SEXP forest_objectSEXP, SEXP max_depthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_depth(max_depthSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_split_frequencies(forest_object, max_depth));
    return rcpp_result_gen;
END_RCPP
}
// compute_weights
Eigen::SparseMatrix<double> compute_weights(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, const Rcpp::NumericMatrix& test_matrix, unsigned int num_threads);
RcppExport SEXP _laserf_compute_weights(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP test_matrixSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type test_matrix(test_matrixSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_weights(forest_object, train_matrix, test_matrix, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// compute_weights_oob
Eigen::SparseMatrix<double> compute_weights_oob(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, unsigned int num_threads);
RcppExport SEXP _laserf_compute_weights_oob(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(compute_weights_oob(forest_object, train_matrix, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// merge
Rcpp::List merge(const Rcpp::List& forest_objects);
RcppExport SEXP _laserf_merge(SEXP forest_objectsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_objects(forest_objectsSEXP);
    rcpp_result_gen = Rcpp::wrap(merge(forest_objects));
    return rcpp_result_gen;
END_RCPP
}
// multi_causal_train
Rcpp::List multi_causal_train(const Rcpp::NumericMatrix& train_matrix, const std::vector<size_t>& outcome_index, const std::vector<size_t>& treatment_index, size_t sample_weight_index, bool use_sample_weights, const std::vector<double>& gradient_weights, unsigned int mtry, unsigned int num_trees, unsigned int min_node_size, double sample_fraction, bool honesty, double honesty_fraction, bool honesty_prune_leaves, size_t ci_group_size, double alpha, double imbalance_penalty, bool stabilize_splits, std::vector<size_t> clusters, unsigned int samples_per_cluster, bool compute_oob_predictions, unsigned int num_threads, unsigned int seed, bool legacy_seed);
RcppExport SEXP _laserf_multi_causal_train(SEXP train_matrixSEXP, SEXP outcome_indexSEXP, SEXP treatment_indexSEXP, SEXP sample_weight_indexSEXP, SEXP use_sample_weightsSEXP, SEXP gradient_weightsSEXP, SEXP mtrySEXP, SEXP num_treesSEXP, SEXP min_node_sizeSEXP, SEXP sample_fractionSEXP, SEXP honestySEXP, SEXP honesty_fractionSEXP, SEXP honesty_prune_leavesSEXP, SEXP ci_group_sizeSEXP, SEXP alphaSEXP, SEXP imbalance_penaltySEXP, SEXP stabilize_splitsSEXP, SEXP clustersSEXP, SEXP samples_per_clusterSEXP, SEXP compute_oob_predictionsSEXP, SEXP num_threadsSEXP, SEXP seedSEXP, SEXP legacy_seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< const std::vector<size_t>& >::type outcome_index(outcome_indexSEXP);
    Rcpp::traits::input_parameter< const std::vector<size_t>& >::type treatment_index(treatment_indexSEXP);
    Rcpp::traits::input_parameter< size_t >::type sample_weight_index(sample_weight_indexSEXP);
    Rcpp::traits::input_parameter< bool >::type use_sample_weights(use_sample_weightsSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type gradient_weights(gradient_weightsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type min_node_size(min_node_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type sample_fraction(sample_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty(honestySEXP);
    Rcpp::traits::input_parameter< double >::type honesty_fraction(honesty_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty_prune_leaves(honesty_prune_leavesSEXP);
    Rcpp::traits::input_parameter< size_t >::type ci_group_size(ci_group_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type imbalance_penalty(imbalance_penaltySEXP);
    Rcpp::traits::input_parameter< bool >::type stabilize_splits(stabilize_splitsSEXP);
    Rcpp::traits::input_parameter< std::vector<size_t> >::type clusters(clustersSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type samples_per_cluster(samples_per_clusterSEXP);
    Rcpp::traits::input_parameter< bool >::type compute_oob_predictions(compute_oob_predictionsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type legacy_seed(legacy_seedSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_causal_train(train_matrix, outcome_index, treatment_index, sample_weight_index, use_sample_weights, gradient_weights, mtry, num_trees, min_node_size, sample_fraction, honesty, honesty_fraction, honesty_prune_leaves, ci_group_size, alpha, imbalance_penalty, stabilize_splits, clusters, samples_per_cluster, compute_oob_predictions, num_threads, seed, legacy_seed));
    return rcpp_result_gen;
END_RCPP
}
// multi_causal_predict
Rcpp::List multi_causal_predict(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, const Rcpp::NumericMatrix& test_matrix, size_t num_outcomes, size_t num_treatments, unsigned int num_threads, bool estimate_variance);
RcppExport SEXP _laserf_multi_causal_predict(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP test_matrixSEXP, SEXP num_outcomesSEXP, SEXP num_treatmentsSEXP, SEXP num_threadsSEXP, SEXP estimate_varianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type test_matrix(test_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_outcomes(num_outcomesSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_treatments(num_treatmentsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< bool >::type estimate_variance(estimate_varianceSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_causal_predict(forest_object, train_matrix, test_matrix, num_outcomes, num_treatments, num_threads, estimate_variance));
    return rcpp_result_gen;
END_RCPP
}
// multi_causal_predict_oob
Rcpp::List multi_causal_predict_oob(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, size_t num_outcomes, size_t num_treatments, unsigned int num_threads, bool estimate_variance);
RcppExport SEXP _laserf_multi_causal_predict_oob(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP num_outcomesSEXP, SEXP num_treatmentsSEXP, SEXP num_threadsSEXP, SEXP estimate_varianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_outcomes(num_outcomesSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_treatments(num_treatmentsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< bool >::type estimate_variance(estimate_varianceSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_causal_predict_oob(forest_object, train_matrix, num_outcomes, num_treatments, num_threads, estimate_variance));
    return rcpp_result_gen;
END_RCPP
}
// multi_regression_train
Rcpp::List multi_regression_train(const Rcpp::NumericMatrix& train_matrix, const std::vector<size_t>& outcome_index, size_t sample_weight_index, bool use_sample_weights, unsigned int mtry, unsigned int num_trees, unsigned int min_node_size, double sample_fraction, bool honesty, double honesty_fraction, bool honesty_prune_leaves, double alpha, double imbalance_penalty, std::vector<size_t>& clusters, unsigned int samples_per_cluster, bool compute_oob_predictions, unsigned int num_threads, unsigned int seed, bool legacy_seed);
RcppExport SEXP _laserf_multi_regression_train(SEXP train_matrixSEXP, SEXP outcome_indexSEXP, SEXP sample_weight_indexSEXP, SEXP use_sample_weightsSEXP, SEXP mtrySEXP, SEXP num_treesSEXP, SEXP min_node_sizeSEXP, SEXP sample_fractionSEXP, SEXP honestySEXP, SEXP honesty_fractionSEXP, SEXP honesty_prune_leavesSEXP, SEXP alphaSEXP, SEXP imbalance_penaltySEXP, SEXP clustersSEXP, SEXP samples_per_clusterSEXP, SEXP compute_oob_predictionsSEXP, SEXP num_threadsSEXP, SEXP seedSEXP, SEXP legacy_seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< const std::vector<size_t>& >::type outcome_index(outcome_indexSEXP);
    Rcpp::traits::input_parameter< size_t >::type sample_weight_index(sample_weight_indexSEXP);
    Rcpp::traits::input_parameter< bool >::type use_sample_weights(use_sample_weightsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type min_node_size(min_node_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type sample_fraction(sample_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty(honestySEXP);
    Rcpp::traits::input_parameter< double >::type honesty_fraction(honesty_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty_prune_leaves(honesty_prune_leavesSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type imbalance_penalty(imbalance_penaltySEXP);
    Rcpp::traits::input_parameter< std::vector<size_t>& >::type clusters(clustersSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type samples_per_cluster(samples_per_clusterSEXP);
    Rcpp::traits::input_parameter< bool >::type compute_oob_predictions(compute_oob_predictionsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type legacy_seed(legacy_seedSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_regression_train(train_matrix, outcome_index, sample_weight_index, use_sample_weights, mtry, num_trees, min_node_size, sample_fraction, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty, clusters, samples_per_cluster, compute_oob_predictions, num_threads, seed, legacy_seed));
    return rcpp_result_gen;
END_RCPP
}
// multi_regression_predict
Rcpp::List multi_regression_predict(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, const Rcpp::NumericMatrix& test_matrix, size_t num_outcomes, unsigned int num_threads);
RcppExport SEXP _laserf_multi_regression_predict(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP test_matrixSEXP, SEXP num_outcomesSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type test_matrix(test_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_outcomes(num_outcomesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_regression_predict(forest_object, train_matrix, test_matrix, num_outcomes, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// multi_regression_predict_oob
Rcpp::List multi_regression_predict_oob(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, size_t num_outcomes, unsigned int num_threads);
RcppExport SEXP _laserf_multi_regression_predict_oob(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP num_outcomesSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_outcomes(num_outcomesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(multi_regression_predict_oob(forest_object, train_matrix, num_outcomes, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// regression_train
Rcpp::List regression_train(const Rcpp::NumericMatrix& train_matrix, size_t outcome_index, size_t sample_weight_index, bool use_sample_weights, unsigned int mtry, unsigned int num_trees, unsigned int min_node_size, double sample_fraction, bool honesty, double honesty_fraction, bool honesty_prune_leaves, size_t ci_group_size, double alpha, double imbalance_penalty, std::vector<size_t> clusters, unsigned int samples_per_cluster, bool compute_oob_predictions, unsigned int num_threads, unsigned int seed, bool legacy_seed);
RcppExport SEXP _laserf_regression_train(SEXP train_matrixSEXP, SEXP outcome_indexSEXP, SEXP sample_weight_indexSEXP, SEXP use_sample_weightsSEXP, SEXP mtrySEXP, SEXP num_treesSEXP, SEXP min_node_sizeSEXP, SEXP sample_fractionSEXP, SEXP honestySEXP, SEXP honesty_fractionSEXP, SEXP honesty_prune_leavesSEXP, SEXP ci_group_sizeSEXP, SEXP alphaSEXP, SEXP imbalance_penaltySEXP, SEXP clustersSEXP, SEXP samples_per_clusterSEXP, SEXP compute_oob_predictionsSEXP, SEXP num_threadsSEXP, SEXP seedSEXP, SEXP legacy_seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type outcome_index(outcome_indexSEXP);
    Rcpp::traits::input_parameter< size_t >::type sample_weight_index(sample_weight_indexSEXP);
    Rcpp::traits::input_parameter< bool >::type use_sample_weights(use_sample_weightsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type min_node_size(min_node_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type sample_fraction(sample_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty(honestySEXP);
    Rcpp::traits::input_parameter< double >::type honesty_fraction(honesty_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type honesty_prune_leaves(honesty_prune_leavesSEXP);
    Rcpp::traits::input_parameter< size_t >::type ci_group_size(ci_group_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type imbalance_penalty(imbalance_penaltySEXP);
    Rcpp::traits::input_parameter< std::vector<size_t> >::type clusters(clustersSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type samples_per_cluster(samples_per_clusterSEXP);
    Rcpp::traits::input_parameter< bool >::type compute_oob_predictions(compute_oob_predictionsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type legacy_seed(legacy_seedSEXP);
    rcpp_result_gen = Rcpp::wrap(regression_train(train_matrix, outcome_index, sample_weight_index, use_sample_weights, mtry, num_trees, min_node_size, sample_fraction, honesty, honesty_fraction, honesty_prune_leaves, ci_group_size, alpha, imbalance_penalty, clusters, samples_per_cluster, compute_oob_predictions, num_threads, seed, legacy_seed));
    return rcpp_result_gen;
END_RCPP
}
// regression_predict
Rcpp::List regression_predict(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, size_t outcome_index, const Rcpp::NumericMatrix& test_matrix, unsigned int num_threads, unsigned int estimate_variance);
RcppExport SEXP _laserf_regression_predict(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP outcome_indexSEXP, SEXP test_matrixSEXP, SEXP num_threadsSEXP, SEXP estimate_varianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type outcome_index(outcome_indexSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type test_matrix(test_matrixSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type estimate_variance(estimate_varianceSEXP);
    rcpp_result_gen = Rcpp::wrap(regression_predict(forest_object, train_matrix, outcome_index, test_matrix, num_threads, estimate_variance));
    return rcpp_result_gen;
END_RCPP
}
// regression_predict_oob
Rcpp::List regression_predict_oob(const Rcpp::List& forest_object, const Rcpp::NumericMatrix& train_matrix, size_t outcome_index, unsigned int num_threads, bool estimate_variance);
RcppExport SEXP _laserf_regression_predict_oob(SEXP forest_objectSEXP, SEXP train_matrixSEXP, SEXP outcome_indexSEXP, SEXP num_threadsSEXP, SEXP estimate_varianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type forest_object(forest_objectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type train_matrix(train_matrixSEXP);
    Rcpp::traits::input_parameter< size_t >::type outcome_index(outcome_indexSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< bool >::type estimate_variance(estimate_varianceSEXP);
    rcpp_result_gen = Rcpp::wrap(regression_predict_oob(forest_object, train_matrix, outcome_index, num_threads, estimate_variance));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_laserf_compute_split_frequencies", (DL_FUNC) &_laserf_compute_split_frequencies, 2},
    {"_laserf_compute_weights", (DL_FUNC) &_laserf_compute_weights, 4},
    {"_laserf_compute_weights_oob", (DL_FUNC) &_laserf_compute_weights_oob, 3},
    {"_laserf_merge", (DL_FUNC) &_laserf_merge, 1},
    {"_laserf_multi_causal_train", (DL_FUNC) &_laserf_multi_causal_train, 23},
    {"_laserf_multi_causal_predict", (DL_FUNC) &_laserf_multi_causal_predict, 7},
    {"_laserf_multi_causal_predict_oob", (DL_FUNC) &_laserf_multi_causal_predict_oob, 6},
    {"_laserf_multi_regression_train", (DL_FUNC) &_laserf_multi_regression_train, 19},
    {"_laserf_multi_regression_predict", (DL_FUNC) &_laserf_multi_regression_predict, 5},
    {"_laserf_multi_regression_predict_oob", (DL_FUNC) &_laserf_multi_regression_predict_oob, 4},
    {"_laserf_regression_train", (DL_FUNC) &_laserf_regression_train, 20},
    {"_laserf_regression_predict", (DL_FUNC) &_laserf_regression_predict, 6},
    {"_laserf_regression_predict_oob", (DL_FUNC) &_laserf_regression_predict_oob, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_laserf(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
