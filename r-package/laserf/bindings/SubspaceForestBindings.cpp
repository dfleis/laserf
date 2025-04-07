/*-------------------------------------------------------------------------------
  Copyright (c) 2024 GRF Contributors.

  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <Rcpp.h>
#include <vector>

#include "commons/globals.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "RcppUtilities.h"

using namespace grf;

// [[Rcpp::export]]
Rcpp::List subspace_forest_train(const Rcpp::NumericMatrix& train_matrix,
                                 const std::vector<size_t>& outcome_index,
                                 size_t split_rank,
                                 size_t sample_weight_index,
                                 bool use_sample_weights,
                                 unsigned int mtry,
                                 unsigned int num_trees,
                                 unsigned int min_node_size,
                                 double sample_fraction,
                                 bool honesty,
                                 double honesty_fraction,
                                 bool honesty_prune_leaves,
                                 double alpha,
                                 double imbalance_penalty,
                                 std::vector<size_t>& clusters,
                                 unsigned int samples_per_cluster,
                                 bool compute_oob_predictions,
                                 unsigned int num_threads,
                                 unsigned int seed,
                                 bool legacy_seed) {
  Data data = RcppUtilities::convert_data(train_matrix);
  data.set_outcome_index(outcome_index);
  if (use_sample_weights) {
    data.set_weight_index(sample_weight_index);
  }

  size_t ci_group_size = 1;
  ForestOptions options(num_trees, ci_group_size, sample_fraction, mtry, min_node_size, honesty,
      honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty, num_threads, seed, legacy_seed, clusters, samples_per_cluster);

  // NEW
  // TODO Subspace forest pipeline 
  // TODO Complete
  Rcpp::Rcout << "SubspaceForestBindings: Initializing `ForestTrainer trainer = subspace_trainer(...)`\n";
  ForestTrainer trainer = subspace_trainer(data.get_num_outcomes(), split_rank);
  Rcpp::Rcout << "SubspaceForestBindings: Successfully initialized `trainer`\n";
  
  // Rcpp::Function sys_sleep("Sys.sleep");
  // Rcpp::Rcout << "Sleep 1\n";
  // sys_sleep(1);
  Rcpp::Rcout << "SubspaceForestBindings: Training `Forest forest = trainer.train(...)`\n";
  Forest forest = trainer.train(data, options);
  Rcpp::Rcout << "SubspaceForestBindings: Successfully trained `forest`\n";
  // std::vector<Prediction> predictions;
  // if (compute_oob_predictions) {
  //   ForestPredictor predictor = multi_regression_predictor(num_threads, data.get_num_outcomes());
  //   //ForestPredictor predictor = multi_regression_predictor(num_threads, data.get_num_outcomes() * data.get_num_outcomes());
  //   predictions = predictor.predict_oob(forest, data, false);
  // }
  //
  // return RcppUtilities::create_forest_object(forest, predictions);
  
  
  // OLD (multi_regression_trainer used as a template)
  // TODO Multivariate response regression forest pipeline
  // TODO Remove
  Rcpp::Rcout << "SubspaceForestBindings: Running `multi_regression_forest` pipeline (TODO Remove)\n";
  ForestTrainer trainer_OLD = multi_regression_trainer(data.get_num_outcomes());
  Forest forest_OLD = trainer_OLD.train(data, options);
  std::vector<Prediction> predictions_OLD;
  if (compute_oob_predictions) {
    ForestPredictor predictor_OLD = multi_regression_predictor(num_threads, data.get_num_outcomes());
    predictions_OLD = predictor_OLD.predict_oob(forest_OLD, data, false);
  }

  Rcpp::Rcout << "SubspaceForestBindings: Returning trained multi_regression_forest (and optionally, OOB predictions)\n";
  return RcppUtilities::create_forest_object(forest_OLD, predictions_OLD);
}

// [[Rcpp::export]]
Rcpp::List subspace_forest_predict(const Rcpp::List& forest_object,
                                   const Rcpp::NumericMatrix& train_matrix,
                                   const Rcpp::NumericMatrix& test_matrix,
                                   size_t num_outcomes,
                                   unsigned int num_threads) {
  // TODO laserf implementation
  // TODO laserf implementation
  // TODO laserf implementation
  Data train_data = RcppUtilities::convert_data(train_matrix);

  Data data = RcppUtilities::convert_data(test_matrix);
  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  bool estimate_variance = false;
  ForestPredictor predictor = multi_regression_predictor(num_threads, num_outcomes);
  std::vector<Prediction> predictions = predictor.predict(forest, train_data, data, estimate_variance);

  return RcppUtilities::create_prediction_object(predictions);
}

// [[Rcpp::export]]
Rcpp::List subspace_forest_predict_oob(const Rcpp::List& forest_object,
                                       const Rcpp::NumericMatrix& train_matrix,
                                       size_t num_outcomes,
                                       unsigned int num_threads) {
  // TODO laserf implementation
  // TODO laserf implementation
  // TODO laserf implementation
  Data data = RcppUtilities::convert_data(train_matrix);

  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  bool estimate_variance = false;
  ForestPredictor predictor = multi_regression_predictor(num_threads, num_outcomes);
  std::vector<Prediction> predictions = predictor.predict_oob(forest, data, estimate_variance);

  Rcpp::List result = RcppUtilities::create_prediction_object(predictions);
  return result;
}
