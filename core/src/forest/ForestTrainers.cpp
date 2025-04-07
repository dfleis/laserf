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
#include <Rcpp.h> // TODO debugging
 
#include "forest/ForestTrainers.h"
#include "prediction/MultiCausalPredictionStrategy.h"
#include "prediction/RegressionPredictionStrategy.h"
#include "prediction/MultiRegressionPredictionStrategy.h"
#include "relabeling/MultiCausalRelabelingStrategy.h"
#include "relabeling/NoopRelabelingStrategy.h"
#include "relabeling/MultiNoopRelabelingStrategy.h"
#include "relabeling/SubspaceRelabelingStrategy.h"
#include "splitting/factory/RegressionSplittingRuleFactory.h"
#include "splitting/factory/MultiCausalSplittingRuleFactory.h"
#include "splitting/factory/MultiRegressionSplittingRuleFactory.h"

namespace grf {

ForestTrainer multi_causal_trainer(size_t num_treatments,
                                   size_t num_outcomes,
                                   bool stabilize_splits,
                                   const std::vector<double>& gradient_weights) {
  size_t response_length = num_treatments * num_outcomes;
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new MultiCausalRelabelingStrategy(response_length, gradient_weights));
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory = stabilize_splits
    ? std::unique_ptr<SplittingRuleFactory>(new MultiCausalSplittingRuleFactory(response_length, num_treatments))
    : std::unique_ptr<SplittingRuleFactory>(new MultiRegressionSplittingRuleFactory(response_length));
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiCausalPredictionStrategy(num_treatments, num_outcomes));

  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}

ForestTrainer regression_trainer() {
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new NoopRelabelingStrategy());
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new RegressionSplittingRuleFactory());
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new RegressionPredictionStrategy());

  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}

ForestTrainer multi_regression_trainer(size_t num_outcomes) {
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new MultiNoopRelabelingStrategy(num_outcomes));
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new MultiRegressionSplittingRuleFactory(num_outcomes));
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiRegressionPredictionStrategy(num_outcomes));

  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}

ForestTrainer subspace_trainer(size_t num_outcomes, 
                               size_t split_rank) {
  // TODO Give num_outcomes, response_length, (and possibly split_rank) more descriptive names
  // TODO Should we implement an "OptimizedPredictionStrategy"? 
  // If I understand correctly, an OptimizedPredictionStrategy leads to a more time-efficient 
  // Stage II at the cost of a less memory-efficient forest. This is because an "OptimizedPredictionStrategy"
  // stores the node-wise summary statistics that are computed during the the splitting procedure
  // so that we do not need to re-compute them during Stage II.
  
  size_t response_length = num_outcomes * num_outcomes; // dimension of pseudo-outcomes
  
  Rcpp::Rcout << "ForestTrainers subspace_trainer(): num_outcomes (p) = " << num_outcomes << "\n";
  Rcpp::Rcout << "ForestTrainers subspace_trainer(): split_rank (r) = " << split_rank << "\n";
  Rcpp::Rcout << "ForestTrainers subspace_trainer(): response_length (p^2) = " << response_length << "\n";
  
  /* TODO
   * num_outcomes: dimensionality of primary covariates Y (p).
   * split_rank: target subspace dimensionality during the splitting mechanism (r).
   * response_length: pseudo-outcome dimensionality.
   * 
   * When the score \psi is expressed to target the projection matrix W, pseudo-outcomes will
   * be p-by-p dimensional, so response_length will be p^2 (at first glance).
   *  
   * Can we get away with vectorizing pseudo-outcomes? If not, the modification to GRF's codebase
   * will be much more substantial. This is because pseudo-outcomes are stored as a 2-dimensional 
   * array (row-wise observations, col-wise dimensions).
   */

  // std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiRegressionPredictionStrategy(response_length));
  // std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new MultiRegressionSplittingRuleFactory(num_outcomes));
  // std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new MultiRegressionPredictionStrategy(num_outcomes));
  // return ForestTrainer(std::move(relabeling_strategy),
  //                      std::move(splitting_rule_factory),
  //                      std::move(prediction_strategy));
  
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new SubspaceRelabelingStrategy(num_outcomes, split_rank, response_length));
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new MultiRegressionSplittingRuleFactory(response_length));
  
  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       nullptr);
}

} // namespace grf
