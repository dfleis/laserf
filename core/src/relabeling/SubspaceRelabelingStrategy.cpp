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
#include "commons/utility.h"
#include "relabeling/SubspaceRelabelingStrategy.h"
 
namespace grf {

SubspaceRelabelingStrategy::SubspaceRelabelingStrategy(size_t num_outcomes,
                                                       size_t split_rank,
                                                       size_t response_length) :
  num_outcomes(num_outcomes),
  split_rank(split_rank),
  response_length(response_length) {}

bool SubspaceRelabelingStrategy::relabel(
    const std::vector<size_t>& samples,
    const Data& data,
    Eigen::ArrayXXd& responses_by_sample) const {

      size_t num_samples = samples.size();
      size_t num_outcomes = data.get_num_outcomes();
      
      // stop if fewer parent samples than dimensions
      // TODO Verify that we want to do this? Should it be num_samples <= split_rank?
      if (num_samples <= num_outcomes) { 
        return true;
      }
    
      // TODO Verify that we want to center Y over P?
      // TODO Efficient subspace solver built-in to Eigen?
      Eigen::MatrixXd Y_centered = Eigen::MatrixXd(num_samples, num_outcomes);
      Eigen::VectorXd weights = Eigen::VectorXd(num_samples); // optional user-specified sample weights (sample.weights)
      Eigen::VectorXd Y_mean = Eigen::VectorXd::Zero(num_outcomes);
      double sum_weight = 0;
      for (size_t i = 0; i < num_samples; i++) {
        size_t sample = samples[i];
        double weight = data.get_weight(sample);
        Eigen::VectorXd outcome = data.get_outcomes(sample);
        Y_centered.row(i) = outcome;
        weights(i) = weight;
        Y_mean += weight * outcome;
        sum_weight += weight;
      }
      Y_mean /= sum_weight;
      Y_centered.rowwise() -= Y_mean.transpose();
    
      if (std::abs(sum_weight) <= 1e-16) { 
        return true;
      }
    
      // TODO Compute pseudo-outcomes \rho_i (vectorized?)
      // e.g. if we're using the projection matrix form of the score then 
      // \rho_i = (I - \widehat W_P) Y_i Y_i^\top
      // Note that (I - \widehat W_P) is common to all i 
      
      for (size_t sample : samples) { 
        // TODO placeholder
        responses_by_sample.row(sample) = Eigen::VectorXd::Random(response_length);
    }
    return false;
  }
 
 size_t SubspaceRelabelingStrategy::get_response_length() const {
   return response_length;
 }
 
 } // namespace grf
 