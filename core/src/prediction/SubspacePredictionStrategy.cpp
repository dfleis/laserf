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

#include <algorithm>
#include <vector>

#include "prediction/SubspacePredictionStrategy.h"

namespace grf {

SubspacePredictionStrategy::SubspacePredictionStrategy(size_t num_features,
                                                       size_t rank):
    num_features(num_features),
    rank(rank),
    pred_length(rank * num_features + rank + num_features) {}

size_t SubspacePredictionStrategy::prediction_length() const { 
  // I would have name this function to `get_prediction_length()` to be consistent 
  // with the relabeling strategy convention `get_response_length()`, but doing so 
  // involves modifying deeper parts of the general prediction backend.
  return pred_length;
}

std::vector<double> SubspacePredictionStrategy::predict(
    size_t sampleID,
    const std::unordered_map<size_t, double>& weights_by_sampleID,
    const Data& train_data,
    const Data& data) const {
  size_t num_nonzero_weights = weights_by_sampleID.size();
  
  std::vector<size_t> indices(num_nonzero_weights);
  Eigen::MatrixXd weights_vec = Eigen::VectorXd::Zero(num_nonzero_weights);
  { 
    size_t i = 0;
    for (auto& it : weights_by_sampleID) {
      size_t index = it.first;
      double weight = it.second;
      indices[i] = index;
      weights_vec(i) = weight;
      i++;
    }
  }
  
  // Step 1a: Form the training matrix Y and prepare the statistics to center/rescale.
  Eigen::MatrixXd Y_centered(num_nonzero_weights, num_features);
  Eigen::RowVectorXd Y_mean = Eigen::RowVectorXd::Zero(num_features);
  double sum_weight = 0.0; // I forgot if they've already been normalized to sum to 1.
  for (size_t i = 0; i < num_nonzero_weights; ++i) {
    Eigen::VectorXd y = train_data.get_outcomes(indices[i]);
    sum_weight += weights_vec(i);
    Y_mean += weights_vec(i) * y;
    Y_centered.row(i) = y;
  }
  Y_mean /= sum_weight;
  
  // Step 1b: Center and rescale Y
  // TODO Can this be optimized by doing it incrementally during the first scan over the samples?
  for (size_t i = 0; i < num_nonzero_weights; ++i) { 
    double scale_factor = std::sqrt(weights_vec(i) / sum_weight);
    Y_centered.row(i) = scale_factor * (Y_centered.row(i) - Y_mean);
  }
  
  // Step 2: Spectral decomposition
  // IMPORTANT: Eigen::SelfAdjointEigenSolver arranges the eigenvalues in increasing order.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
  Eigen::VectorXd eigenvalues(rank); 
  Eigen::MatrixXd V(num_features, rank);
  
  if (num_nonzero_weights < num_features) {
    // For n < d, use the kernel trick: Rather than directly computing Y^T Y, we
    // instead start by computing the eigenvectors of the outer product Y Y^T.
    Eigen::MatrixXd kernel_matrix = Y_centered * Y_centered.transpose();
    eigen_solver.compute(kernel_matrix, Eigen::ComputeEigenvectors);
    
    // Get top-r eigenvalues and corresponding eigenvectors.
    eigenvalues = eigen_solver.eigenvalues().tail(rank);
    Eigen::MatrixXd kernel_eigenvectors = eigen_solver.eigenvectors().rightCols(rank);
    
    // Convert to eigenvectors of the covariance matrix using the relation V = Y^T U / sqrt(lambda)
    for (size_t k = 0; k < rank; ++k) {
      V.col(k) = Y_centered.transpose() * kernel_eigenvectors.col(k);
      if (eigenvalues(k) > 1e-10) {  // Numerical stability check
        V.col(k) /= std::sqrt(eigenvalues(k));
      }
    }
  } else {
    // For n >= d, directly compute covariance matrix.
    Eigen::MatrixXd covariance = Y_centered.transpose() * Y_centered;
    eigen_solver.compute(covariance, Eigen::ComputeEigenvectors);
    
    eigenvalues = eigen_solver.eigenvalues().tail(rank);
    V = eigen_solver.eigenvectors().rightCols(rank);
  }
  
 /**
  * Populate the prediction vector for this sample. This will be the vectorized `V` matrix
  * (to be re-formed as a matrix in R) followed by the local mean `Y_mean`. Because
  * Eigen::SelfAdjointEigenSolver arranges its eigenvalues in INCREASING order, we scan
  * through the columns of `V` in reverse so that it more aligns with how R orders 
  * its eigenvalues/eigenvectors returned by `base::eigen()` and `base::svd()`.
  *
  * We first place the vectorized columns of `V` into the `predictions`, followed by
  * the mean vector `Y_mean`.
  */
  std::vector<double> predictions(pred_length);
  
  // Populate predictions with eigenvectors from V in reversed column order
  size_t pos = 0;
  for (size_t k = rank; k-- > 0; ) {
    for (size_t j = 0; j < num_features; ++j) {
      predictions[pos++] = V(j, k);
    }
  }
  
  // Append the eigenvalues
  for (size_t k = rank; k-- > 0; ) {
    predictions[pos++] = eigenvalues(k);
  }

  // Append the feature-space mean vector
  for (size_t j = 0; j < num_features; ++j) {
    predictions[pos++] = Y_mean(j);
  }
  
  return predictions;
}


std::vector<double> SubspacePredictionStrategy::compute_variance(
    size_t sampleID,
    const std::vector<std::vector<size_t>>& samples_by_tree,
    const std::unordered_map<size_t, double>& weights_by_sampleID,
    const Data& train_data,
    const Data& data,
    size_t ci_group_size) const {
  return { 0.0 };
}

} // namespace grf