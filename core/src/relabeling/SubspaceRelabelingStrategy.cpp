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

SubspaceRelabelingStrategy::SubspaceRelabelingStrategy(size_t split_rank,
                                                       size_t response_length) :
  split_rank(split_rank),
  response_length(response_length) {}

bool SubspaceRelabelingStrategy::relabel(
    const std::vector<size_t>& samples,
    const Data& data,
    Eigen::ArrayXXd& responses_by_sample) const {

      size_t num_samples = samples.size(); // n_P (parent samples)
      size_t num_features = data.get_num_outcomes(); // d
      
      // TODO
      // TODO Verify that we want to do this? 
      // TODO Should it be num_samples <= split_rank or num_features?
      // TODO
      // Stop if fewer parent samples than the dimension r of the target subspace
      if (num_samples <= split_rank) { 
        return true; 
      }
    
      // Step 1a: Calculate the sample statistics needed to center and rescale/normalize Y.
      Eigen::MatrixXd Y_centered = Eigen::MatrixXd(num_samples, num_features);
      Eigen::VectorXd weights = Eigen::VectorXd(num_samples); // observational weights (via optional `sample.weights`)
      Eigen::RowVectorXd Y_mean = Eigen::RowVectorXd::Zero(num_features);
      double sum_weight = 0.0;
      for (size_t i = 0; i < num_samples; ++i) {
        size_t sample = samples[i];
        Eigen::VectorXd y = data.get_outcomes(sample);
        double weight = data.get_weight(sample);
        sum_weight += weight;
        Y_mean += weight * y;
        Y_centered.row(i) = y;
        weights(i) = weight;
      }
      // Check whether only weight-zero samples were drawn.
      if (std::abs(sum_weight) <= 1e-16) { 
        return true;
      }
      Y_mean /= sum_weight;
      
      // Step 1b: Rescale/normalize the observations of Y by sqrt(weights(i)/sum_weights).
      // When sample weights are left as the default, this is equivalent to 1/sqrt(n).
      for (size_t i = 0; i < num_samples; ++i) {
        double scale_factor = std::sqrt(weights(i) / sum_weight);
        Y_centered.row(i) = scale_factor * (Y_centered.row(i) - Y_mean);
      }
    
      // Step 2: Calculate eigenvectors (efficient self-adjoint eigendecomposition).
      // Note that SelfAdjointEigenSolver sorts the eigenvalues in increasing order.
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
      Eigen::VectorXd eigenvalues(split_rank); 
      Eigen::MatrixXd V(num_features, split_rank);
      
      if (num_samples < num_features) {
        // For n < d, use the kernel trick: Rather than directly computing Y^T Y, we
        // instead start by computing the eigenvectors of the outer product Y Y^T.
        Eigen::MatrixXd kernel_matrix = Y_centered * Y_centered.transpose();
        eigen_solver.compute(kernel_matrix, Eigen::ComputeEigenvectors);
        
        // Get top-r eigenvalues and corresponding eigenvectors.
        eigenvalues = eigen_solver.eigenvalues().tail(split_rank);
        Eigen::MatrixXd kernel_eigenvectors = eigen_solver.eigenvectors().rightCols(split_rank);
        
        // Convert to eigenvectors of the covariance matrix using the relation V = Y^T U / sqrt(lambda)
        for (size_t k = 0; k < split_rank; ++k) {
          V.col(k) = Y_centered.transpose() * kernel_eigenvectors.col(k);
          if (eigenvalues(k) > 1e-10) {  // Numerical stability check
            V.col(k) /= std::sqrt(eigenvalues(k));
          }
        }
      } else {
        // For n >= d, directly compute covariance matrix.
        Eigen::MatrixXd covariance = Y_centered.transpose() * Y_centered;
        eigen_solver.compute(covariance, Eigen::ComputeEigenvectors);
      
        eigenvalues = eigen_solver.eigenvalues().tail(split_rank);
        V = eigen_solver.eigenvectors().rightCols(split_rank);
      }
    
     /**
      * Vectorize d-by-r dimensional matrix pseudo-responses to (rd)-dimensional vectors.
      * Mathematically, if Y_i is d-by-1 and V is d-by-r then 
      *    z_i := V^\top Y_i   (r-by-1)
      *    e_i := Y_i - V z_i  (d-by-1)
      * and then the d-by-r matrix-valued pseudo-response is
      *    \rho_i := e_i z_i^\top  (d-by-r)
      * The vectorized version is formed by arranging the columns of \rho_i side-by-side. If R_i
      * denotes a (rd)-dimensional vectorized pseudo-response then, for k = 1,..., r,
      *    R_i[(k - 1)d, ..., kd] = z_{i,k} \cdot e_i.
      */
      Eigen::RowVectorXd z(split_rank); 
      Eigen::RowVectorXd e(num_features);
      for (size_t i = 0; i < num_samples; ++i) {
        size_t sample = samples[i];
        
        z = Y_centered.row(i) * V; 
        e = Y_centered.row(i) - z * V.transpose(); 
        
        for (size_t k = 0; k < split_rank; ++k) {
         /**
          * For a matrix M, the function call M.block(i, j, p, q) selects the submatrix of M
          * according to
          *   i: Starting row index (indexing from 0 for the first row).
          *   j: Starting column index (indexing from 0 for the first column).
          *   p: Number of rows to select, starting from the i-th row (inclusive).
          *   q: Number of columns to select, starting from the j-th column (inclusive).
          */
          responses_by_sample.block(sample, k * num_features, 1, num_features) = z(k) * e;
        }
      }
      return false;
  }
 
 size_t SubspaceRelabelingStrategy::get_response_length() const {
   return response_length;
 }
 
 } // namespace grf
 