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

#ifndef GRF_SUBSPACEPREDICTIONSTRATEGY_H
#define GRF_SUBSPACEPREDICTIONSTRATEGY_H


#include <cstddef>
#include <unordered_map>
#include "commons/Data.h"
#include "prediction/DefaultPredictionStrategy.h"

namespace grf {

class SubspacePredictionStrategy final: public DefaultPredictionStrategy {

public:
  SubspacePredictionStrategy(size_t num_features,
                             size_t rank);

  size_t prediction_length() const;

  std::vector<double> predict(size_t sampleID,
                              const std::unordered_map<size_t, double>& weights_by_sampleID,
                              const Data& train_data,
                              const Data& data) const;
    
  std::vector<double> compute_variance(
      size_t sampleID,
      const std::vector<std::vector<size_t>>& samples_by_tree,
      const std::unordered_map<size_t, double>& weights_by_sampleID,
      const Data& train_data,
      const Data& data,
      size_t ci_group_size) const;

private:
  size_t num_features; // Input feature space dimension d
  size_t rank;         // Target subspace dimension/rank r
 /**
  * Given a test covariate $x$ in the auxiliary covariate space, we want to return the
  *   1) fitted local eigenvectors $\widehat V(x)$, (d-by-r dimensional)
  *   2) corresponding local eigenvalues (r-dimensional), 
  *   3) fitted local feature-space means $\overline Y := \overline Y(x)$ (d-dimensional).
  * 
  * The GRF architecture is built around the fitted predictions being of type `std::vector<double>`. 
  * To fit the local subspace model into this framework we vectorize the eigenvector matrix `V`
  * (column-by-column), then append the eigenvalues and the feature space mean. This means that
  * for a single test point $x$, the `predictions` vector returned by `predict()` will be
  * $(rd + r + d)$-dimensional.
  * 
  * Note that the predicted scores $z := V^\top (y - \bar y)$ and the predicted projections 
  * $\hat y := \bar y + Vz$ also require a test value of $y$, whereas the eigenvectors, eigenvalues
  * and feature-space mean vector only require a test value of $x$. The C++ prediction backend 
  * does not compute the predictions that require a test value of $y$, and we save calculating
  * these quantities (the scores and projections) once we're back in R.
  */
  size_t pred_length; 
};

} // namespace grf

#endif //GRF_SUBSPACEPREDICTIONSTRATEGY_H