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

#ifndef GRF_SUBSPACERELABELINGSTRATEGY_H
#define GRF_SUBSPACERELABELINGSTRATEGY_H
 
#include <vector>
 
#include "commons/Data.h"
#include "relabeling/RelabelingStrategy.h"
#include "tree/Tree.h"
 
namespace grf {

class SubspaceRelabelingStrategy final: public RelabelingStrategy {
public:
  SubspaceRelabelingStrategy(size_t num_outcomes, 
                             size_t split_rank,
                             size_t response_length);
  bool relabel(
      const std::vector<size_t>& samples,
      const Data& data,
      Eigen::ArrayXXd& responses_by_sample) const;
  
  size_t get_response_length() const;
    
private:
  size_t num_outcomes; // input dimensionality p
  size_t split_rank;   // target subspace dimensional/rank r
  size_t response_length; // projection matrix dimensionality p * p (can we reduce pseudo-outcome dimension to p-by-r?)
 };
 
} // namespace grf
 
#endif // GRF_SUBSPACERELABELINGSTRATEGY_H