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
#include "forest/ForestTrainers.h"
#include "relabeling/SubspaceRelabelingStrategy.h"
#include "splitting/factory/MultiRegressionSplittingRuleFactory.h"

namespace grf {

ForestTrainer subspace_trainer(size_t num_features, 
                               size_t split_rank) {
  size_t response_length = num_features * split_rank; // Pseudo-response dimension
  
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new SubspaceRelabelingStrategy(num_features, split_rank, response_length));
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new MultiRegressionSplittingRuleFactory(response_length));
  
  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       nullptr);
}

} // namespace grf
