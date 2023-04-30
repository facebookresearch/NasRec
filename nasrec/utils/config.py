"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# NOTE: On search, the "MAX_NUM_EMBEDDINGS" should be 0.5M. 
# On evalaution, you should uncomment the "*10000" to utilize full embedding.
MAX_NUM_EMBEDDINGS = 500000 * 10000
#--------------Criteo Embedding size---------------------------------
NUM_EMBEDDINGS_CRITEO = [1461, 584, 10131227, 2202609, 306, 25, 12518, 634, \
    4, 93146, 5684, 8351593, 3195, 28, 14993, 5461307, 11, 5653, 2174, 5, 7046548,
    19, 16, 286182, 106, 142573]
# Prime embedding table size.
NUM_EMBEDDINGS_CRITEO = [min(x, MAX_NUM_EMBEDDINGS) for x in NUM_EMBEDDINGS_CRITEO]

#--------------Avazu Embedding size---------------------------------
# Position-0: ID is trivial.
# Prime Embedding
NUM_EMBEDDINGS_AVAZU = [10000, 241, 8, 8, 4738, 7746, 27, 8553, 560, 37, 2686409, \
    6729487, 8252, 6, 5, 2627, 9, 10, 436, 5, 69, 173, 61]

NUM_EMBEDDINGS_AVAZU = [min(x, MAX_NUM_EMBEDDINGS) for x in NUM_EMBEDDINGS_AVAZU]

#-------------KDD Embedding size-------------------------------------
# NUM_EMBEDDINGS_KDD = [26274, 641708, 14848, 24122077, 1188090, 3735797, 2934102, 22023548, 4, 8]
NUM_EMBEDDINGS_KDD = [26274, 641708, 14848, 22122011, 1188090, 3735797, 2934102, 20004011, 4, 8]

NUM_EMBEDDINGS_KDD = [min(x, MAX_NUM_EMBEDDINGS) for x in NUM_EMBEDDINGS_KDD]

NUM_EMBEDDINGS_TEST = [100] * 26
