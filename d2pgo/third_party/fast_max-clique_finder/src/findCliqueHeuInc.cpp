// This file is modified from the original file at:
// https://github.com/MIT-SPARK/Kimera-RPGO/blob/master/include/KimeraRPGO/max_clique_finder/findCliqueHeu.cpp
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstddef>

#include "findClique.h"
namespace FMC {
int maxCliqueHeuIncremental(CGraphIO& gio, size_t num_new_lc,
                            size_t prev_maxclique_size,
                            vector<int>& max_clique_data) {
  vector<int>* p_v_i_Vertices = gio.GetVerticesPtr();
  vector<int>* p_v_i_Edges = gio.GetEdgesPtr();
  // srand(time(NULL));

  int maxDegree = gio.GetMaximumVertexDegree();
  // TODO initialize maxClq with the best so far from prev steps
  int maxClq = prev_maxclique_size, u, icc;
  vector<int> v_i_S;
  vector<int> v_i_S1;
  v_i_S.resize(maxDegree + 1, 0);
  v_i_S1.resize(maxDegree + 1, 0);

  int iPos, iPos1, iCount, iCount1;

  int notComputed = 0;

  // compute the max clique for each vertex
  // TODO tricky indexing ...
  for (size_t iCandidateVertex = p_v_i_Vertices->size() - num_new_lc - 1;
       iCandidateVertex < p_v_i_Vertices->size() - 1; iCandidateVertex++) {
    // Pruning 1
    if (maxClq > ((*p_v_i_Vertices)[iCandidateVertex + 1] -
                  (*p_v_i_Vertices)[iCandidateVertex])) {
      notComputed++;
      continue;
    }

    iPos = 0;
    v_i_S[iPos++] = iCandidateVertex;
    int z, imdv = iCandidateVertex,
           imd = (*p_v_i_Vertices)[iCandidateVertex + 1] -
                 (*p_v_i_Vertices)[iCandidateVertex];

    int iLoopCount = (*p_v_i_Vertices)[iCandidateVertex + 1];
    for (int j = (*p_v_i_Vertices)[iCandidateVertex]; j < iLoopCount; j++) {
      // Pruning 3
      if (maxClq <= ((*p_v_i_Vertices)[(*p_v_i_Edges)[j] + 1] -
                     (*p_v_i_Vertices)[(*p_v_i_Edges)[j]]))
        v_i_S[iPos++] = (*p_v_i_Edges)[j];
    }

    icc = 0;

    while (iPos > 0) {
      int imdv1 = -1, imd1 = -1;

      icc++;

      // generate a random number x from 0 to iPos -1
      // aasign that imdv = x and imd = d(x)
      // imdv = v_i_S[rand() % iPos];
      imdv = v_i_S[iPos - 1];
      imd = (*p_v_i_Vertices)[imdv + 1] - (*p_v_i_Vertices)[imdv];

      iPos1 = 0;

      for (int j = 0; j < iPos; j++) {
        iLoopCount = (*p_v_i_Vertices)[imdv + 1];
        for (int k = (*p_v_i_Vertices)[imdv]; k < iLoopCount; k++) {
          // Pruning 5
          if (v_i_S[j] == (*p_v_i_Edges)[k] &&
              maxClq <= ((*p_v_i_Vertices)[(*p_v_i_Edges)[k] + 1] -
                         (*p_v_i_Vertices)[(*p_v_i_Edges)[k]])) {
            v_i_S1[iPos1++] = v_i_S[j];  // calculate the max degree vertex here

            break;
          }
        }
      }

      for (int j = 0; j < iPos1; j++) v_i_S[j] = v_i_S1[j];

      iPos = iPos1;

      imdv = imdv1;
      imd = imd1;
    }

    if (maxClq < icc) {
      max_clique_data = v_i_S;
      maxClq = icc;
    }
  }

  return maxClq;
}
}  // namespace FMC