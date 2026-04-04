// Asymmetric q8_0 K + tbq3_0 V template instance

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_0);
