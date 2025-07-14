#pragma once

#include <utility>
#include <glm/glm.hpp>

template<typename VecType>
using VecPair = std::pair<VecType, VecType>;
using DVecPair = VecPair<glm::dvec3>;

namespace glm {
    typedef vec<4, unsigned char, defaultp> ucvec4;
}
