#pragma once

#include <Eigen/Core>
#include <vector>
template <template<typename ...> class ContainerT, typename T, typename ...Args>
using aligned = ContainerT<T, Args..., Eigen::aligned_allocator<T> >;

template <typename T>
using aligned_vector = aligned<std::vector, T>;
