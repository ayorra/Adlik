// Copyright 2019 ZTE corporation. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ADLIK_SERVING_RUNTIME_TENSORFLOW_LITE_TENSOR_SHAPE_DIMS_H
#define ADLIK_SERVING_RUNTIME_TENSORFLOW_LITE_TENSOR_SHAPE_DIMS_H

#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace adlik {
namespace serving {
class TensorShapeDims {
  std::vector<std::unique_ptr<tensorflow::TensorShapeProto_Dim>> dimStorage;
  std::unique_ptr<const tensorflow::TensorShapeProto_Dim*[]> dimsStorage;
  const tensorflow::TensorShapeProto_Dim* const* first;
  const tensorflow::TensorShapeProto_Dim* const* last;

  TensorShapeDims(std::vector<std::unique_ptr<tensorflow::TensorShapeProto_Dim>> dimStorage,
                  std::unique_ptr<const tensorflow::TensorShapeProto_Dim*[]> dimsStorage,
                  const tensorflow::TensorShapeProto_Dim* const* first,
                  const tensorflow::TensorShapeProto_Dim* const* last)
      : dimStorage(std::move(dimStorage)), dimsStorage(std::move(dimsStorage)), first(first), last(last) {
  }

public:
  bool operator==(const TensorShapeDims& rhs) const {
    return std::equal(this->first,
                      this->last,
                      rhs.first,
                      rhs.last,
                      [](const tensorflow::TensorShapeProto_Dim* l, const tensorflow::TensorShapeProto_Dim* r) {
                        return l->size() == r->size();
                      });
  }

  const tensorflow::TensorShapeProto_Dim* const* begin() const {
    return this->first;
  }

  const tensorflow::TensorShapeProto_Dim* const* end() const {
    return this->last;
  }

  template <class Iterator>
  static TensorShapeDims owned(Iterator first, Iterator last) {
    std::vector<std::unique_ptr<tensorflow::TensorShapeProto_Dim>> dimStorage;

    std::transform(first, last, std::back_inserter(dimStorage), [](const auto& dim) {
      auto result = std::make_unique<tensorflow::TensorShapeProto_Dim>();

      result->set_size(dim);

      return result;
    });

    const auto size = static_cast<size_t>(std::distance(first, last));
    auto dimsStorage = std::make_unique<const tensorflow::TensorShapeProto_Dim*[]>(size);

    std::transform(dimStorage.begin(), dimStorage.end(), dimsStorage.get(), [](const auto& dim) { return dim.get(); });

    const auto firstDims = dimsStorage.get();
    const auto lastDims = dimsStorage.get() + size;

    return TensorShapeDims{std::move(dimStorage), std::move(dimsStorage), firstDims, lastDims};
  }

  static TensorShapeDims borrowed(const tensorflow::TensorShapeProto_Dim* const* first,
                                  const tensorflow::TensorShapeProto_Dim* const* last) {
    return TensorShapeDims{{}, nullptr, first, last};
  }
};
}  // namespace serving
}  // namespace adlik

#endif  // ADLIK_SERVING_RUNTIME_TENSORFLOW_LITE_TENSOR_SHAPE_DIMS_H
