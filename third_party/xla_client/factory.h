#pragma once

#include <memory>

namespace torch_xla {

template <typename T>
class GenericFactory {
public:
    virtual std::unique_ptr<T> create() = 0;
};

}  // namespace torch_xla
