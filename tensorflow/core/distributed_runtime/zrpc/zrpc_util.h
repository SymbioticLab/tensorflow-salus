/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_UTIL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_UTIL_H_

#include "tensorflow/core/lib/core/status.h"

#include "zmq.hpp"

#include <memory>
#include <vector>

namespace zmq {

class MultiPartMessage
{
public:
    MultiPartMessage() = default;
    MultiPartMessage(MultiPartMessage &&other) : m_parts(std::move(other.m_parts)) {};
    MultiPartMessage(std::vector<message_t> *ptr) : m_parts(std::move(*ptr)) {};
    MultiPartMessage(const MultiPartMessage&) = delete;

    MultiPartMessage &operator=(const MultiPartMessage&) = delete;
    MultiPartMessage &operator=(MultiPartMessage &&other)
    {
        m_parts = std::move(other.m_parts);
        return *this;
    }

    MultiPartMessage &merge(MultiPartMessage &&other)
    {
        if (m_parts.empty()) {
            m_parts = std::move(other.m_parts);
        } else {
            m_parts.reserve(m_parts.size() + other.m_parts.size());
            std::move(std::begin(other.m_parts), std::end(other.m_parts), std::back_inserter(m_parts));
            other.m_parts.clear();
        }
        return *this;
    }

    MultiPartMessage clone()
    {
        MultiPartMessage mpm;
        for (auto &m : m_parts) {
            mpm->emplace_back();
            mpm->back().copy(&m);
        }
        return mpm;
    }

    std::vector<message_t> *release()
    {
        auto ptr = new std::vector<zmq::message_t>(std::move(m_parts));
        return ptr;
    }

    std::vector<message_t> *operator->()
    {
        return &m_parts;
    }

    std::vector<message_t> &messages()
    {
        return m_parts;
    }

    template<typename ... Args>
    message_t &emplace_back(Args&&... args)
    {
        m_parts.emplace_back(std::forward<Args>(args)...);
        return m_parts.back();
    }

private:
    std::vector<message_t> m_parts;
};

} // namespace zmq

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_UTIL_H_
