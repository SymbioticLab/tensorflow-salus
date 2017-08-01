/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORIES
#define TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORIES

#include "tensorflow/core/framework/allocator.h"

#include <memory>
#include <functional>

namespace tensorflow {
class WrappedDeviceSettings
{
public:
    typedef std::function<std::unique_ptr<Allocator>(Allocator*)> AllocatorFactory;

    static void setWrapperFactory(AllocatorFactory fact);
    static Allocator* getWrapped(Allocator* alloc);

private:
    static AllocatorFactory m_allocatorFactory;
};

} // namespace tensorflow

#endif // TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORIES
