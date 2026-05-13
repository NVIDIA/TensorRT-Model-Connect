#pragma once

#include "trtmc/config/schema_registry.h"

namespace trtmc::config::schemas {

Schema make_parallel_schema();
void register_parallel_schema(::trtmc::config::SchemaRegistry& registry);

} // namespace trtmc::config::schemas
