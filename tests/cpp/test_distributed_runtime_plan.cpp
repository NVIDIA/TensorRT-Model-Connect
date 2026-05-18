#include "trtmc/runtime/distributed_runtime.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_parse_tensor_parallel_plan() {
    const std::string plan = R"json(
{
  "schema_version": "1.0",
  "mesh": {
    "world_size": 2,
    "axes": {"tp": 2, "pp": 1, "cp": 1, "dp": 1, "ep": 1}
  },
  "bundle_sections": {
    "decoder": {
      "rank_section_pattern": "decoder_rank{rank}_plan"
    }
  }
}
)json";

    const auto cfg = trtmc::parse_distributed_plan_runtime_config(plan, "decoder");
    check(cfg.enabled, "plan should enable distributed runtime");
    check(cfg.world_size == 2, "world_size parsed");
    check(cfg.tp_size == 2, "tp_size parsed");
    check(cfg.pp_size == 1 && cfg.cp_size == 1 && cfg.dp_size == 1 && cfg.ep_size == 1,
          "non-TP axes parsed as one");
    check(cfg.rank_section_pattern == "decoder_rank{rank}_plan",
          "rank section pattern parsed");
    check(trtmc::distributed_rank_section_name(cfg.rank_section_pattern, 1) ==
              "decoder_rank1_plan",
          "rank section replacement");
}

void test_reject_non_tp_axis_for_initial_runtime() {
    const std::string plan = R"json(
{
  "schema_version": "1.0",
  "mesh": {
    "world_size": 4,
    "axes": {"tp": 2, "pp": 1, "cp": 1, "dp": 2, "ep": 1}
  },
  "bundle_sections": {
    "decoder": {
      "rank_section_pattern": "engine_plan_rank{rank}"
    }
  }
}
)json";

    bool threw = false;
    try {
        (void)trtmc::parse_distributed_plan_runtime_config(plan, "decoder");
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("only tensor-parallel") != std::string::npos;
    }
    check(threw, "non-TP mesh axes should be rejected until MeshRuntime supports them");
}

void test_missing_component_is_reported() {
    const std::string plan = R"json(
{
  "schema_version": "1.0",
  "mesh": {
    "world_size": 1,
    "axes": {"tp": 1, "pp": 1, "cp": 1, "dp": 1, "ep": 1}
  },
  "bundle_sections": {
    "encoder": {
      "section": "engine_plan"
    }
  }
}
)json";

    bool threw = false;
    try {
        (void)trtmc::parse_distributed_plan_runtime_config(plan, "decoder");
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("decoder") != std::string::npos;
    }
    check(threw, "missing component should name the missing component");
}

void test_disabled_mesh_runtime_group_is_default() {
    trtmc::MeshRuntimeConfig cfg;
    cfg.enabled = false;
    const auto group = trtmc::initialize_mesh_runtime_group(cfg);
    check(group.world_size == 1, "disabled mesh runtime should use world_size=1");
    check(group.rank == 0, "disabled mesh runtime should use rank=0");
    check(group.communicator == nullptr, "disabled mesh runtime should not create communicator");
}

} // namespace

int main() {
    test_parse_tensor_parallel_plan();
    test_reject_non_tp_axis_for_initial_runtime();
    test_missing_component_is_reported();
    test_disabled_mesh_runtime_group_is_default();
    std::cerr << "All distributed runtime plan tests passed\n";
    return 0;
}
