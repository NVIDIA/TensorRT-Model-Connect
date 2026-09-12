---
title: Family-owned Runtime Configuration
---

A Task defines required inputs, outputs, and the execution lifecycle. A family
implements that Task and declares its optional runtime parameters in a field
table. The semantic Task SDK transports these parameters through the C API;
it does not require a shared registry entry for every parameter name.

Build inputs still use the existing Python `BuildRequest`. Families that have
not migrated to the semantic SDK keep their existing execution path until
their own migration.

## Choose the owner

| Need | Owner |
| --- | --- |
| Required input such as text, an image, or source frames | The typed Task request; the family implements its meaning. |
| Native batch or streaming execution | A separate Task interface, not a Config switch. |
| Optional sampling, limits, or model-specific controls | The family's Config field table and implementation. |
| Model identity, graph construction, weights, and build dependencies | The family's `support.py`, `model.py`, and dependency declarations. |
| Model-independent loading or resource control | The existing loader or control API. |

## Declare and bind parameters inside the family

Reuse `ConfigField` from `trtmc/internal/config.h`. Supported values are int64,
double, bool, string, and homogeneous lists of int64, double, or strings.

```cpp
static const trtmc::internal::ConfigField sampling[] = {
    {"temperature", trtmc::internal::ConfigKind::F64,
     trtmc::internal::ConfigValue{1.0}, "Sampling temperature"},
    {"max_new_tokens", trtmc::internal::ConfigKind::I64,
     std::nullopt, "Default derived from the available context"},
};

// Inside the family's IModel implementation:
std::vector<trtmc::internal::TaskInstance> task_bindings() override {
    return {trtmc::internal::bind<trtmc::internal::ITextContinuation>(
        *this, sampling)};
}
```

The field storage is static or model-owned, not a temporary vector returned
while constructing a binding. The Core snapshots the declared metadata during
load; the family must continue to use the same immutable field definitions.
The binding helper records the adjusted Task-interface address, not a copy of
the model or a global registration.

## Resolve defaults and validate behavior

The Core rejects unknown keys, duplicate keys, and incorrect value types before
calling the family. The family checks ranges, parameter combinations, and
input-dependent limits before changing execution state.

```cpp
auto temperature = trtmc::internal::config_get<double>(
    config, sampling, "temperature").value();
auto requested = trtmc::internal::config_get<std::int64_t>(
    config, sampling, "max_new_tokens");
auto limit = requested ? *requested : context_dependent_default(request);
validate_generation_settings(request, temperature, limit); // family-owned
```

An absent fixed default means the family decides from its input or context, or
rejects the request if it cannot choose a meaningful value. Explicit
`0`, `false`, empty strings, and empty lists are not absence.
`config_provided(config, name)` checks whether the caller supplied a value.
The Core does not insert defaults into the caller's Config.

Strings and lists in internal `ConfigView`, including values returned by
`config_get`, are borrowed. A stream or session must copy or parse everything
it retains before the start/create call returns.

## Call through the existing SDK and CLI

```cpp
auto text = model.task<trtmc::TextContinuation>();
auto fields = text.config_fields();
auto result = text.run({"Hello"}, {{"temperature", 0.8}});
```

The CLI uses the same discovered field metadata to parse `--set name=value`.
A parameter does not need a new dedicated CLI flag. Programs using the C API
pass the existing typed name/value entries; the family chooses which names it
accepts.

Adding an optional key using an existing value type changes the owning family
and its tests, not the shared Task, C ABI, or wrapper. Adding a genuinely new
Task or value type is a shared-contract change and must be implemented and
tested at that boundary first. Do not silently ignore unsupported parameters
or retry another family or execution path.

The [config-registry status document](../context/config-registry-status.md)
records the retired registry; it is not the semantic SDK configuration design.
