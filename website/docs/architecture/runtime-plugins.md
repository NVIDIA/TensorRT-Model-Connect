---
title: Family Runtime DSOs
---

The pre-#1093 runtime-plugin registry no longer exists. The stable load unit is
one family DSO named `libtrtmc_model_<family>.so`.

The bundle header names exactly one family and backend. The loader opens that
family DSO from the explicit runtime root, resolves `trtmc_create_family`, and
receives an implementation of an abstract Task interface. A family owns its
factory, pipeline, preprocessing, postprocessing, dispatch, bindings, state,
samplers, and any genuinely model-specific TensorRT plugin.

There is no registrar macro, runtime-strategy map, central manifest, sibling
fallback, or hot plugin marketplace. To extend an existing family, edit only
its `families/<family>/runtime/` implementation and family-owned tests. To add
a new family, follow [Add a Model Family](../extend/add-model-family.md).
