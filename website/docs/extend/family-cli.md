---
title: Family-Owned CLI Commands
---

Each migrated family declares its commands in `families/<family>/cli.json`.
The Python and native applications read that data to discover commands,
parse arguments and display help. The selected family owns the handler and
the meaning of every argument. Adding a command does not require a central
command registry or another field in the shared `BuildRequest`.

## Ownership boundary

The family owns the command handler as well as its declaration. It decides
which runtime operations to call, how to compose them, how to prepare inputs
and how to present results. A handler can reuse the existing Task SDK without
moving these decisions into the public CLI.

The shared dispatchers only discover descriptions, validate the description
format, parse declared types, render help and invoke the selected handler.
They must not contain family IDs, business command or option definitions,
model defaults, or command-to-Task mappings. The native host depends on the
generic CLI entry-point contract; Task SDK calls belong inside owner handlers.

Adding a family command, option or workflow must require only the family's
declaration, handler and tests. A new Task contract or a new generic CLI value
type can require a separate shared-contract change. The existing flat CLI is
a temporary compatibility path; new family features must use the owner path.

## Discover commands without loading a model

```bash
python -m tensorrt_model_connect bert --help
python -m tensorrt_model_connect boltz2 prepare-structure --help
trtmc bert encode --help
```

Help reads installed descriptions. It does not download a checkpoint, import
TensorRT or load a family/backend DSO. The native application can display both
Python and native command descriptions without a Python interpreter. Executing
a Python command requires the builder package in the selected Python environment.

The initial owners are BERT and Boltz2. Other families retain their existing
entry points. CLI ownership is independent of migration to the semantic Task
SDK: existing Task bindings, Config field tables and public C ABI remain the
runtime contracts.

## Declare the command inside its family

```json
{
  "version": 1,
  "commands": [
    {
      "name": "prepare",
      "help": "Prepare this family's input",
      "executor": "python",
      "handler": "cli:prepare",
      "arguments": [
        {"name": "input", "type": "path", "help": "Input document"},
        {"name": "output", "flags": ["-o", "--output"], "type": "path", "required": true}
      ]
    }
  ]
}
```

This example describes `trtmc <family> prepare INPUT --output OUTPUT`. The
Python handler is `families.<family>.cli.prepare`, called with keyword arguments
named by the declaration. It returns an integer exit status. Import heavy
dependencies inside the selected handler; a build handler selects its backend
before importing a model module that imports TensorRT.

An argument without `flags` is positional. Supported scalar types are `string`,
`path`, `int`, `float` and `bool`. Integers use the signed 64-bit range and
floating-point values must be finite. Boolean values are `true` or `false`;
`action: "store_true"` declares a switch, and `action: "append"` declares a
repeatable option. Optional fields include `required`, `default`, `choices`
and `help`. Omit a default when the family must distinguish absence from an
explicit value. Unknown declaration fields, duplicate names/flags, unknown
arguments and unsupported versions are errors.

Keep cross-field constraints and input-dependent defaults in the owner. A
family-local typed request gives the builder a narrow input contract. The
parser only interprets declared types; it does not know model option names or
perform graph, precision or cache policy decisions.

## Native handlers

For `executor: "native"`, `handler` identifies an operation within the selected
family. Its `libtrtmc_cli_<family>.so` adapter supplies the internal
`trtmc_family_cli_v1` entry point
from `trtmc/internal/cli.h`. The CLI loads this entry point only for execution,
passes the parsed values and output callbacks, and keeps the DSO alive for the
call. The family converts values to its existing typed runtime operations and
reports errors without allowing exceptions to cross the entry point.

This internal CLI entry point ships with the matching runtime and family DSOs.
The adapter calls the existing runtime loader; the model DSO keeps its original
dependency boundary. It does not add operations to the public semantic Task ABI.
A family using
semantic Tasks continues to use its existing Config declarations and validation;
its CLI options do not become shared Task fields.

CMake discovers `families/*/cli.json` automatically. Build trees and packaged
native binaries carry descriptions under `families/<family>/cli.json` beside
the executable; normal installations also provide them under
`share/trtmc/families/`. The wheel retains the original family description.
Benchmark build identities include its contents, so changing a declared
default invalidates the corresponding cached build.

## Migrate one family

After the shared CLI change is merged, subsequent command migrations should
change only `families/<family>/**`. There is no central owner registration or
per-family exception to add. Keep these contracts together in that change:

- A `cli.json` declaration and its lazy Python handlers. A Python module
  referenced only by the declaration is a valid entry point; it does not need
  an artificial import from `model.py` or a test merely to satisfy reachability.
- For native commands, the owner CMake file builds `trtmc_cli_<family>` with
  output `libtrtmc_cli_<family>.so` alongside the model DSO and supplies its
  install rule. Selected-family CI discovers that target from the declaration.
- Build commands used by Benchmark declare logical `model` and `output`
  arguments. Their CLI spellings, including positional output, remain owned by
  the family. Benchmark uses the same declaration for temporary and final
  output paths, and includes the declaration in its build cache identity.
- Owner tests preserve existing behavior and test their declared inputs. A
  source/CPU onboarding rehearsal proves integration mechanics, not checkpoint
  accuracy, GPU modes or performance; those remain each family's qualification.

1. Declare its actual commands and supported arguments, preserving defaults and
   existing workloads. A task name alone does not imply a command is supported.
2. Add its lazy Python/native handlers and narrow build request. Keep graphs,
   checkpoint interpretation, preprocessing and output semantics in the family.
3. Update owner tests and consumers together. Verify help without model
   dependencies, invalid/unsupported arguments, one selected execution, and
   the existing numerical and lifecycle criteria.
4. Verify the installed description and selected-family runtime layout. A
   source-tree help test alone does not prove the packaged CLI works.

During migration, undeclared families keep the existing entry points. A declared
command with an invalid description or missing handler fails explicitly; it
never retries the old path. Existing Python callers of a migrated family may
use an owner-local compatibility conversion that preserves unsupported-value
rejection. New options must only extend the owner declaration and request.

After all families migrate, remove the old fixed command tables and shared
request union in a separate cleanup. No central list tracks migration status.
