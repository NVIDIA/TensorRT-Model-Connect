# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cases where textual C++ extraction must preserve source or avoid overclaims."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "native_details", Path(__file__).with_name("collect-native-details.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class NativeDetailsTests(unittest.TestCase):
    def extract(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        repo = Path(directory.name)
        family = repo / "families" / "example"
        runtime = family / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "config.h").write_text(source)
        return _MODULE.collect_native_details(repo, family)

    def test_literals_templates_comments_and_source_positions(self):
        records = self.extract('''// struct FakeConfig { int absent{4}; };
struct RealConfig {
    // Maximum number of tokens.
    int32_t max_tokens{256};
    std::string pattern{"value; } /* literal */"};
    std::array<float, 3> mean{{0.1F, 0.2F, 0.3F}};
    std::vector<int32_t> tokens;
    static constexpr int limit = 1'024;
};
''')
        self.assertEqual([record["name"] for record in records], ["RealConfig"])
        fields = {field["name"]: field for field in records[0]["fields"]}
        self.assertEqual(set(fields), {"max_tokens", "pattern", "mean", "tokens", "limit"})
        self.assertEqual(fields["max_tokens"]["source"]["line"], 4)
        self.assertEqual(fields["max_tokens"]["description"], "Maximum number of tokens.")
        self.assertEqual(fields["pattern"]["default"], '"value; } /* literal */"')
        self.assertNotIn("literal", fields["pattern"]["description"])
        self.assertEqual(fields["mean"]["default"], "{0.1F, 0.2F, 0.3F}")
        self.assertIsNone(fields["tokens"]["default"])
        self.assertFalse(fields["tokens"]["required"])
        self.assertTrue(fields["limit"]["readOnly"])

    def test_complex_declarations_are_preserved_not_guessed(self):
        declaration = '''struct ComplexOptions {
    unsigned first : 2;
    int left, right;
    void (*callback)(int);
    int count{2};
    void reset() { count = 3; }
    bool enabled{false};
};'''
        record = self.extract(declaration)[0]
        self.assertEqual([field["name"] for field in record["fields"]], ["count", "enabled"])
        self.assertEqual(record["declaration"], declaration)
        self.assertEqual(len(record["unparsedDeclarations"]), 4)
        self.assertIn("not a list of CLI-adjustable options", record["description"])

    def test_raw_strings_forward_declarations_and_non_config_types(self):
        records = self.extract('''struct ForwardConfig;
enum class Options { kDefault, kOther };
struct NotConfigurationButConfigurable { int value; };
struct TensorShape { int width; };
class RuntimeOptions final {
public:
    std::string expression = R"tag(}; // raw string)tag";
    int enabled{1};
};
''')
        self.assertEqual([record["name"] for record in records], ["NotConfigurationButConfigurable", "RuntimeOptions"])
        self.assertEqual(records[-1]["fields"][0]["default"], 'R"tag(}; // raw string)tag"')
        self.assertEqual(records[-1]["fields"][1]["name"], "enabled")

    def test_nested_initializers_and_multiple_declarators(self):
        record = self.extract('''struct MixedConfig {
    int first{1}, second{2};
    std::array<int, 2> dimensions{{2, 3}};
    // Runtime-selected seed, not a measured support claim.
    int seed = choose_seed(4, 5);
    bool enabled{true}; // Enables processing.
    int steps{2};
};''')[0]
        self.assertEqual([field["name"] for field in record["fields"]], ["dimensions", "seed", "enabled", "steps"])
        self.assertEqual(record["fields"][1]["default"], "choose_seed(4, 5)")
        self.assertEqual(record["fields"][2]["description"], "Enables processing.")
        self.assertNotIn("Enables", record["fields"][3]["description"])
        self.assertEqual(record["unparsedDeclarations"][0]["declaration"], "int first{1}, second{2};")


if __name__ == "__main__":
    unittest.main()
