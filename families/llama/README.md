# Ordered tokenizer splits

MiniCPM5 checkpoints use a sequence of isolated regular-expression splits before
ByteLevel conversion. Each step operates on the pieces from the preceding step.
For example, isolating a number makes the spaces before it trailing whitespace
for the next split. Combining the regexes into one scanner changes BPE boundaries.

The native tokenizer recognizes the published MiniCPM5 two-regex profile,
compiles it once at load and applies its steps in order. Existing variant
ambiguity checks remain in force. This path requires isolated, non-inverted
regex steps followed by ByteLevel with explicit `use_regex: false` and
`add_prefix_space: false`. Other multi-Split profiles are rejected.

ICU supplies regex execution. The letter and number sets are pinned to Unicode
16.0.0 rather than the installed ICU version, matching the Tokenizers 0.23.3
reference. Restricting the profile avoids silently accepting incompatible ICU
and Oniguruma regex dialects. The family-owned generator verifies the SHA256 of
[UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt):
`ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`.
The Unicode license is installed with the family notice.

Rebuild the family runtime to obtain corrected token IDs. ICU `uc` and `i18n`
are required. The bundle schema and public API are unchanged. Native token-ID
comparisons do not establish GPU generation, model quality or performance.
