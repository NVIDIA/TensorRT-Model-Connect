// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const assert = require("node:assert/strict");
const { mkdtemp, rm, writeFile } = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const { imageSizeFromFile } = require("./fromFile.js");

async function withFixture(contents, callback) {
  const root = await mkdtemp(path.join(os.tmpdir(), "trtmc-image-size-"));
  const fixture = path.join(root, "fixture.img");
  try {
    await writeFile(fixture, contents);
    await callback(fixture);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test("reads explicit SVG dimensions", async () => {
  await withFixture(
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1508"></svg>',
    async (fixture) => {
      assert.deepEqual(await imageSizeFromFile(fixture), {
        height: 1508,
        width: 1200,
        type: "svg",
      });
    },
  );
});

test("uses a valid viewBox when explicit dimensions are absent", async () => {
  await withFixture('<svg viewBox="0 0 64 32"></svg>', async (fixture) => {
    assert.deepEqual(await imageSizeFromFile(fixture), {
      height: 32,
      width: 64,
      type: "svg",
    });
  });
});

test("rejects unsupported binary formats without looping", async () => {
  const vulnerableSamples = [
    Buffer.from([0x69, 0x63, 0x6e, 0x73, 0, 0, 0, 8, 0x69, 0x63, 0x30, 0, 0, 0, 0, 0]),
    Buffer.from([0, 0, 0, 0, 0x6a, 0x78, 0x6c, 0x70]),
    Buffer.from([0, 0, 0, 0, 0x69, 0x73, 0x70, 0x65]),
  ];
  for (const sample of vulnerableSamples) {
    await withFixture(sample, async (fixture) => {
      await assert.rejects(imageSizeFromFile(fixture), /Only SVG images are supported/);
    });
  }
});

test("rejects oversized inputs before reading them", async () => {
  await withFixture(Buffer.alloc(1024 * 1024 + 1, 0x20), async (fixture) => {
    await assert.rejects(imageSizeFromFile(fixture), /SVG size must be between/);
  });
});
