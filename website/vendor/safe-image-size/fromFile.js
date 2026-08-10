/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const { open } = require("node:fs/promises");

const MAX_SVG_BYTES = 1024 * 1024;
const NUMBER = "[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?";
const DIMENSION_RE = new RegExp(`^(${NUMBER})(?:px)?$`, "i");
const VIEWBOX_RE = new RegExp(
  `^\\s*(${NUMBER})[\\s,]+(${NUMBER})[\\s,]+(${NUMBER})[\\s,]+(${NUMBER})\\s*$`,
  "i",
);

function positiveFinite(value, field) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new TypeError(`SVG ${field} must be a positive finite number`);
  }
  return parsed;
}

function attribute(openingTag, name) {
  const match = openingTag.match(
    new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "i"),
  );
  return match ? (match[1] ?? match[2]) : undefined;
}

function explicitDimension(openingTag, name) {
  const raw = attribute(openingTag, name);
  if (raw === undefined) {
    return undefined;
  }
  const match = raw.trim().match(DIMENSION_RE);
  if (!match) {
    return undefined;
  }
  return positiveFinite(match[1], name);
}

function svgDimensions(source) {
  const prefix = source.slice(0, MAX_SVG_BYTES).toString("utf8");
  const start = prefix.search(/<svg(?:\s|>)/i);
  if (start < 0) {
    throw new TypeError("Only SVG images are supported by this documentation site");
  }
  const end = prefix.indexOf(">", start);
  if (end < 0) {
    throw new TypeError("SVG opening tag is incomplete");
  }

  const openingTag = prefix.slice(start, end + 1);
  const width = explicitDimension(openingTag, "width");
  const height = explicitDimension(openingTag, "height");
  if (width !== undefined && height !== undefined) {
    return { height, width, type: "svg" };
  }

  const viewBox = attribute(openingTag, "viewBox");
  const match = viewBox?.match(VIEWBOX_RE);
  if (!match) {
    throw new TypeError("SVG must declare positive width/height or a valid viewBox");
  }
  return {
    height: positiveFinite(match[4], "viewBox height"),
    width: positiveFinite(match[3], "viewBox width"),
    type: "svg",
  };
}

async function imageSizeFromFile(path) {
  const file = await open(path, "r");
  try {
    const stats = await file.stat();
    if (!stats.isFile()) {
      throw new TypeError("Image path must identify a regular file");
    }
    if (stats.size <= 0 || stats.size > MAX_SVG_BYTES) {
      throw new TypeError(`SVG size must be between 1 and ${MAX_SVG_BYTES} bytes`);
    }

    const source = Buffer.alloc(stats.size);
    const { bytesRead } = await file.read(source, 0, stats.size, 0);
    if (bytesRead !== stats.size) {
      throw new Error("Could not read the complete SVG");
    }
    return svgDimensions(source);
  } finally {
    await file.close();
  }
}

module.exports = { imageSizeFromFile };
