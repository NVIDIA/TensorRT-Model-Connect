# Documentation diagram sources

The documentation embeds checked-in SVG files from `static/img/diagrams/`.
Editable flow sources live here as Graphviz DOT (`*.dot.txt`), and editable
interaction sources use the deliberately small `*.sequence.json` schema. The
website therefore does not need a client-side diagram renderer. The `.txt`
suffix keeps DOT assets outside the repository's source-language legal-header
classifier; each DOT file still carries SPDX provenance in comments.

Every source must include accessible metadata:

```dot
// @title: Short diagram title
// @description: One sentence that describes the complete diagram.
```

A sequence source provides the same fields as JSON, followed by two to six
participants and a short ordered message list:

```json
{
  "title": "Build one bundle",
  "description": "The complete interaction in one sentence.",
  "participants": [
    {"id": "cli", "label": "Build CLI", "color": "api"},
    {"id": "builder", "label": "Builder", "color": "build"}
  ],
  "messages": [
    {"from": "cli", "to": "builder", "label": "Build engine"},
    {"from": "builder", "to": "cli", "label": "Return plan", "style": "return"}
  ]
}
```

Render or verify the SVG files from `website/`:

```bash
npm run diagrams:build
npm run diagrams:check
```

The renderer uses the exact `@viz-js/viz` version pinned in `package-lock.json`.
The generated SVGs are committed, so page loads do not execute a client-side
layout engine. `npm run build` first rerenders in check mode and fails if a
checked-in SVG is stale; it does not rewrite assets.

Embed an output through the shared component rather than raw Markdown image
syntax. It preserves a readable intrinsic width on small screens and makes the
diagram horizontally scrollable with a keyboard-focusable viewport:

```mdx
import Diagram from '@site/src/components/Diagram';

<Diagram
  src="/img/diagrams/architecture/build-route-selection.svg"
  alt="Build route from checkpoint to bundle"
  caption="The bundle is the handoff between build and runtime."
/>
```

Use the shared visual defaults below unless a diagram needs a deliberate
exception:

- 16-point or larger node text and short labels;
- left-to-right flow for user journeys and top-to-bottom flow for ownership;
- no more than six primary cards in one row;
- explicit phase labels for long sequences;
- light cards on a neutral background so the same SVG remains legible in both
  website color modes; and
- a nearby figure caption in the Markdown page explaining what to notice.
