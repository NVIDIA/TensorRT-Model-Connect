/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Diagram({src, alt, caption, sequence = false}) {
  const resolvedSrc = useBaseUrl(src);
  const instanceId = React.useId().replaceAll(/[^a-zA-Z0-9_-]/g, '-');
  const captionId = `diagram-${instanceId}-caption`;
  const imageClassName = sequence
    ? 'trtmc-diagram__image trtmc-diagram__image--sequence'
    : 'trtmc-diagram__image';

  return (
    <figure className="trtmc-diagram trtmc-diagram--wide">
      <div
        className="trtmc-diagram__media"
        tabIndex={0}
        aria-label={`Scrollable diagram: ${alt}`}
      >
        <img
          className={imageClassName}
          src={resolvedSrc}
          alt={alt}
          aria-describedby={caption ? captionId : undefined}
          loading="lazy"
          decoding="async"
        />
      </div>
      {caption ? <figcaption id={captionId}>{caption}</figcaption> : null}
    </figure>
  );
}
