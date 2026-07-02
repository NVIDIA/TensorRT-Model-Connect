/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

function copyCmd(id) {
  var el = document.getElementById(id);
  if (!el) return;
  var text = el.textContent || el.innerText;
  navigator.clipboard.writeText(text).then(function() {
    var btn = el.parentElement.querySelector('.copy-btn');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(function() {
      btn.textContent = 'Copy'; }, 1500); }
  });
}
function filterModels() {
  var q = (document.getElementById('search-box').value || '').toLowerCase();
  var s = document.getElementById('status-filter').value;
  var rows = document.querySelectorAll('.summary-row');
  for (var i = 0; i < rows.length; i++) {
    var name = rows[i].getAttribute('data-name') || '';
    var status = rows[i].getAttribute('data-status') || '';
    var statuses = status.split(/\s+/);
    var show = (!q || name.indexOf(q) >= 0) &&
      (!s || statuses.indexOf(s) >= 0);
    rows[i].style.display = show ? '' : 'none';
  }
}
