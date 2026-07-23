/* ============================================================
   markdown.js —— 安全 Markdown 子集渲染器
   支持：标题 / 粗体 / 斜体 / 无序与有序列表 / 代码块 /
         行内代码 / 表格 / 链接
   安全策略：先整体转义 HTML，再渲染标记；链接仅允许
   http(s)://、mailto:、/ 与 # 开头，其余按纯文本处理。
   暴露全局函数 renderMarkdown(src) -> HTML 字符串
   ============================================================ */
(function () {
  'use strict';

  var SENTINEL = '\u0001';

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function isSafeUrl(url) {
    return /^(https?:\/\/|mailto:|\/|#)/i.test(url);
  }

  /* 行内格式：行内代码、链接、粗体、斜体（输入已转义） */
  function renderInline(escaped) {
    // 1. 行内代码先占位提取，避免内部内容被后续规则误处理
    var codeSpans = [];
    var s = escaped.replace(/`([^`\n]+)`/g, function (m, code) {
      codeSpans.push(code);
      return SENTINEL + (codeSpans.length - 1) + SENTINEL;
    });

    // 2. 链接 [text](url)
    s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, function (m, text, url) {
      if (!isSafeUrl(url)) return text;
      return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + text + '</a>';
    });

    // 3. 粗体 **x** / __x__
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');

    // 4. 斜体 *x* / _x_
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/(^|[^\w])_([^_\n]+)_(?=[^\w]|$)/g, '$1<em>$2</em>');

    // 5. 还原行内代码
    var restoreRe = new RegExp(SENTINEL + '(\\d+)' + SENTINEL, 'g');
    s = s.replace(restoreRe, function (m, i) {
      return '<code>' + codeSpans[Number(i)] + '</code>';
    });

    return s;
  }

  function splitRow(line) {
    var t = line.trim();
    if (t.charAt(0) === '|') t = t.slice(1);
    if (t.charAt(t.length - 1) === '|') t = t.slice(0, -1);
    return t.split('|').map(function (c) { return c.trim(); });
  }

  function isTableSeparator(line) {
    // 形如 |---|---| 或 ---|---（允许对齐冒号）
    return /^\|?[\s:|-]*-+[\s:|-]*(\|[\s:|-]*-+[\s:|-]*)*\|?\s*$/.test(line)
      && line.indexOf('-') !== -1;
  }

  function isTableStart(lines, i) {
    return i + 1 < lines.length
      && lines[i].indexOf('|') !== -1
      && isTableSeparator(lines[i + 1]);
  }

  function isListLine(line) {
    return /^\s*(?:[-*+]|\d{1,9}[.)])\s+/.test(line);
  }

  function isOrdered(line) {
    return /^\s*\d{1,9}[.)]\s+/.test(line);
  }

  /**
   * 渲染 Markdown 子集为安全 HTML。
   * @param {string} src 原始 markdown 文本
   * @returns {string} HTML
   */
  function renderMarkdown(src) {
    var raw = String(src == null ? '' : src).replace(/\r\n/g, '\n');
    var lines = escapeHtml(raw).split('\n');
    var out = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      // 空行
      if (!line.trim()) { i++; continue; }

      // 代码块 ``` ... ```
      if (/^\s*```/.test(line)) {
        var buf = [];
        i++;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
          buf.push(lines[i]);
          i++;
        }
        if (i < lines.length) i++; // 跳过收尾 ```
        out.push('<pre><code>' + buf.join('\n') + '</code></pre>');
        continue;
      }

      // 标题 # .. ######
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        var lvl = h[1].length;
        out.push('<h' + lvl + '>' + renderInline(h[2]) + '</h' + lvl + '>');
        i++;
        continue;
      }

      // 表格
      if (isTableStart(lines, i)) {
        var headers = splitRow(lines[i]);
        i += 2; // 跳过表头与分隔行
        var rows = [];
        while (i < lines.length && lines[i].trim() && lines[i].indexOf('|') !== -1) {
          rows.push(splitRow(lines[i]));
          i++;
        }
        var th = headers.map(function (c) { return '<th>' + renderInline(c) + '</th>'; }).join('');
        var trs = rows.map(function (r) {
          return '<tr>' + r.map(function (c) { return '<td>' + renderInline(c) + '</td>'; }).join('') + '</tr>';
        }).join('');
        out.push('<table><thead><tr>' + th + '</tr></thead><tbody>' + trs + '</tbody></table>');
        continue;
      }

      // 列表（连续的列表行，按有序/无序分组）
      if (isListLine(line)) {
        var ordered = isOrdered(line);
        var tag = ordered ? 'ol' : 'ul';
        var items = [];
        while (i < lines.length && isListLine(lines[i]) && isOrdered(lines[i]) === ordered) {
          items.push('<li>' + renderInline(lines[i].replace(/^\s*(?:[-*+]|\d{1,9}[.)])\s+/, '')) + '</li>');
          i++;
        }
        out.push('<' + tag + '>' + items.join('') + '</' + tag + '>');
        continue;
      }

      // 普通段落：合并连续的非特殊行，行间以 <br> 连接
      var para = [];
      while (i < lines.length && lines[i].trim()
        && !/^\s*```/.test(lines[i])
        && !/^(#{1,6})\s+/.test(lines[i])
        && !isListLine(lines[i])
        && !isTableStart(lines, i)) {
        para.push(renderInline(lines[i]));
        i++;
      }
      if (para.length) out.push('<p>' + para.join('<br>') + '</p>');
    }

    return out.join('\n');
  }

  window.renderMarkdown = renderMarkdown;
})();
