/* 首帧前确定主题，避免闪烁：localStorage 优先，缺省跟随系统。
   独立外联文件（CSP script-src 'self' 禁内联脚本）。 */
(function () {
  try {
    var t = localStorage.getItem('fia_theme');
    if (!t) {
      t = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
        ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', t);
  } catch (e) { /* 忽略，默认浅色 */ }
})();
