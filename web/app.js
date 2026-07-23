/* ============================================================
   app.js —— Fine 前端主逻辑
   纯原生 JS，零依赖。接口契约见后端 /api/* 实现。
   ============================================================ */
(function () {
  'use strict';

  /* ==================== 可配置常量 ==================== */
  var APP_NAME = 'Fine';         // 品牌名 / 页面标题，改名只动这里
  var PACK_SIZE = 50;            // 每个加油包条数（与后端 WEB_PACK_SIZE 默认一致）
  var STORAGE_TOKEN = 'fia_token';
  var STORAGE_USER = 'fia_username';
  var STORAGE_THEME = 'fia_theme';

  /* 生成中占位小球（双眼高光，流式首帧到达后被 markdown 覆盖替换） */
  var GEN_BALL_HTML =
    '<span class="gen-ball">' +
    '<span class="gen-eye left"></span>' +
    '<span class="gen-eye right"></span>' +
    '</span>';

  /* ==================== DOM 引用 ==================== */
  var $ = function (id) { return document.getElementById(id); };
  var authView = $('auth-view');
  var appView = $('app-view');
  var authForm = $('auth-form');
  var authUsername = $('auth-username');
  var authPassword = $('auth-password');
  var authError = $('auth-error');
  var authSubmit = $('auth-submit');
  var tabLogin = $('tab-login');
  var tabRegister = $('tab-register');

  var newChatBtn = $('new-chat-btn');
  var convList = $('conv-list');
  var convEmpty = $('conv-empty');
  var userName = $('user-name');
  var logoutBtn = $('logout-btn');
  var themeToggle = $('theme-toggle');
  var quotaFill = $('quota-fill');
  var quotaText = $('quota-text');
  var topupEntryBtn = $('topup-entry-btn');
  var packCount = $('pack-count');

  var chatView = $('chat-view');
  var quotaBanner = $('quota-banner');
  var quotaBannerBtn = $('quota-banner-btn');
  var quotaBannerReset = $('quota-banner-reset');
  var messagesEl = $('messages');
  var welcomeEl = $('welcome');
  var composer = $('composer');
  var composerCenterSlot = $('composer-center-slot');
  var composerBottomSlot = $('composer-bottom-slot');
  var input = $('input');
  var sendBtn = $('send-btn');
  var stopBtn = $('stop-btn');

  var quotaView = $('quota-view');
  var quotaBackBtn = $('quota-back-btn');
  var quotaBigNum = $('quota-big-num');
  var quotaBarFill = $('quota-bar-fill');
  var quotaBarText = $('quota-bar-text');
  var quotaResetNote = $('quota-reset-note');
  var packBalance = $('pack-balance');
  var buyPackBtn = $('buy-pack-btn');

  var convMenu = $('conv-menu');
  var menuRename = $('menu-rename');
  var menuPin = $('menu-pin');
  var menuDelete = $('menu-delete');

  /* ==================== 状态 ==================== */
  var state = {
    token: localStorage.getItem(STORAGE_TOKEN) || null,
    username: localStorage.getItem(STORAGE_USER) || null,
    me: null,               // /api/me 返回的额度信息
    conversations: [],
    activeConvId: null,
    streaming: false,
    abortController: null,
    authMode: 'login',      // 'login' | 'register'
    view: 'chat',           // 'chat' | 'quota'
    menuConvId: null        // 当前打开操作菜单的对话 id
  };

  /* ==================== 基础工具 ==================== */
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  /* 隐藏统一走「hidden 属性 + .hidden 类」双保险 */
  function setHidden(node, hidden) {
    if (hidden) {
      node.setAttribute('hidden', '');
      node.classList.add('hidden');
    } else {
      node.removeAttribute('hidden');
      node.classList.remove('hidden');
    }
  }

  function isHidden(node) {
    return node.hasAttribute('hidden');
  }

  function authHeaders(extra) {
    var h = extra || {};
    if (state.token) h['Authorization'] = 'Bearer ' + state.token;
    return h;
  }

  /* 统一 JSON 请求；401 自动回登录页 */
  async function api(path, options) {
    var opts = options || {};
    var headers = authHeaders({ 'Content-Type': 'application/json' });
    var res = await fetch(path, {
      method: opts.method || 'GET',
      headers: headers,
      body: opts.body != null ? JSON.stringify(opts.body) : undefined
    });
    if (res.status === 401) {
      handleUnauthorized();
      throw new Error('unauthorized');
    }
    return res;
  }

  function handleUnauthorized() {
    clearSession();
    showAuth();
  }

  function clearSession() {
    state.token = null;
    state.username = null;
    state.me = null;
    state.conversations = [];
    state.activeConvId = null;
    localStorage.removeItem(STORAGE_TOKEN);
    localStorage.removeItem(STORAGE_USER);
  }

  function findConv(id) {
    for (var i = 0; i < state.conversations.length; i++) {
      if (state.conversations[i].id === id) return state.conversations[i];
    }
    return null;
  }

  /* ==================== 主题切换 ==================== */
  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function toggleTheme() {
    var t = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(STORAGE_THEME, t); } catch (e) { /* 忽略 */ }
  }

  /* ==================== hero 视差 ====================
     mousemove 记录目标位移，rAF 循环 lerp 平滑；
     各层按 data-depth 系数以不同幅度跟随（--px/--py）。 */
  function initHero(hero) {
    if (!hero) return;
    var layers = Array.prototype.slice.call(hero.querySelectorAll('[data-depth]'));
    if (!layers.length) return;
    var tx = 0, ty = 0, cx = 0, cy = 0, raf = null;

    function loop() {
      cx += (tx - cx) * 0.12;
      cy += (ty - cy) * 0.12;
      for (var i = 0; i < layers.length; i++) {
        var d = parseFloat(layers[i].getAttribute('data-depth')) || 0;
        layers[i].style.setProperty('--px', (cx * d).toFixed(2) + 'px');
        layers[i].style.setProperty('--py', (cy * d).toFixed(2) + 'px');
      }
      if (Math.abs(tx - cx) > 0.05 || Math.abs(ty - cy) > 0.05) {
        raf = requestAnimationFrame(loop);
      } else {
        raf = null;
      }
    }
    function kick() { if (!raf) raf = requestAnimationFrame(loop); }

    hero.addEventListener('mousemove', function (ev) {
      var r = hero.getBoundingClientRect();
      if (!r.width || !r.height) return;
      tx = ((ev.clientX - r.left) / r.width - 0.5) * 18;
      ty = ((ev.clientY - r.top) / r.height - 0.5) * 12;
      kick();
    });
    hero.addEventListener('mouseleave', function () {
      tx = 0; ty = 0;
      kick();
    });
  }

  /* ==================== 视图切换 ==================== */
  function showAuth() {
    setHidden(appView, true);
    setHidden(authView, false);
    setQuotaBanner(false);
    closeConvMenu();
    authPassword.value = '';
    authUsername.focus();
  }

  function showApp() {
    setHidden(authView, true);
    setHidden(appView, false);
    renderUserBar();
    loadMe();
    loadConversations();
    startNewChat();
  }

  /* ==================== 认证 ==================== */
  function setAuthMode(mode) {
    state.authMode = mode;
    tabLogin.classList.toggle('active', mode === 'login');
    tabRegister.classList.toggle('active', mode === 'register');
    authSubmit.textContent = mode === 'login' ? '登录' : '注册';
    setHidden(authError, true);
  }

  async function handleAuthSubmit(ev) {
    ev.preventDefault();
    var username = authUsername.value.trim();
    var password = authPassword.value;
    if (!username || !password) return;

    authSubmit.disabled = true;
    setHidden(authError, true);
    var path = state.authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
    try {
      var res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username, password: password })
      });
      var data = await res.json().catch(function () { return {}; });
      if (res.ok) {
        state.token = data.token;
        state.username = (data.user && data.user.username) || username;
        localStorage.setItem(STORAGE_TOKEN, state.token);
        localStorage.setItem(STORAGE_USER, state.username);
        showApp();
      } else if (res.status === 409) {
        showAuthError('该用户名已被注册，请换一个或直接登录');
      } else if (res.status === 401) {
        showAuthError('用户名或密码错误');
      } else {
        showAuthError(data.error || '请求失败，请稍后重试');
      }
    } catch (e) {
      showAuthError('网络异常，请稍后重试');
    } finally {
      authSubmit.disabled = false;
    }
  }

  function showAuthError(msg) {
    authError.textContent = msg;
    setHidden(authError, false);
  }

  async function logout() {
    try {
      await api('/api/auth/logout', { method: 'POST' });
    } catch (e) { /* 忽略网络错误，本地照常登出 */ }
    clearSession();
    showAuth();
  }

  /* ==================== 额度信息 ==================== */
  async function loadMe() {
    try {
      var res = await api('/api/me');
      if (!res.ok) return;
      state.me = await res.json();
      renderUserBar();
      updateQuotaState();
      if (state.view === 'quota') renderQuotaView();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  function renderUserBar() {
    userName.textContent = state.username || '—';

    var me = state.me;
    if (!me) {
      quotaText.textContent = '—';
      quotaFill.style.width = '0%';
      packCount.textContent = '';
      return;
    }
    var quota = me.monthly_quota || 0;
    var used = me.monthly_used || 0;
    var pct = quota > 0 ? Math.min(100, Math.round(used / quota * 100)) : 0;
    quotaFill.style.width = pct + '%';
    quotaFill.classList.toggle('low', (me.monthly_remaining || 0) <= 0 && quota > 0);
    quotaText.textContent = '本月 ' + (me.monthly_remaining != null ? me.monthly_remaining : '—') + '/' + quota;
    packCount.textContent = me.pack_credits ? '（余 ' + me.pack_credits + '）' : '';
  }

  /* 额度状态：控制提示条与输入禁用 */
  function updateQuotaState() {
    var me = state.me;
    var exhausted = me && typeof me.total_remaining === 'number' && me.total_remaining <= 0;
    setQuotaBanner(!!exhausted);
    input.disabled = !!exhausted;
    if (exhausted && me.reset_date) {
      quotaBannerReset.textContent = '将于 ' + formatDate(me.reset_date) + ' 自动重置';
    } else {
      quotaBannerReset.textContent = '次月1日自动重置';
    }
    updateSendState();
  }

  function setQuotaBanner(show) {
    setHidden(quotaBanner, !show);
  }

  function formatDate(iso) {
    if (!iso) return '次月1日';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return (d.getMonth() + 1) + '月' + d.getDate() + '日';
  }

  /* ==================== 额度页（独立视图） ==================== */
  function showQuotaView() {
    state.view = 'quota';
    closeConvMenu();
    setHidden(chatView, true);
    setHidden(quotaView, false);
    renderQuotaView();
    loadMe(); // 进入时拉取最新额度
  }

  function showChatView() {
    state.view = 'chat';
    setHidden(quotaView, true);
    setHidden(chatView, false);
  }

  function renderQuotaView() {
    var me = state.me;
    if (!me) {
      quotaBigNum.textContent = '—';
      quotaBarFill.style.width = '0%';
      quotaBarText.textContent = '加载中…';
      quotaResetNote.textContent = '';
      packBalance.textContent = '—';
      return;
    }
    quotaBigNum.textContent = me.total_remaining != null ? me.total_remaining : '—';
    var quota = me.monthly_quota || 0;
    var used = me.monthly_used || 0;
    var pct = quota > 0 ? Math.min(100, Math.round(used / quota * 100)) : 0;
    quotaBarFill.style.width = pct + '%';
    quotaBarFill.classList.toggle('low', (me.monthly_remaining || 0) <= 0 && quota > 0);
    quotaBarText.textContent = '已用 ' + used + ' / ' + quota + ' 条';
    quotaResetNote.textContent = '月度额度将于 ' + formatDate(me.reset_date) + ' 重置';
    packBalance.textContent = me.pack_credits != null ? me.pack_credits : 0;
  }

  async function buyPack() {
    buyPackBtn.disabled = true;
    var oldText = buyPackBtn.textContent;
    buyPackBtn.textContent = '到账中…';
    try {
      var res = await api('/api/topup', { method: 'POST', body: { pack_count: 1 } });
      if (res.ok) {
        await loadMe();          // 刷新额度页与侧边栏
        buyPackBtn.textContent = oldText;
      } else {
        buyPackBtn.textContent = '充值失败，请重试';
        setTimeout(function () { buyPackBtn.textContent = oldText; }, 1500);
      }
    } catch (e) {
      buyPackBtn.textContent = '网络异常，请重试';
      setTimeout(function () { buyPackBtn.textContent = oldText; }, 1500);
    } finally {
      buyPackBtn.disabled = false;
    }
  }

  /* ==================== 对话列表 ==================== */
  async function loadConversations() {
    try {
      var res = await api('/api/conversations');
      if (!res.ok) return;
      state.conversations = await res.json();
      renderConvList();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  /* 置顶优先，组内保持服务端 updated_at 降序（稳定排序） */
  function sortedConversations() {
    return state.conversations.slice().sort(function (a, b) {
      return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
    });
  }

  function renderConvList() {
    convList.innerHTML = '';
    setHidden(convEmpty, state.conversations.length > 0);
    sortedConversations().forEach(function (conv) {
      var li = el('li', 'conv-item' + (conv.id === state.activeConvId ? ' active' : ''));
      li.dataset.convId = conv.id;

      if (conv.pinned) li.appendChild(el('span', 'conv-pin', '📌'));
      li.appendChild(el('span', 'conv-title', conv.title || '未命名对话'));

      var menuBtn = el('button', 'conv-menu-btn', '···');
      menuBtn.title = '更多操作';
      menuBtn.addEventListener('click', function (ev) {
        ev.stopPropagation();
        openConvMenu(conv, menuBtn);
      });
      li.appendChild(menuBtn);

      li.addEventListener('click', function () {
        if (li.querySelector('.conv-rename-input')) return; // 重命名中不切换
        openConversation(conv.id);
      });
      convList.appendChild(li);
    });
  }

  /* ==================== 对话操作菜单 ==================== */
  function openConvMenu(conv, anchorBtn) {
    state.menuConvId = conv.id;
    menuPin.textContent = conv.pinned ? '📌 取消置顶' : '📌 置顶';

    var rect = anchorBtn.getBoundingClientRect();
    var menuW = 148, menuH = 120;
    var left = Math.max(8, Math.min(rect.right - menuW, window.innerWidth - menuW - 8));
    var top = rect.bottom + 4;
    if (top + menuH > window.innerHeight - 8) top = Math.max(8, rect.top - menuH - 4);
    convMenu.style.left = left + 'px';
    convMenu.style.top = top + 'px';
    setHidden(convMenu, false);
  }

  function closeConvMenu() {
    setHidden(convMenu, true);
    state.menuConvId = null;
  }

  function menuConvLi() {
    return convList.querySelector('[data-conv-id="' + state.menuConvId + '"]');
  }

  /* 编辑标题：就地 input，Enter/blur 提交，Esc 取消 */
  function startRename(conv) {
    var li = menuConvLi();
    if (!li || li.querySelector('.conv-rename-input')) return;
    var titleEl = li.querySelector('.conv-title');
    if (!titleEl) return;

    var renameInput = document.createElement('input');
    renameInput.className = 'conv-rename-input';
    renameInput.value = conv.title || '';
    renameInput.maxLength = 60;
    titleEl.replaceWith(renameInput);
    renameInput.focus();
    renameInput.select();

    var cancelled = false;
    renameInput.addEventListener('click', function (ev) { ev.stopPropagation(); });
    renameInput.addEventListener('keydown', function (ev) {
      ev.stopPropagation();
      if (ev.key === 'Enter') { renameInput.blur(); }
      else if (ev.key === 'Escape') { cancelled = true; renameInput.blur(); }
    });
    renameInput.addEventListener('blur', function () {
      if (cancelled) { renderConvList(); return; }
      commitRename(conv, renameInput.value.trim());
    });
  }

  async function commitRename(conv, title) {
    if (title && title !== conv.title) {
      try {
        var res = await api('/api/conversations/' + encodeURIComponent(conv.id), {
          method: 'PATCH',
          body: { title: title }
        });
        if (res.ok) conv.title = title;
      } catch (e) {
        if (e.message !== 'unauthorized') console.error(e);
      }
    }
    renderConvList();
  }

  async function togglePin(conv) {
    var next = !conv.pinned;
    conv.pinned = next;               // 乐观更新
    renderConvList();
    try {
      var res = await api('/api/conversations/' + encodeURIComponent(conv.id), {
        method: 'PATCH',
        body: { pinned: next }
      });
      if (!res.ok) { conv.pinned = !next; renderConvList(); }
    } catch (e) {
      if (e.message !== 'unauthorized') { conv.pinned = !next; renderConvList(); }
    }
  }

  async function openConversation(id) {
    if (state.streaming) return;
    try {
      var res = await api('/api/conversations/' + encodeURIComponent(id));
      if (res.status === 404) { loadConversations(); return; }
      if (!res.ok) return;
      var conv = await res.json();
      state.activeConvId = conv.id;
      showChatView();
      renderConvList();
      renderMessages(conv.messages || []);
      input.focus();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  async function deleteConversation(id) {
    try {
      var res = await api('/api/conversations/' + encodeURIComponent(id), { method: 'DELETE' });
      if (!res.ok && res.status !== 404) return;
      state.conversations = state.conversations.filter(function (c) { return c.id !== id; });
      if (state.activeConvId === id) startNewChat();
      renderConvList();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  /* ==================== 空状态 / 对话状态布局 ==================== */
  function showWelcome(show) {
    setHidden(welcomeEl, !show);
    setHidden(messagesEl, show);
    if (show) {
      composerCenterSlot.appendChild(composer);
    } else {
      composerBottomSlot.appendChild(composer);
    }
  }

  function startNewChat() {
    state.activeConvId = null;
    showChatView();
    messagesEl.innerHTML = '';
    showWelcome(true);
    renderConvList();
    input.focus();
  }

  /* ==================== 消息渲染 ==================== */
  function renderMessages(messages) {
    messagesEl.innerHTML = '';
    if (!messages.length) {
      showWelcome(true);
      return;
    }
    showWelcome(false);
    messages.forEach(function (m) { appendMessage(m.role, m.content); });
    scrollToBottom();
  }

  function appendMessage(role, content) {
    showWelcome(false);
    var row = el('div', 'msg-row ' + (role === 'user' ? 'user' : 'assistant'));
    var bubble = el('div', 'msg-bubble');
    if (role === 'assistant') {
      bubble.classList.add('md');
      bubble.innerHTML = window.renderMarkdown(content || '');
    } else {
      bubble.textContent = content;
    }
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /* ==================== 发送与 SSE 流 ==================== */
  function updateSendState() {
    var me = state.me;
    var exhausted = me && typeof me.total_remaining === 'number' && me.total_remaining <= 0;
    sendBtn.disabled = !!exhausted || !input.value.trim();
  }

  async function sendMessage() {
    var text = input.value.trim();
    if (!text || state.streaming) return;
    if (state.me && typeof state.me.total_remaining === 'number' && state.me.total_remaining <= 0) {
      updateQuotaState();
      return;
    }

    input.value = '';
    autoResize();
    appendMessage('user', text);
    var bubble = appendMessage('assistant', '');
    // 生成中占位：灵动小球（首帧内容到达后被 markdown 覆盖）
    bubble.classList.remove('md');
    bubble.classList.add('gen-stage');
    bubble.innerHTML = GEN_BALL_HTML;
    setStreaming(true);

    try {
      // 新对话：先创建会话，标题取首条消息前 20 字
      if (!state.activeConvId) {
        var res = await api('/api/conversations', {
          method: 'POST',
          body: { title: text.slice(0, 20) }
        });
        if (!res.ok) throw new Error('create_conversation_failed');
        var conv = await res.json();
        state.activeConvId = conv.id;
      }

      await streamChat(state.activeConvId, text, bubble);

      // 流正常结束：刷新额度与列表（updated_at 变化、首条后标题入列）
      await Promise.all([loadMe(), loadConversations()]);
    } catch (e) {
      if (e && e.name === 'AbortError') {
        bubble.classList.remove('gen-stage');
        if (!bubble.dataset.filled) bubble.innerHTML = '';
        var note = el('p', 'stopped-note', '（已停止生成）');
        bubble.appendChild(note);
      } else if (e && e.quotaExhausted) {
        // 429：服务端未落库，移除本地乐观气泡并恢复输入，跳转额度页
        removeLastTwoBubbles();
        input.value = text;
        autoResize();
        await loadMe();
        updateQuotaState();
        showQuotaView();
      } else if (e && e.message === 'unauthorized') {
        // handleUnauthorized 已处理
      } else {
        bubble.classList.remove('gen-stage');
        bubble.classList.remove('md');
        bubble.textContent = '出错了，请稍后重试。';
      }
    } finally {
      setStreaming(false);
    }
  }

  function removeLastTwoBubbles() {
    var rows = messagesEl.querySelectorAll('.msg-row');
    for (var k = 0; k < 2 && rows.length > 0; k++) {
      var last = rows[rows.length - 1];
      last.parentNode.removeChild(last);
      rows = messagesEl.querySelectorAll('.msg-row');
    }
    if (rows.length === 0) showWelcome(true);
  }

  function setStreaming(on) {
    state.streaming = on;
    setHidden(sendBtn, on);
    setHidden(stopBtn, !on);
    if (!on) {
      state.abortController = null;
      updateSendState();
      input.focus();
    }
  }

  /* 消费 POST /api/chat 的 SSE 流 */
  async function streamChat(conversationId, message, bubble) {
    var controller = new AbortController();
    state.abortController = controller;

    var res = await fetch('/api/chat', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ conversation_id: conversationId, message: message }),
      signal: controller.signal
    });

    if (res.status === 401) { handleUnauthorized(); throw new Error('unauthorized'); }
    if (res.status === 429) {
      var quotaErr = new Error('quota_exhausted');
      quotaErr.quotaExhausted = true;
      throw quotaErr;
    }
    if (!res.ok || !res.body) throw new Error('chat_failed_' + res.status);

    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';
    var full = '';
    var done = false;

    while (!done) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });

      var idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        var line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line || line.indexOf('data:') !== 0) continue;
        var payload = line.slice(5).trim();
        if (payload === '[DONE]') { done = true; break; }
        try {
          var json = JSON.parse(payload);
          var delta = json.choices && json.choices[0] && json.choices[0].delta;
          if (delta && typeof delta.content === 'string' && delta.content) {
            full += delta.content;
            bubble.dataset.filled = '1';
            bubble.classList.remove('gen-stage');
            bubble.classList.add('md');
            bubble.innerHTML = window.renderMarkdown(full);  // 覆盖生成中小球
            scrollToBottom();
          }
        } catch (parseErr) { /* 忽略不完整/非 JSON 行 */ }
      }
    }

    if (!full) {
      bubble.classList.remove('gen-stage');
      bubble.classList.remove('md');
      bubble.textContent = '（未收到回复内容）';
    }
  }

  function stopStreaming() {
    if (state.abortController) state.abortController.abort();
  }

  /* ==================== 输入框 ==================== */
  function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 180) + 'px';
  }

  function handleKeydown(ev) {
    if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      sendMessage();
    }
  }

  function handleInput() {
    autoResize();
    updateSendState();
  }

  /* ==================== 初始化 ==================== */
  function applyBranding() {
    document.title = APP_NAME;
  }

  function bindEvents() {
    tabLogin.addEventListener('click', function () { setAuthMode('login'); });
    tabRegister.addEventListener('click', function () { setAuthMode('register'); });
    authForm.addEventListener('submit', handleAuthSubmit);
    logoutBtn.addEventListener('click', logout);
    themeToggle.addEventListener('click', toggleTheme);
    newChatBtn.addEventListener('click', function () { if (!state.streaming) startNewChat(); });
    sendBtn.addEventListener('click', sendMessage);
    stopBtn.addEventListener('click', stopStreaming);
    input.addEventListener('keydown', handleKeydown);
    input.addEventListener('input', handleInput);

    /* 额度页入口与返回 */
    topupEntryBtn.addEventListener('click', showQuotaView);
    quotaBannerBtn.addEventListener('click', showQuotaView);
    quotaBackBtn.addEventListener('click', showChatView);
    buyPackBtn.addEventListener('click', buyPack);

    /* 对话操作菜单：三项动作 + 点击外部 / Esc 关闭 */
    menuRename.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var conv = findConv(state.menuConvId);
      closeConvMenu();
      if (conv) startRename(conv);
    });
    menuPin.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var conv = findConv(state.menuConvId);
      closeConvMenu();
      if (conv) togglePin(conv);
    });
    menuDelete.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var id = state.menuConvId;
      closeConvMenu();
      if (id) deleteConversation(id);
    });
    convMenu.addEventListener('click', function (ev) { ev.stopPropagation(); });
    document.addEventListener('click', function () { closeConvMenu(); });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && !isHidden(convMenu)) closeConvMenu();
    });
  }

  async function init() {
    applyBranding();
    bindEvents();
    initHero($('hero-auth'));
    initHero($('hero-welcome'));
    setAuthMode('login');
    buyPackBtn.textContent = '充值 +' + PACK_SIZE + ' 条';
    closeConvMenu();
    updateSendState();

    if (state.token) {
      // 已有 token：验证有效性，有效则直接进入主界面
      try {
        var res = await api('/api/me');
        if (res.ok) {
          state.me = await res.json();
          showApp();
          return;
        }
      } catch (e) {
        if (e.message === 'unauthorized') return; // 已回登录页
      }
      showAuth();
    } else {
      showAuth();
    }
  }

  init();
})();
