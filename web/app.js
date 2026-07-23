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
  var quotaFill = $('quota-fill');
  var quotaText = $('quota-text');
  var topupEntryBtn = $('topup-entry-btn');
  var packCount = $('pack-count');

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

  var topupModal = $('topup-modal');
  var buyPackBtn = $('buy-pack-btn');
  var topupCloseBtn = $('topup-close-btn');
  var modalResetDate = $('modal-reset-date');

  /* ==================== 状态 ==================== */
  var state = {
    token: localStorage.getItem(STORAGE_TOKEN) || null,
    username: localStorage.getItem(STORAGE_USER) || null,
    me: null,               // /api/me 返回的额度信息
    conversations: [],
    activeConvId: null,
    streaming: false,
    abortController: null,
    authMode: 'login'       // 'login' | 'register'
  };

  /* ==================== 基础工具 ==================== */
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  /* 隐藏统一走「hidden 属性 + .hidden 类」双保险，
     CSS 侧 [hidden]/.hidden 均为 display:none !important */
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

  /* ==================== 视图切换 ==================== */
  function showAuth() {
    setHidden(appView, true);
    setHidden(authView, false);
    setQuotaBanner(false);
    closeTopupModal();           // 回到登录页时绝不残留弹层
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
      modalResetDate.textContent = '月度额度将于 ' + formatDate(me.reset_date) + ' 自动重置。';
    } else {
      quotaBannerReset.textContent = '次月1日自动重置';
      modalResetDate.textContent = '月度额度将于次月1日自动重置。';
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

  /* ==================== 加油包弹层 ====================
     只在两处打开：用户点「加油包」入口 / 429 额度耗尽。
     关闭路径：暂不购买按钮、遮罩点击、Esc 键。 */
  function openTopupModal() { setHidden(topupModal, false); }
  function closeTopupModal() { setHidden(topupModal, true); }

  async function buyPack() {
    buyPackBtn.disabled = true;
    buyPackBtn.textContent = '到账中…';
    try {
      var res = await api('/api/topup', { method: 'POST', body: { pack_count: 1 } });
      if (res.ok) {
        await loadMe();          // 刷新额度并解除禁用
        closeTopupModal();
      } else {
        buyPackBtn.textContent = '购买失败，请重试';
        setTimeout(function () { buyPackBtn.textContent = '购买加油包（+' + PACK_SIZE + '条）'; }, 1500);
      }
    } catch (e) {
      buyPackBtn.textContent = '网络异常，请重试';
      setTimeout(function () { buyPackBtn.textContent = '购买加油包（+' + PACK_SIZE + '条）'; }, 1500);
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

  function renderConvList() {
    convList.innerHTML = '';
    setHidden(convEmpty, state.conversations.length > 0);
    state.conversations.forEach(function (conv) {
      var li = el('li', 'conv-item' + (conv.id === state.activeConvId ? ' active' : ''));
      li.appendChild(el('span', 'conv-title', conv.title || '未命名对话'));
      var del = el('button', 'conv-del', '✕');
      del.title = '删除对话';
      del.addEventListener('click', function (ev) {
        ev.stopPropagation();
        deleteConversation(conv.id);
      });
      li.appendChild(del);
      li.addEventListener('click', function () { openConversation(conv.id); });
      convList.appendChild(li);
    });
  }

  async function openConversation(id) {
    if (state.streaming) return;
    try {
      var res = await api('/api/conversations/' + encodeURIComponent(id));
      if (res.status === 404) { loadConversations(); return; }
      if (!res.ok) return;
      var conv = await res.json();
      state.activeConvId = conv.id;
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

  /* ==================== 空状态 / 对话状态布局 ====================
     空状态：welcome 居中（问候语 + 居中输入卡片）；
     对话状态：messages 滚动区 + 底部输入卡片。
     同一个 #composer 节点在两个插槽之间移动。 */
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
    // 无内容 / 额度耗尽时发送键置灰；流式期间隐藏发送键改显停止键
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
    bubble.classList.add('thinking');
    bubble.textContent = '思考中…';
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
        bubble.classList.remove('thinking');
        if (!bubble.dataset.filled) bubble.innerHTML = '';
        var note = el('p', 'stopped-note', '（已停止生成）');
        bubble.appendChild(note);
      } else if (e && e.quotaExhausted) {
        // 429：服务端未落库，移除本地乐观气泡并恢复输入
        removeLastTwoBubbles();
        input.value = text;
        autoResize();
        await loadMe();
        updateQuotaState();
        openTopupModal();
      } else if (e && e.message === 'unauthorized') {
        // handleUnauthorized 已处理
      } else {
        bubble.classList.remove('thinking');
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
            bubble.classList.remove('thinking');
            bubble.innerHTML = window.renderMarkdown(full);
            scrollToBottom();
          }
        } catch (parseErr) { /* 忽略不完整/非 JSON 行 */ }
      }
    }

    if (!full) {
      bubble.classList.remove('thinking');
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
    $('auth-app-name').textContent = APP_NAME;
    $('welcome-title').textContent = '你好，我是 ' + APP_NAME;
  }

  function bindEvents() {
    tabLogin.addEventListener('click', function () { setAuthMode('login'); });
    tabRegister.addEventListener('click', function () { setAuthMode('register'); });
    authForm.addEventListener('submit', handleAuthSubmit);
    logoutBtn.addEventListener('click', logout);
    newChatBtn.addEventListener('click', function () { if (!state.streaming) startNewChat(); });
    sendBtn.addEventListener('click', sendMessage);
    stopBtn.addEventListener('click', stopStreaming);
    input.addEventListener('keydown', handleKeydown);
    input.addEventListener('input', handleInput);

    /* 加油包弹层：三个关闭路径 + 两个打开入口 */
    topupEntryBtn.addEventListener('click', openTopupModal);      // 打开：用户主动
    quotaBannerBtn.addEventListener('click', openTopupModal);     // 打开：额度提示条
    topupCloseBtn.addEventListener('click', closeTopupModal);     // 关闭①：暂不购买
    topupModal.addEventListener('click', function (ev) {          // 关闭②：点击遮罩
      if (ev.target === topupModal) closeTopupModal();
    });
    document.addEventListener('keydown', function (ev) {          // 关闭③：Esc 键
      if (ev.key === 'Escape' && !isHidden(topupModal)) closeTopupModal();
    });
  }

  async function init() {
    applyBranding();
    bindEvents();
    setAuthMode('login');
    buyPackBtn.textContent = '购买加油包（+' + PACK_SIZE + '条）';
    closeTopupModal();      // 保险：任何初始路径下弹层都处于关闭态
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
