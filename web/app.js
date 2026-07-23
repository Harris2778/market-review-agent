/* ============================================================
   app.js —— Fine 前端主逻辑
   纯原生 JS，零依赖。接口契约见后端 /api/* 实现。
   ============================================================ */
(function () {
  'use strict';

  /* ==================== 可配置常量 ==================== */
  var APP_NAME = 'Fine';         // 品牌名 / 页面标题，改名只动这里
  var STORAGE_TOKEN = 'fia_token';
  var STORAGE_USER = 'fia_username';
  var STORAGE_THEME = 'fia_theme';
  var STORAGE_DEVICE = 'fia_device';
  var STORAGE_ADMIN_BACKUP = 'fia_admin_backup';  // 管理员切换身份前的会话备份
  var QUESTIONS_PAGE_SIZE = 20;

  /* 管理端点（后端契约） */
  var ADMIN_USERS_API = '/api/admin/users';
  function adminQuotaApi(username) {
    return '/api/admin/users/' + encodeURIComponent(username) + '/quota';
  }
  function adminQuestionsApi(username, offset, limit) {
    return '/api/admin/users/' + encodeURIComponent(username) +
      '/questions?offset=' + offset + '&limit=' + limit;
  }

  /* 生成中占位：小球 + 状态文案（思考中→首帧到达变「思考已完成」，流结束移除） */
  var GEN_BALL_HTML =
    '<span class="gen-status">' +
    '<span class="gen-ball">' +
    '<span class="gen-eye left"></span>' +
    '<span class="gen-eye right"></span>' +
    '</span>' +
    '<span class="gen-label">正在思考中</span>' +
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
  var returnAdminBtn = $('return-admin-btn');
  var themeToggle = $('theme-toggle');
  var quotaRow = $('quota-row');
  var quotaFill = $('quota-fill');
  var quotaText = $('quota-text');
  var quotaResetInline = $('quota-reset-inline');
  var adminEntryBtn = $('admin-entry-btn');
  var menuToggle = $('menu-toggle');
  var sidebarMask = $('sidebar-mask');

  var chatView = $('chat-view');
  var quotaBanner = $('quota-banner');
  var quotaBannerText = $('quota-banner-text');
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

  var adminView = $('admin-view');
  var adminBackBtn = $('admin-back-btn');
  var adminTbody = $('admin-tbody');
  var adminEmpty = $('admin-empty');
  var adminUsersWrap = $('admin-users-wrap');
  var adminQuestions = $('admin-questions');
  var questionsBackBtn = $('questions-back-btn');
  var questionsTitle = $('questions-title');
  var questionsList = $('questions-list');
  var questionsEmpty = $('questions-empty');
  var questionsMore = $('questions-more');

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
    view: 'chat',           // 'chat' | 'quota' | 'admin'
    menuConvId: null,       // 当前打开操作菜单的对话 id
    adminUsers: [],
    questions: { username: null, offset: 0, loading: false, hasMore: false }
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
    localStorage.removeItem(STORAGE_ADMIN_BACKUP);
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

  /* ==================== 设备指纹（注册防多号） ==================== */
  function getDeviceId() {
    try {
      var id = localStorage.getItem(STORAGE_DEVICE);
      if (!id) {
        id = (window.crypto && typeof window.crypto.randomUUID === 'function')
          ? window.crypto.randomUUID()
          : 'dev-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 12);
        localStorage.setItem(STORAGE_DEVICE, id);
      }
      return id;
    } catch (e) {
      return null; // localStorage 不可用时降级为不上报
    }
  }

  /* ==================== 额度字段兼容解析 ====================
     新契约：quota_used / quota_limit / reset_date / is_admin；
     过渡期兼容旧字段 monthly_used / monthly_quota。 */
  function meUsed(me) { return me.quota_used != null ? me.quota_used : (me.monthly_used || 0); }
  function meLimit(me) { return me.quota_limit != null ? me.quota_limit : (me.monthly_quota || 0); }
  function meRemaining(me) {
    var r = meLimit(me) - meUsed(me);
    return r > 0 ? r : 0;
  }
  function meExhausted(me) {
    return !!me && meLimit(me) > 0 && meRemaining(me) <= 0;
  }

  /* ==================== 移动端抽屉侧边栏 ==================== */
  function openSidebar() {
    appView.classList.add('sidebar-open');
    setHidden(sidebarMask, false);
  }
  function closeSidebar() {
    appView.classList.remove('sidebar-open');
    setHidden(sidebarMask, true);
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
    var isRegister = state.authMode === 'register';
    var path = isRegister ? '/api/auth/register' : '/api/auth/login';
    var payload = { username: username, password: password };
    if (isRegister) {
      var deviceId = getDeviceId();
      if (deviceId) payload.device_id = deviceId;   // 注册防多号：上报设备指纹
    }
    try {
      var res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      var data = await res.json().catch(function () { return {}; });
      if (res.ok) {
        state.token = data.token;
        state.username = (data.user && data.user.username) || username;
        localStorage.setItem(STORAGE_TOKEN, state.token);
        localStorage.setItem(STORAGE_USER, state.username);
        showApp();
      } else if (res.status === 409 && data.error === 'device_limit') {
        showAuthError('该设备注册账号数量已达上限');
      } else if (res.status === 409) {
        showAuthError('该用户名已被注册，请换一个或直接登录');
      } else if (res.status === 429) {
        showAuthError('当前网络注册过于频繁，请稍后再试');
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
      setHidden(adminEntryBtn, !state.me.is_admin);   // 管理入口仅管理员可见
      if (state.view === 'quota') renderQuotaView();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  /* 用户栏：额度进度条（已用/上限）+ 重置日期小字 */
  function renderUserBar() {
    userName.textContent = state.username || '—';
    // 身份切换中：显示「↩ 管理员」返回按钮
    setHidden(returnAdminBtn, !localStorage.getItem(STORAGE_ADMIN_BACKUP));

    var me = state.me;
    if (!me) {
      quotaText.textContent = '—';
      quotaFill.style.width = '0%';
      quotaResetInline.textContent = '';
      return;
    }
    var limit = meLimit(me);
    var used = meUsed(me);
    var pct = limit > 0 ? Math.min(100, Math.round(used / limit * 100)) : 0;
    quotaFill.style.width = pct + '%';
    quotaFill.classList.toggle('low', meExhausted(me));
    quotaText.textContent = '已用 ' + used + '/' + limit;
    quotaResetInline.textContent = me.reset_date ? formatDate(me.reset_date) + ' 重置' : '';
  }

  /* 额度状态：控制提示条与输入禁用（提示条为纯说明，无购买引导） */
  function updateQuotaState() {
    var me = state.me;
    var exhausted = meExhausted(me);
    setQuotaBanner(!!exhausted);
    input.disabled = !!exhausted;
    if (exhausted) {
      quotaBannerText.textContent = '本月额度已用完，将于 ' + formatDate(me.reset_date) + ' 重置';
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

  function formatDateTime(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    var hh = String(d.getHours()).padStart(2, '0');
    var mm = String(d.getMinutes()).padStart(2, '0');
    return (d.getMonth() + 1) + '月' + d.getDate() + '日 ' + hh + ':' + mm;
  }

  /* ==================== 三视图切换（chat / quota / admin） ==================== */
  function showView(name) {
    state.view = name;
    closeConvMenu();
    setHidden(chatView, name !== 'chat');
    setHidden(quotaView, name !== 'quota');
    setHidden(adminView, name !== 'admin');
  }

  function showChatView() { showView('chat'); }

  function showQuotaView() {
    showView('quota');
    renderQuotaView();
    loadMe(); // 进入时拉取最新额度
  }

  function showAdminView() {
    showView('admin');
    showUsersPanel();
    loadAdminUsers();
  }

  function renderQuotaView() {
    var me = state.me;
    if (!me) {
      quotaBigNum.textContent = '—';
      quotaBarFill.style.width = '0%';
      quotaBarText.textContent = '加载中…';
      quotaResetNote.textContent = '';
      return;
    }
    var limit = meLimit(me);
    var used = meUsed(me);
    quotaBigNum.textContent = meRemaining(me);
    var pct = limit > 0 ? Math.min(100, Math.round(used / limit * 100)) : 0;
    quotaBarFill.style.width = pct + '%';
    quotaBarFill.classList.toggle('low', meExhausted(me));
    quotaBarText.textContent = '已用 ' + used + ' / ' + limit + ' 条';
    quotaResetNote.textContent = '月度额度将于 ' + formatDate(me.reset_date) + ' 重置';
  }

  /* ==================== 管理视图（仅 is_admin） ==================== */
  function showUsersPanel() {
    setHidden(adminUsersWrap, false);
    setHidden(adminQuestions, true);
  }

  async function loadAdminUsers() {
    try {
      var res = await api(ADMIN_USERS_API);
      if (!res.ok) return;
      var data = await res.json();
      state.adminUsers = Array.isArray(data) ? data : (data.users || data.items || []);
      renderAdminUsers();
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    }
  }

  function renderAdminUsers() {
    adminTbody.innerHTML = '';
    setHidden(adminEmpty, state.adminUsers.length > 0);
    state.adminUsers.forEach(function (u) {
      var used = u.quota_used != null ? u.quota_used : (u.monthly_used || 0);
      var limit = u.quota_limit != null ? u.quota_limit : (u.monthly_quota || 0);
      var tr = el('tr');

      tr.appendChild(el('td', null, u.username)).dataset.label = '用户名';
      tr.appendChild(el('td', null, formatDate(u.created_at))).dataset.label = '注册时间';
      var usageTd = el('td', null, '已用 ' + used + ' / ' + limit);
      usageTd.dataset.label = '本周期用量';
      tr.appendChild(usageTd);
      tr.appendChild(el('td', null, formatDate(u.reset_date))).dataset.label = '重置日期';

      // 操作：额度上限就地编辑 + 保存；查看提问
      var opsTd = el('td', 'admin-ops');
      opsTd.dataset.label = '操作';
      var quotaInput = document.createElement('input');
      quotaInput.className = 'quota-edit-input';
      quotaInput.type = 'number';
      quotaInput.min = '0';
      quotaInput.value = limit;
      var saveBtn = el('button', 'btn btn-primary btn-sm', '保存');
      var hint = el('span', 'row-saved-hint');
      saveBtn.addEventListener('click', function () {
        saveUserQuota(u, quotaInput, saveBtn, hint, usageTd);
      });
      var viewBtn = el('button', 'btn btn-ghost btn-sm', '查看提问');
      viewBtn.addEventListener('click', function () { showQuestionsPanel(u.username); });
      var impBtn = el('button', 'btn btn-ghost btn-sm', '进入账号');
      impBtn.title = '免密码以该用户身份使用网站';
      impBtn.addEventListener('click', function () { impersonateUser(u.username); });
      opsTd.appendChild(quotaInput);
      opsTd.appendChild(saveBtn);
      opsTd.appendChild(viewBtn);
      opsTd.appendChild(impBtn);
      opsTd.appendChild(hint);
      tr.appendChild(opsTd);

      adminTbody.appendChild(tr);
    });
  }

  async function saveUserQuota(user, inputEl, btn, hint, usageTd) {
    var n = parseInt(inputEl.value, 10);
    if (isNaN(n) || n < 0) {
      hint.textContent = '请输入非负整数';
      return;
    }
    btn.disabled = true;
    hint.textContent = '';
    try {
      var res = await api(adminQuotaApi(user.username), {
        method: 'PATCH',
        body: { quota_limit: n }
      });
      if (res.ok) {
        user.quota_limit = n;   // 实时刷新该行
        var used = user.quota_used != null ? user.quota_used : (user.monthly_used || 0);
        usageTd.textContent = '已用 ' + used + ' / ' + n;
        hint.textContent = '已保存';
        setTimeout(function () { hint.textContent = ''; }, 2000);
      } else {
        hint.textContent = '保存失败';
      }
    } catch (e) {
      if (e.message !== 'unauthorized') hint.textContent = '网络异常';
    } finally {
      btn.disabled = false;
    }
  }

  /* ---- 管理员免密切换身份 ---- */
  async function impersonateUser(username) {
    if (!window.confirm('确定要以「' + username + '」的身份使用网站吗？\n当前管理员会话会保留，可随时点左下角「↩ 管理员」返回。')) return;
    try {
      var res = await api('/api/admin/users/' + encodeURIComponent(username) + '/impersonate', { method: 'POST' });
      if (!res.ok) { window.alert('切换失败（' + res.status + '）'); return; }
      var data = await res.json();
      // 备份当前管理员会话，便于一键返回
      localStorage.setItem(STORAGE_ADMIN_BACKUP, JSON.stringify({
        token: state.token, username: state.username
      }));
      localStorage.setItem(STORAGE_TOKEN, data.token);
      localStorage.setItem(STORAGE_USER, data.user.username);
      location.reload();
    } catch (e) {
      if (e.message !== 'unauthorized') window.alert('网络异常，请稍后重试');
    }
  }

  /* 结束身份切换，恢复管理员会话 */
  function returnToAdmin() {
    var raw = localStorage.getItem(STORAGE_ADMIN_BACKUP);
    if (!raw) return;
    try {
      var backup = JSON.parse(raw);
      localStorage.setItem(STORAGE_TOKEN, backup.token);
      localStorage.setItem(STORAGE_USER, backup.username);
      localStorage.removeItem(STORAGE_ADMIN_BACKUP);
      location.reload();
    } catch (e) {
      localStorage.removeItem(STORAGE_ADMIN_BACKUP);
    }
  }

  /* ---- 查看提问：分页加载 ---- */
  function showQuestionsPanel(username) {
    state.questions = { username: username, offset: 0, loading: false, hasMore: false };
    questionsTitle.textContent = '「' + username + '」的提问记录';
    questionsList.innerHTML = '';
    setHidden(adminUsersWrap, true);
    setHidden(adminQuestions, false);
    loadQuestions();
  }

  async function loadQuestions() {
    var q = state.questions;
    if (!q.username || q.loading) return;
    q.loading = true;
    questionsMore.disabled = true;
    try {
      var res = await api(adminQuestionsApi(q.username, q.offset, QUESTIONS_PAGE_SIZE));
      if (!res.ok) return;
      var data = await res.json();
      var items = Array.isArray(data) ? data : (data.items || data.questions || []);
      q.hasMore = Array.isArray(data) ? items.length === QUESTIONS_PAGE_SIZE
        : (data.has_more != null ? !!data.has_more : items.length === QUESTIONS_PAGE_SIZE);
      items.forEach(appendQuestionItem);
      q.offset += items.length;
      setHidden(questionsEmpty, questionsList.children.length > 0);
      setHidden(questionsMore, !q.hasMore);
    } catch (e) {
      if (e.message !== 'unauthorized') console.error(e);
    } finally {
      q.loading = false;
      questionsMore.disabled = false;
    }
  }

  function appendQuestionItem(item) {
    var li = el('li', 'question-item');
    var time = formatDateTime(item.created_at || item.time);
    var convTitle = item.conversation_title || item.title || '未命名对话';
    li.appendChild(el('div', 'question-meta', time + ' · ' + convTitle));
    li.appendChild(el('div', 'question-content', item.content || item.question || ''));
    questionsList.appendChild(li);
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
      var li = el('li', 'conv-item' + (conv.id === state.activeConvId ? ' active' : '') + (conv.pinned ? ' pinned' : ''));
      li.dataset.convId = conv.id;

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
    menuPin.textContent = conv.pinned ? '取消置顶' : '置顶';

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
      closeSidebar();                 // 移动端：选中对话后自动收起抽屉
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
    closeSidebar();                   // 移动端：新对话后自动收起抽屉
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
    sendBtn.disabled = meExhausted(state.me) || !input.value.trim();
  }

  async function sendMessage() {
    var text = input.value.trim();
    if (!text || state.streaming) return;
    if (meExhausted(state.me)) {
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
        // 429：服务端未落库，移除本地乐观气泡并恢复输入；提示条说明重置日期
        removeLastTwoBubbles();
        input.value = text;
        autoResize();
        await loadMe();
        updateQuotaState();
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
            if (!bubble.dataset.filled) {
              // 首帧到达：状态从「正在思考中」切换为「思考已完成」，正文在下方流式渲染
              bubble.dataset.filled = '1';
              bubble.classList.remove('gen-stage');
              bubble.classList.add('md');
              var status = bubble.querySelector('.gen-status');
              if (status) {
                status.classList.add('done');
                var label = status.querySelector('.gen-label');
                if (label) label.textContent = '思考已完成';
              }
              bubble.appendChild(el('div', 'gen-content'));
            }
            var content = bubble.querySelector('.gen-content') || bubble;
            content.innerHTML = window.renderMarkdown(full);
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
    // 有内容时状态行保留：小球 + 「思考已完成」常驻在正文上方
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
    returnAdminBtn.addEventListener('click', returnToAdmin);
    newChatBtn.addEventListener('click', function () { if (!state.streaming) startNewChat(); });
    sendBtn.addEventListener('click', sendMessage);
    stopBtn.addEventListener('click', stopStreaming);
    input.addEventListener('keydown', handleKeydown);
    input.addEventListener('input', handleInput);

    /* 额度页入口（点击用户栏额度区）与返回；管理页入口与返回 */
    quotaRow.addEventListener('click', showQuotaView);
    quotaBackBtn.addEventListener('click', showChatView);
    adminEntryBtn.addEventListener('click', function () {
      closeSidebar();
      showAdminView();
    });
    adminBackBtn.addEventListener('click', showChatView);
    questionsBackBtn.addEventListener('click', showUsersPanel);
    questionsMore.addEventListener('click', loadQuestions);

    /* 移动端抽屉：汉堡打开 / 遮罩与 Esc 关闭 */
    menuToggle.addEventListener('click', openSidebar);
    sidebarMask.addEventListener('click', closeSidebar);

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
      if (ev.key === 'Escape') {
        if (!isHidden(convMenu)) closeConvMenu();
        closeSidebar();
      }
    });
  }

  async function init() {
    applyBranding();
    bindEvents();
    initHero($('hero-auth'));
    initHero($('hero-welcome'));
    setAuthMode('login');
    closeConvMenu();
    closeSidebar();
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
