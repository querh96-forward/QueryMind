const api = async (path, options = {}) => {
  const response = await fetch(path, {headers: {'Content-Type': 'application/json', ...(options.headers || {})}, ...options});
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response;
};

const state = {
  currentThread: localStorage.getItem('qm_thread') || null,
  tokenBudget: Number(localStorage.getItem('qm_budget') || 20000),
  lastTokens: {input: 0, output: 0, embedding: 0, total: 0, budget: 20000, usage_ratio: 0, estimated_cost: 0},
  charts: {},
  running: false,
  runTimer: null,
  runStartedAt: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const fmt = (value, digits = 2) => typeof value === 'number' ? value.toLocaleString('zh-CN', {maximumFractionDigits: digits}) : (value ?? '—');
const terminal = new Set(['completed', 'completed_with_warning', 'failed', 'canceled']);

function setView(name) {
  $$('.view').forEach(v => v.classList.remove('active'));
  $$('.nav-item').forEach(v => v.classList.toggle('active', v.dataset.view === name));
  $(`#view-${name}`).classList.add('active');
  if (name === 'business') loadBusiness();
  if (name === 'system') loadSystem();
  if (name === 'source') loadSource();
  if (name === 'metrics') loadMetrics();
  if (name === 'settings') loadSettings();
  if (window.innerWidth < 760) $('#sidebar').classList.remove('open');
}

async function loadThreads() {
  const threads = await api('/api/v1/threads');
  const list = $('#threadList');
  list.innerHTML = '';
  threads.forEach(thread => {
    const row = document.createElement('div');
    row.className = `thread-item ${thread.id === state.currentThread ? 'active' : ''}`;
    row.title = thread.title;
    row.innerHTML = `<span class="thread-title">${escapeHtml(thread.title)}</span><button class="thread-delete" title="删除">×</button>`;
    row.querySelector('.thread-title').onclick = () => openThread(thread.id);
    row.querySelector('.thread-delete').onclick = async (event) => {
      event.stopPropagation();
      if (!confirm(`删除对话“${thread.title}”？`)) return;
      await api(`/api/v1/threads/${thread.id}`, {method: 'DELETE'});
      if (state.currentThread === thread.id) {
        state.currentThread = null;
        localStorage.removeItem('qm_thread');
        clearChat();
      }
      await loadThreads();
    };
    list.appendChild(row);
  });
}

async function createThread() {
  const thread = await api('/api/v1/threads', {method: 'POST', body: JSON.stringify({title: '新分析'})});
  state.currentThread = thread.id;
  localStorage.setItem('qm_thread', thread.id);
  clearChat();
  await loadThreads();
  setView('chat');
  $('#questionInput').focus();
  return thread.id;
}

async function openThread(id) {
  state.currentThread = id;
  localStorage.setItem('qm_thread', id);
  setView('chat');
  await loadMessages();
  await loadThreads();
}

function clearChat() {
  $('#messages').innerHTML = '';
  $('#welcome').style.display = '';
  setProgress('');
  setTokenDisplay({input:0, output:0, embedding:0, total:0, budget:state.tokenBudget, usage_ratio:0, estimated_cost:0});
}

async function loadMessages() {
  if (!state.currentThread) return clearChat();
  const messages = await api(`/api/v1/threads/${state.currentThread}/messages`);
  $('#messages').innerHTML = '';
  $('#welcome').style.display = messages.length ? 'none' : '';
  messages.forEach(renderMessage);
  const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant' && m.payload?.tokens);
  if (lastAssistant) setTokenDisplay(lastAssistant.payload.tokens);
  scrollBottom();
}

function renderMessage(message) {
  const wrap = document.createElement('div');
  wrap.className = `message ${message.role}`;
  if (message.role === 'user') {
    wrap.innerHTML = `<div class="bubble">${escapeHtml(message.content).replace(/\n/g,'<br>')}</div>`;
  } else {
    wrap.innerHTML = `<div class="avatar">Q</div><div class="assistant-block"></div>`;
    renderAssistant(wrap.querySelector('.assistant-block'), message.content, message.payload || {}, message.run_id);
  }
  $('#messages').appendChild(wrap);
}

function processHtml(process) {
  const steps = process?.steps || [];
  if (!steps.length) return '';
  const seconds = Number(process.duration_ms || 0) / 1000;
  const meta = [
    seconds ? `${fmt(seconds, 1)}秒` : '',
    process.loop_count ? `${process.loop_count}轮` : '',
    process.tool_call_count ? `${process.tool_call_count}次工具调用` : '',
  ].filter(Boolean).join(' · ');
  return `<details class="analysis-process">
    <summary><span class="process-summary-icon">✓</span><span>已完成分析</span>${meta ? `<small>${escapeHtml(meta)}</small>` : ''}</summary>
    <div class="process-history">${steps.map(step => `<div class="process-history-item"><span>✓</span><div>${escapeHtml(step.message)}</div></div>`).join('')}</div>
  </details>`;
}

function renderAssistant(container, content, payload, runId) {
  const cards = payload.cards || [];
  const table = payload.table || {columns: [], rows: []};
  const evidence = payload.evidence || {};
  const chartId = `chart-${runId || Math.random().toString(36).slice(2)}`;
  container.innerHTML = `
    <div class="bubble">
      ${processHtml(payload.process)}
      <div class="assistant-summary">${escapeHtml(payload.summary || content).replace(/\n/g,'<br>')}</div>
      ${cards.length ? `<div class="result-cards">${cards.map(c => `<div class="result-card"><span>${escapeHtml(c.label)}</span><strong>${fmt(c.value)}</strong></div>`).join('')}</div>` : ''}
      ${payload.chart ? `<div id="${chartId}" class="inline-chart"></div>` : ''}
      ${table.rows?.length ? tableHtml(table) : ''}
      <div class="answer-meta"><span>${payload.tokens ? `本次 ${fmt(payload.tokens.total,0)} Token · 预估费用 ${fmt(payload.tokens.estimated_cost,6)}` : ''}</span>${evidence.verification_passed ? '<span>✓ 查询执行完成</span>' : ''}</div>
      <div class="answer-actions">
        ${evidence.sql ? '<button data-evidence>查看数据依据</button>' : ''}
        ${table.rows?.length && runId ? `<button data-download>导出CSV</button>` : ''}
        ${runId ? '<button data-helpful="true">有帮助</button><button data-helpful="false">没帮助</button>' : ''}
      </div>
      <div class="evidence">
        <div><strong>数据源：</strong>${escapeHtml(evidence.source || '—')}</div>
        <div><strong>使用对象：</strong>${escapeHtml((evidence.tables || []).join('、') || '—')}</div>
        <div><strong>返回行数：</strong>${fmt(evidence.row_count,0)}　<strong>查询耗时：</strong>${fmt(evidence.query_latency_ms)} ms</div>
        ${evidence.sql ? `<pre>${escapeHtml(evidence.sql)}</pre>` : ''}
      </div>
    </div>`;
  const evidenceBtn = container.querySelector('[data-evidence]');
  if (evidenceBtn) evidenceBtn.onclick = () => container.querySelector('.evidence').classList.toggle('open');
  const downloadBtn = container.querySelector('[data-download]');
  if (downloadBtn) downloadBtn.onclick = () => window.location.href = `/api/v1/runs/${runId}/export.csv`;
  container.querySelectorAll('[data-helpful]').forEach(btn => btn.onclick = async () => {
    await api(`/api/v1/runs/${runId}/feedback`, {method:'POST', body: JSON.stringify({helpful: btn.dataset.helpful === 'true'})});
    btn.textContent = '已记录';
  });
  if (payload.chart) setTimeout(() => drawAnswerChart(chartId, payload.chart), 0);
}

function tableHtml(table) {
  return `<div class="result-table-wrap"><table><thead><tr>${table.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${table.rows.slice(0,100).map(row => `<tr>${table.columns.map(c => `<td>${escapeHtml(fmt(row[c]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function fallbackChart(element, x, y) {
  const max = Math.max(...y.map(Number), 1);
  element.innerHTML = `<div class="fallback-bars">${y.map((value,i)=>`<div class="fallback-bar" title="${escapeHtml(x[i])}: ${fmt(value)}" style="height:${Math.max(4, Number(value)/max*100)}%"><span>${escapeHtml(String(x[i]).slice(0,8))}</span></div>`).join('')}</div><div class="fallback-note">当前未加载ECharts，已使用轻量柱状图展示。</div>`;
}

function drawAnswerChart(id, chart) {
  const element = document.getElementById(id);
  if (!element) return;
  if (!window.echarts) return fallbackChart(element, chart.x, chart.y);
  const instance = echarts.init(element);
  instance.setOption({
    tooltip: {trigger:'axis'}, grid:{left:72,right:20,top:30,bottom:55},
    xAxis:{type:'category',data:chart.x,axisLabel:{rotate:chart.x.length>10?35:0}},
    yAxis:{type:'value'},
    series:[{type:chart.type,data:chart.y,smooth:true,barMaxWidth:34,itemStyle:{borderRadius:[4,4,0,0]},lineStyle:{width:2}}]
  });
  state.charts[id] = instance;
}

function liveRunElement() { return $('#run-loading'); }

function liveProcessStep(type, message, status = 'active') {
  const live = liveRunElement();
  if (!live || !message) return;
  const list = live.querySelector('.live-steps');
  const previous = list.querySelector('.live-step.active');
  if (previous) previous.classList.replace('active', 'done');
  const last = list.lastElementChild;
  if (last && last.dataset.type === type && last.querySelector('.live-step-text')?.textContent === message) {
    last.classList.remove('done');
    last.classList.add(status);
    return;
  }
  const row = document.createElement('div');
  row.className = `live-step ${status}`;
  row.dataset.type = type;
  row.innerHTML = `<span class="live-step-dot"></span><span class="live-step-text"></span>`;
  row.querySelector('.live-step-text').textContent = message;
  list.appendChild(row);
  const statusText = live.querySelector('.live-process-title');
  if (statusText) statusText.textContent = message;
  scrollBottom();
}

function resetLiveAnswer() {
  const live = liveRunElement();
  if (!live) return;
  const answer = live.querySelector('.live-answer');
  answer.textContent = '';
  answer.classList.remove('visible', 'finished');
}

function appendLiveAnswer(delta) {
  const live = liveRunElement();
  if (!live || !delta) return;
  const answer = live.querySelector('.live-answer');
  answer.classList.add('visible');
  answer.appendChild(document.createTextNode(delta));
  scrollBottom();
}

function finishLiveAnswer() {
  const live = liveRunElement();
  if (!live) return;
  live.querySelector('.live-answer')?.classList.add('finished');
}

function stopRunTimer() {
  if (state.runTimer) clearInterval(state.runTimer);
  state.runTimer = null;
}

async function submitQuestion(question) {
  if (state.running) return;
  if (!state.currentThread) await createThread();
  state.running = true;
  state.runStartedAt = Date.now();
  $('#sendButton').disabled = true;
  $('#welcome').style.display = 'none';
  renderMessage({role:'user', content:question});
  const loading = document.createElement('div');
  loading.className = 'message assistant';
  loading.id = 'run-loading';
  loading.innerHTML = `<div class="avatar">Q</div><div class="assistant-block"><div class="bubble live-run">
    <div class="live-process-head"><span class="live-spinner"></span><strong class="live-process-title">正在接收问题</strong><span class="live-elapsed">0秒</span></div>
    <div class="live-steps"></div>
    <div class="live-answer" aria-live="polite"></div>
  </div></div>`;
  $('#messages').appendChild(loading);
  liveProcessStep('queued', '正在提交分析任务');
  state.runTimer = setInterval(() => {
    const elapsed = liveRunElement()?.querySelector('.live-elapsed');
    if (elapsed) elapsed.textContent = `${Math.max(0, Math.floor((Date.now() - state.runStartedAt) / 1000))}秒`;
  }, 1000);
  scrollBottom();
  try {
    const created = await api(`/api/v1/threads/${state.currentThread}/analyses`, {method:'POST', body: JSON.stringify({question, token_budget:state.tokenBudget})});
    watchRun(created.run_id);
  } catch (error) {
    stopRunTimer();
    loading.remove();
    renderMessage({role:'assistant', content:`提交失败：${error.message}`, payload:{}});
    finishRunning();
  }
}

function watchRun(runId) {
  const source = new EventSource(`/api/v1/runs/${runId}/events`);
  const labels = {
    queued:'问题已提交', started:'开始分析问题', context_building:'正在理解问题和业务口径', context_ready:'业务上下文已准备',
    thinking:'正在选择合适的数据工具', tool_running:'正在查询经营数据', tool_finished:'数据查询完成', verifying:'正在核对数据结果',
    replanning:'正在调整分析方法', verified:'结果核对完成', completed:'分析完成', failed:'分析失败', canceled:'任务已取消'
  };
  const processTypes = Object.keys(labels);
  processTypes.forEach(type => source.addEventListener(type, event => {
    let payload = {};
    try { payload = JSON.parse(event.data); } catch (_) {}
    const message = payload.data?.message || labels[type];
    const status = ['completed','verified'].includes(type) ? 'done' : (type === 'failed' ? 'error' : 'active');
    liveProcessStep(type, message, status);
    setProgress('');
    if (type === 'failed' && payload.data?.error) {
      appendLiveAnswer(`分析失败：${payload.data.error}`);
      finishLiveAnswer();
    }
  }));
  source.addEventListener('answer_reset', () => resetLiveAnswer());
  source.addEventListener('answer_start', () => {
    const live = liveRunElement();
    if (live) live.querySelector('.live-process-title').textContent = '正在生成回答';
  });
  source.addEventListener('answer_delta', event => {
    try { appendLiveAnswer(JSON.parse(event.data).data?.delta || ''); } catch (_) {}
  });
  source.addEventListener('answer_end', () => finishLiveAnswer());
  source.addEventListener('done', async () => {
    source.close();
    stopRunTimer();
    finishLiveAnswer();
    await new Promise(resolve => setTimeout(resolve, 220));
    $('#run-loading')?.remove();
    await loadMessages();
    await loadThreads();
    finishRunning();
  });
  source.onerror = async () => {
    source.close();
    stopRunTimer();
    const run = await api(`/api/v1/runs/${runId}`).catch(() => null);
    $('#run-loading')?.remove();
    if (run && terminal.has(run.status)) {
      await loadMessages();
      if (run.status === 'failed') renderMessage({role:'assistant', content:`分析失败：${run.error || '请查看Worker日志'}`, payload:{}});
    } else {
      renderMessage({role:'assistant', content:'实时连接已中断，请刷新对话查看结果。', payload:{}});
    }
    finishRunning();
  };
}

function finishRunning() { stopRunTimer(); state.running = false; $('#sendButton').disabled = false; setProgress(''); }
function setProgress(text) { $('#progressText').textContent = text; }
function scrollBottom() { const area=$('#chatArea'); setTimeout(()=>area.scrollTop=area.scrollHeight,20); }

function setTokenDisplay(tokens) {
  state.lastTokens = {...state.lastTokens, ...tokens};
  $('#tokenPill').textContent = `本次用量 ${fmt(state.lastTokens.total,0)} / ${fmt(state.lastTokens.budget || state.tokenBudget,0)} Token`;
  $('#tokenDetails').innerHTML = summaryRows([
    ['输入Token', fmt(state.lastTokens.input,0)], ['输出Token', fmt(state.lastTokens.output,0)], ['Embedding Token', fmt(state.lastTokens.embedding,0)],
    ['累计Token', fmt(state.lastTokens.total,0)], ['预算使用率', `${fmt(state.lastTokens.usage_ratio)}%`], ['预估费用', fmt(state.lastTokens.estimated_cost,6)]
  ]);
}

function summaryRows(rows) { return rows.map(([a,b]) => `<div class="summary-row"><span>${escapeHtml(a)}</span><strong>${escapeHtml(b)}</strong></div>`).join(''); }
function disposeChart(id) { if(state.charts[id]) { state.charts[id].dispose(); delete state.charts[id]; } }
function basicChart(id, x, series, type='bar') {
  disposeChart(id); const el=$(`#${id}`); if(!el)return;
  if (!window.echarts) return fallbackChart(el, x, series[0]?.data || []);
  const chart=echarts.init(el); state.charts[id]=chart;
  chart.setOption({tooltip:{trigger:'axis'},legend:{bottom:0},grid:{left:72,right:20,top:25,bottom:50},xAxis:{type:'category',data:x,axisLabel:{rotate:x.length>12?35:0}},yAxis:{type:'value'},series:series.map(s=>({name:s.name,type:s.type||type,data:s.data,smooth:true,barMaxWidth:30}))});
}

async function loadBusiness() {
  $('#businessCards').innerHTML = '<div class="loading"></div>'.repeat(4);
  const data = await api('/api/v1/dashboard/business');
  const cards = [
    ['销售额', data.cards.sales_amount, '扣除折扣后的销售额'], ['订单量', data.cards.order_count, '去重订单数'],
    ['客单价', data.cards.avg_order_value, '平均每张订单'], ['按期发货率', `${fmt(data.cards.on_time_rate)}%`, '要求日期前发货']
  ];
  $('#businessCards').innerHTML = cards.map(c=>`<div class="metric-card"><span>${c[0]}</span><strong>${fmt(c[1])}</strong><small>${c[2]}</small></div>`).join('');
  basicChart('monthlyChart', data.monthly.map(x=>x.month), [{name:'销售额',data:data.monthly.map(x=>x.sales_amount),type:'line'},{name:'订单量',data:data.monthly.map(x=>x.order_count),type:'bar'}]);
  basicChart('categoryChart', data.categories.map(x=>x.category_name), [{name:'销售额',data:data.categories.map(x=>x.sales_amount)}]);
  basicChart('countryChart', data.countries.map(x=>x.country), [{name:'销售额',data:data.countries.map(x=>x.sales_amount)}]);
}

async function loadSystem() {
  const data = await api(`/api/v1/dashboard/system?days=${$('#metricDays').value}`);
  const cards = [
    ['请求成功率', `${fmt(data.request_success_rate)}%`, `${data.sample_count}次已结束运行`],
    ['任务完成率', `${fmt(data.task_completion_rate)}%`, '执行完整性检查通过'],
    ['P95响应延迟', `${fmt(data.p95_latency_ms/1000)}秒`, '95%任务完成时间'],
    ['首次SQL成功率', `${fmt(data.first_sql_success_rate)}%`, '首次查询直接成功'],
    ['执行检查通过率', `${fmt(data.verification_pass_rate)}%`, '已完成任务口径'],
    ['用户满意度', `${fmt(data.satisfaction_rate)}%`, `${data.feedback_count}条反馈`],
    ['累计Token', fmt(data.total_tokens,0), `最近${data.period_days}天`],
    ['预估费用', fmt(data.estimated_cost,6), '按本地配置估算'],
  ];
  $('#systemCards').innerHTML = cards.map(c=>`<div class="metric-card"><span>${c[0]}</span><strong>${c[1]}</strong><small>${c[2]}</small></div>`).join('');
  basicChart('systemTrendChart', data.daily.map(x=>x.date), [{name:'运行次数',data:data.daily.map(x=>x.runs),type:'bar'},{name:'Token/1000',data:data.daily.map(x=>x.tokens/1000),type:'line'}]);
  $('#costSummary').innerHTML = summaryRows([['累计Token',fmt(data.total_tokens,0)],['预估费用',fmt(data.estimated_cost,6)],['平均循环次数',fmt(data.avg_loops)],['平均工具调用',fmt(data.avg_tool_calls)]]);
  const ev=data.offline_evaluation;
  $('#evalSummary').innerHTML = ev ? summaryRows([
    ['评测集',data.offline_evaluation_name],['样本数',data.offline_evaluation_cases],['结果准确率',`${fmt(ev.result_accuracy || 0)}%`],['RAG Recall@5',fmt(ev.rag_recall_at_5 || 0)]
  ]) : '<p class="muted">尚未运行离线评测。执行 <code>python -m evals.run_eval</code> 后显示结果。</p>';
}

async function loadSource() {
  const data=await api('/api/v1/data-source');
  $('#sourceCard').innerHTML=`<div class="source-main"><h2>${escapeHtml(data.name)} <span class="online-dot"></span></h2><p>${escapeHtml(data.type)} · 只读连接 · 状态${escapeHtml(data.status)}</p></div><div class="source-stat"><strong>${fmt(data.total_rows,0)}</strong><span>核心业务数据行</span></div>`;
  const labels = {
    categories:'商品品类', customers:'客户', employees:'销售员工', orders:'订单',
    order_details:'订单明细', products:'商品', shippers:'物流商', suppliers:'供应商',
    v_order_line_sales:'订单销售明细', v_order_summary:'订单汇总',
    v_product_sales:'商品销售汇总', v_customer_sales:'客户销售汇总',
    v_inventory_status:'库存状态',
  };
  // The API returns schema entries as an array; row_counts covers base tables only.
  $('#tableCatalog').innerHTML=data.tables.map(({name})=>{
    const count = data.row_counts[name];
    const detail = count === undefined ? '分析视图' : `${fmt(count,0)} 行`;
    return `<div class="catalog-item"><strong>${escapeHtml(labels[name] || name)}</strong><small>${escapeHtml(name)} · ${detail}</small></div>`;
  }).join('');
}

async function loadMetrics() {
  const data=await api('/api/v1/metric-definitions');
  $('#metricDefinitions').innerHTML=data.map(x=>`<div class="definition-item"><strong>${escapeHtml(x.name)}</strong><div><div class="formula">${escapeHtml(x.formula)}</div><div class="muted">${escapeHtml(x.description)}</div></div></div>`).join('');
}

async function loadSettings() {
  $('#budgetInput').value=state.tokenBudget;
  const memories=await api('/api/v1/memories');
  $('#memoryList').innerHTML=memories.length?memories.map(m=>`<div class="memory-item"><div><strong>${escapeHtml(m.key)}</strong><div class="muted">${escapeHtml(m.value)}</div></div><button data-memory="${m.id}">删除</button></div>`).join(''):'<p class="muted">暂无长期偏好。</p>';
  $$('[data-memory]').forEach(btn=>btn.onclick=async()=>{await api(`/api/v1/memories/${btn.dataset.memory}`,{method:'DELETE'});loadSettings();});
}

$('#composer').addEventListener('submit', event => { event.preventDefault(); const input=$('#questionInput'); const q=input.value.trim(); if(q){input.value='';input.style.height='auto';submitQuestion(q);} });
$('#questionInput').addEventListener('keydown', event => { if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();$('#composer').requestSubmit();} });
$('#questionInput').addEventListener('input', event => {event.target.style.height='auto';event.target.style.height=Math.min(event.target.scrollHeight,130)+'px';});
$$('.suggestions button').forEach(btn=>btn.onclick=()=>submitQuestion(btn.textContent));
$$('.nav-item').forEach(btn=>btn.onclick=()=>setView(btn.dataset.view));
$$('[data-view-link]').forEach(btn=>btn.onclick=()=>setView(btn.dataset.viewLink));
$('#newThread').onclick=createThread; $('#refreshThreads').onclick=loadThreads; $('#refreshBusiness').onclick=loadBusiness;
$('#metricDays').onchange=loadSystem;
$('#toggleSidebar').onclick=()=>{if(window.innerWidth<760)$('#sidebar').classList.toggle('open');else document.body.classList.toggle('sidebar-collapsed');};
$('#closeSidebar').onclick=()=>$('#sidebar').classList.remove('open');
$('#tokenPill').onclick=()=>$('#tokenModal').classList.add('open');
$('#budgetInline').onclick=()=>setView('settings');
$('[data-close-modal]').onclick=()=>$('#tokenModal').classList.remove('open');
$('#tokenModal').onclick=e=>{if(e.target.id==='tokenModal')e.currentTarget.classList.remove('open');};
$('#saveSettings').onclick=()=>{state.tokenBudget=Math.max(2000,Math.min(100000,Number($('#budgetInput').value)||20000));localStorage.setItem('qm_budget',state.tokenBudget);$('#budgetInline').textContent=`${fmt(state.tokenBudget,0)} Token`;alert('已保存');};
window.addEventListener('resize',()=>Object.values(state.charts).forEach(c=>c.resize()));

(async function init(){
  const config=await api('/api/v1/public-config').catch(()=>({default_token_budget:20000}));
  if(!localStorage.getItem('qm_budget'))state.tokenBudget=config.default_token_budget;
  $('#budgetInline').textContent=`${fmt(state.tokenBudget,0)} Token`;
  setTokenDisplay({...state.lastTokens,budget:state.tokenBudget});
  await loadThreads();
  if(state.currentThread) await loadMessages().catch(()=>{state.currentThread=null;clearChat();});
})();
