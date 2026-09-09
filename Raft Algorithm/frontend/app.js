let latest = null;

async function fetchSnapshot() {
  try {
    const res = await fetch('/api/snapshot');
    latest = await res.json();
    render(latest);
  } catch (e) {
    console.error(e);
  }
}

async function sendCommand(payload) {
  await fetch('/api/control', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload)
  });
  setTimeout(fetchSnapshot, 150);
}

function nodeAction(action) {
  sendCommand({ action, node_id: Number(document.getElementById('nodeSelect').value) });
}

function updateNetwork() {
  sendCommand({
    action: 'network',
    base_latency_ms: Number(document.getElementById('latency').value),
    latency_jitter_ms: Number(document.getElementById('jitter').value),
    packet_loss: Number(document.getElementById('loss').value)
  });
}

function majorityPartition() {
  if (!latest) return;
  const ids = Object.keys(latest.nodes).map(Number).sort((a, b) => a - b);
  const mid = Math.ceil(ids.length / 2) + 1;
  sendCommand({ action: 'partition', groups: [ids.slice(0, mid), ids.slice(mid)] });
}

function render(s) {
  const m = s.metrics;
  document.getElementById('subtitle').textContent = `policy=${s.config.policy_mode} leader=${m.leader_id ?? 'none'} term=${m.current_term}`;

  const select = document.getElementById('nodeSelect');
  const selected = select.value;
  select.innerHTML = Object.keys(s.nodes).map(id => `<option value="${id}">Node ${id}</option>`).join('');
  if (selected && Object.keys(s.nodes).includes(selected)) select.value = selected;

  // Sort nodes: leader always first (top-left in grid), rest by node_id
  const sortedNodes = Object.values(s.nodes).sort((a, b) => {
    if (a.state === 'leader' && b.state !== 'leader') return -1;
    if (a.state !== 'leader' && b.state === 'leader') return 1;
    return a.node_id - b.node_id;
  });

  document.getElementById('nodes').innerHTML = sortedNodes.map(n => `
    <div class="node ${n.state === 'leader' ? 'leader' : ''} ${n.active ? '' : 'dead'}">
      <div class="node-title"><span>Node ${n.node_id}</span><span class="pill ${n.active ? 'active-pill' : ''}">${n.active ? 'active' : 'crashed'}</span></div>
      <div class="role">${n.state}</div>
      <div class="node-stat">term: ${n.term}</div>
      <div class="node-stat">ldr: ${n.leader_id ?? '-'}</div>
      <div class="node-stat">log: ${n.log_len}</div>
      <div class="node-stat">rtt: ${Math.round(n.estimated_rtt_ms || 0)}ms</div>
    </div>`).join('');

  const metricRows = [
    ['Nodes', Object.keys(s.nodes).length],
    ['Leader', m.leader_id ?? 'none'],
    ['Elections', m.elections_started],
    ['Split votes', m.split_votes],
    ['Delivered', m.messages_delivered],
    ['Dropped', m.messages_dropped],
    ['Failover s', fmt(m.last_failover_time_s)],
    ['Stability s', fmt(m.leader_stability_s)]
  ];
  document.getElementById('metrics').innerHTML = metricRows.map(([k, v]) => `<div class="metric">${k}<b>${v}</b></div>`).join('');
  document.getElementById('events').innerHTML = s.recent_events.slice().reverse().map(e => `<div>${e.wall_time || ''} ${e.event} ${JSON.stringify(e)}</div>`).join('');

  const net = s.network;
  if (document.activeElement !== document.getElementById('latency')) document.getElementById('latency').value = net.base_latency_ms ?? 0;
  if (document.activeElement !== document.getElementById('jitter')) document.getElementById('jitter').value = net.latency_jitter_ms ?? 0;
  if (document.activeElement !== document.getElementById('loss')) document.getElementById('loss').value = net.packet_loss ?? 0;

  // ── Populate Pipeline Dropdown & Stats ────────────────
  const electSelect = document.getElementById('electionSelect');
  if (electSelect) {
    const prevSelected = electSelect.value;
    const history = s.election_history || [];
    
    if (history.length === 0) {
      electSelect.innerHTML = '<option value="">No elections yet</option>';
    } else {
      electSelect.innerHTML = history.map(h => 
        `<option value="${h.term}">Term ${h.term} (${h.status.toUpperCase()})</option>`
      ).join('');
      if (prevSelected && history.some(h => String(h.term) === prevSelected)) {
        electSelect.value = prevSelected;
      }
    }

    // Update statistics
    document.getElementById('pipe-total-elections').textContent = m.elections_started;
    const splitRate = m.elections_started > 0 ? ((m.split_votes / m.elections_started) * 100).toFixed(0) + '%' : '0%';
    document.getElementById('pipe-split-rate').textContent = splitRate;
    document.getElementById('pipe-avg-dur').textContent = m.last_election_duration_s ? m.last_election_duration_s.toFixed(2) + 's' : '-';

    // Calculate how many pre-vote attempts were blocked
    let blockedCount = 0;
    history.forEach(h => {
      // Any pre-candidate that didn't make it to candidate was blocked
      const preCandSet = new Set(h.pre_candidates || []);
      const candSet = new Set(h.candidates || []);
      preCandSet.forEach(id => {
        if (!candSet.has(id)) blockedCount++;
      });
    });
    document.getElementById('pipe-blocked').textContent = blockedCount;

    if (activePage === 'pipeline') {
      renderElectionDetails();
    }
  }
}

// ── Two-Page Navigation Toggles ───────────────────────
let activePage = 'dashboard';

function switchPage(page) {
  activePage = page;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  const targetBtn = document.getElementById(`tab-${page}`);
  if (targetBtn) targetBtn.classList.add('active');

  if (page === 'dashboard') {
    document.getElementById('page-dashboard').classList.remove('hidden');
    document.getElementById('page-pipeline').classList.add('hidden');
  } else {
    document.getElementById('page-dashboard').classList.add('hidden');
    document.getElementById('page-pipeline').classList.remove('hidden');
    renderElectionDetails();
  }
}

function selectElection(term) {
  renderElectionDetails();
}

function renderElectionDetails() {
  if (!latest || !latest.election_history) return;
  const select = document.getElementById('electionSelect');
  if (!select) return;
  const termVal = Number(select.value);
  if (!termVal) {
    document.getElementById('election-funnel').innerHTML = `
      <div class="pipeline-placeholder">No election history recorded yet. Trigger a timeout or partition to start an election.</div>`;
    return;
  }

  const h = latest.election_history.find(x => x.term === termVal);
  if (!h) {
    document.getElementById('election-funnel').innerHTML = `
      <div class="pipeline-placeholder">Selected election details not found.</div>`;
    return;
  }

  const majorityNeeded = Math.ceil(Object.keys(latest.nodes).length / 2) + 1;

  let funnelHTML = '';

  // ━━━━━━━━ STAGE 1: CLUSTER CONFIG ━━━━━━━━
  funnelHTML += `
    <div class="funnel-step">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">1</span>
          Stage 1: Followers (Cluster Scope)
        </div>
        <span class="funnel-step-badge">${Object.keys(latest.nodes).length} Nodes</span>
      </div>
      <p style="margin:0;color:var(--muted);font-size:12px;line-height:1.45;">
        All nodes are active followers listening to heartbeats. Total cluster size is <b>${Object.keys(latest.nodes).length} nodes</b>. Majority threshold for decision is <b>${majorityNeeded} nodes</b>.
      </p>
    </div>
  `;

  // ━━━━━━━━ STAGE 2: PRE-CANDIDATES (TIMEOUTS) ━━━━━━━━
  const preCands = h.pre_candidates || [];
  let preCandHTML = '';
  if (preCands.length === 0) {
    preCandHTML = `<p style="margin:0;color:var(--muted);font-size:12px;">No nodes timed out during this term yet.</p>`;
  } else {
    preCandHTML = `
      <p style="margin:0 0 10px;color:var(--muted);font-size:12px;">
        Follower timeouts expired on these nodes. They entered <b>Pre-Candidate</b> state:
      </p>
      <div class="funnel-nodes">
        ${preCands.map(id => `<span class="node-circle pre-candidate">N${id}</span>`).join('')}
      </div>
    `;
  }
  funnelHTML += `
    <div class="funnel-step">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">2</span>
          Stage 2: Timeout Trigger (Pre-Candidates)
        </div>
        <span class="funnel-step-badge">${preCands.length} Triggered</span>
      </div>
      ${preCandHTML}
    </div>
  `;

  // ━━━━━━━━ STAGE 3: PRE-VOTE FUNNEL (ELIMINATIONS) ━━━━━━━━
  let preVoteFunnelHTML = '';
  if (preCands.length === 0) {
    preVoteFunnelHTML = `<p style="margin:0;color:var(--muted);font-size:12px;">No pre-candidate data available.</p>`;
  } else {
    preVoteFunnelHTML = `
      <p style="margin:0 0 12px;color:var(--muted);font-size:12px;">
        Pre-Candidates requested pre-votes from peers. If a peer could hear a leader or if the pre-candidate has stale logs, the pre-vote was **denied**:
      </p>
      <div style="display:flex;flex-direction:column;gap:12px;width:100%;">
    `;

    const candSet = new Set(h.candidates || []);

    preCands.forEach(candId => {
      const votes = h.pre_votes[String(candId)] || {};
      const voteEntries = Object.entries(votes);
      const grants = voteEntries.filter(([_, v]) => v.granted);
      const denials = voteEntries.filter(([_, v]) => !v.granted);
      
      const actuallyAdvanced = candSet.has(candId);
      const hasPreVoteMajority = grants.length + 1 >= majorityNeeded;

      // Aggregate denial reasons
      const reasonCounts = {};
      denials.forEach(([_, v]) => {
        reasonCounts[v.reason] = (reasonCounts[v.reason] || 0) + 1;
      });
      const reasonSummary = Object.entries(reasonCounts)
        .map(([r, count]) => `<span class="vote-reason-tag">${r}: ${count}</span>`)
        .join(' ');

      let resultText = '';
      if (actuallyAdvanced) {
        resultText = '<span style="color:var(--good);font-weight:600;">Advanced to Candidate</span>';
      } else if (hasPreVoteMajority) {
        resultText = '<span style="color:var(--danger);font-weight:600;">Eliminated (Stepped down for faster candidate)</span>';
      } else {
        resultText = '<span style="color:var(--danger);font-weight:600;">Eliminated (No pre-vote majority)</span>';
      }

      preVoteFunnelHTML += `
        <div class="pipeline-candidate-card" style="border-left: 3px solid ${actuallyAdvanced ? 'var(--good)' : 'var(--danger)'}">
          <div class="candidate-card-header">
            <span class="node-circle pre-candidate" style="width:22px;height:22px;font-size:9px;">N${candId}</span>
            <span>Pre-Candidate Campaign</span>
            <span class="funnel-step-badge" style="background:${actuallyAdvanced ? 'var(--good-dim)' : 'var(--danger-dim)'};color:${actuallyAdvanced ? 'var(--good)' : 'var(--danger)'};">
              ${grants.length + 1} / ${majorityNeeded} Votes
            </span>
          </div>
          <div style="font-size:11px;color:var(--muted);">
            <b>Result:</b> ${resultText}
          </div>
          ${denials.length > 0 ? `
            <div class="vote-reason-summary">
              <b>Denial Reasons:</b> ${reasonSummary || 'none'}
            </div>
          ` : ''}
          <div class="candidate-card-votes">
            <span class="vote-micro-pill granted">Self (granted)</span>
            ${voteEntries.slice(0, 15).map(([vId, v]) => `
              <span class="vote-micro-pill ${v.granted ? 'granted' : 'denied'}">N${vId}</span>
            `).join('')}
            ${voteEntries.length > 15 ? `<span style="font-size:9px;color:var(--muted);padding-left:4px;">+${voteEntries.length - 15} more</span>` : ''}
          </div>
        </div>
      `;
    });

    preVoteFunnelHTML += `</div>`;
  }

  funnelHTML += `
    <div class="funnel-step">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">3</span>
          Stage 3: Pre-Vote Filter (Funnel Eliminator)
        </div>
        <span class="funnel-step-badge">Pipeline Shield</span>
      </div>
      ${preVoteFunnelHTML}
    </div>
  `;

  // ━━━━━━━━ STAGE 4: CANDIDATE RUNNERS ━━━━━━━━
  const cands = h.candidates || [];
  let candHTML = '';
  if (cands.length === 0) {
    candHTML = `
      <p style="margin:0;color:var(--muted);font-size:12px;">
        No nodes advanced past the pre-vote funnel. <b>0 elections were officially run</b>, saving network stability!
      </p>
    `;
  } else {
    candHTML = `
      <p style="margin:0 0 10px;color:var(--muted);font-size:12px;">
        These nodes achieved a majority in the pre-vote phase, incremented their terms, and became official <b>Candidates</b>:
      </p>
      <div class="funnel-nodes">
        ${cands.map(id => `<span class="node-circle candidate">N${id}</span>`).join('')}
      </div>
    `;
  }
  funnelHTML += `
    <div class="funnel-step">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">4</span>
          Stage 4: Official Candidates Campaign
        </div>
        <span class="funnel-step-badge">${cands.length} Advanced</span>
      </div>
      ${candHTML}
    </div>
  `;

  // ━━━━━━━━ STAGE 5: OFFICIAL VOTES ━━━━━━━━
  let voteFunnelHTML = '';
  if (cands.length === 0) {
    voteFunnelHTML = `<p style="margin:0;color:var(--muted);font-size:12px;">No official voting took place.</p>`;
  } else {
    voteFunnelHTML = `
      <p style="margin:0 0 12px;color:var(--muted);font-size:12px;">
        Candidates campaigned for votes. A peer grants an official vote only if its term is equal/lower, it hasn't voted yet, and the candidate's log is up-to-date:
      </p>
      <div style="display:flex;flex-direction:column;gap:12px;width:100%;">
    `;

    cands.forEach(candId => {
      const votes = h.votes[String(candId)] || {};
      const voteEntries = Object.entries(votes);
      const grants = voteEntries.filter(([_, v]) => v.granted);
      const denials = voteEntries.filter(([_, v]) => !v.granted);
      const won = h.winner === candId;

      // Aggregate denial reasons
      const reasonCounts = {};
      denials.forEach(([_, v]) => {
        reasonCounts[v.reason] = (reasonCounts[v.reason] || 0) + 1;
      });
      const reasonSummary = Object.entries(reasonCounts)
        .map(([r, count]) => `<span class="vote-reason-tag">${r}: ${count}</span>`)
        .join(' ');

      voteFunnelHTML += `
        <div class="pipeline-candidate-card" style="border-left: 3px solid ${won ? 'var(--good)' : 'var(--danger)'}">
          <div class="candidate-card-header">
            <span class="node-circle candidate" style="width:22px;height:22px;font-size:9px;">N${candId}</span>
            <span>Election Campaign</span>
            <span class="funnel-step-badge" style="background:${won ? 'var(--good-dim)' : 'var(--danger-dim)'};color:${won ? 'var(--good)' : 'var(--danger)'};">
              ${grants.length + 1} / ${majorityNeeded} Votes
            </span>
          </div>
          <div style="font-size:11px;color:var(--muted);">
            <b>Result:</b> ${won ? '<span style="color:var(--good);font-weight:600;">Elected Leader ★</span>' : '<span style="color:var(--danger);font-weight:600;">Lost Election</span>'}
          </div>
          ${denials.length > 0 ? `
            <div class="vote-reason-summary">
              <b>Denial Reasons:</b> ${reasonSummary || 'none'}
            </div>
          ` : ''}
          <div class="candidate-card-votes">
            <span class="vote-micro-pill granted">Self (granted)</span>
            ${voteEntries.slice(0, 15).map(([vId, v]) => `
              <span class="vote-micro-pill ${v.granted ? 'granted' : 'denied'}">N${vId}</span>
            `).join('')}
            ${voteEntries.length > 15 ? `<span style="font-size:9px;color:var(--muted);padding-left:4px;">+${voteEntries.length - 15} more</span>` : ''}
          </div>
        </div>
      `;
    });

    voteFunnelHTML += `</div>`;
  }

  funnelHTML += `
    <div class="funnel-step">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">5</span>
          Stage 5: Official Voting Filter
        </div>
        <span class="funnel-step-badge">Raft Election</span>
      </div>
      ${voteFunnelHTML}
    </div>
  `;

  // ━━━━━━━━ STAGE 6: OUTCOME ━━━━━━━━
  let outcomeHTML = '';
  if (h.status === 'success' && h.winner !== null) {
    outcomeHTML = `
      <div style="display:flex;align-items:center;gap:12px;">
        <span class="node-circle winner">N${h.winner}</span>
        <div>
          <b style="color:var(--good);font-size:14px;">ELECTION SUCCESSFUL</b>
          <p style="margin:4px 0 0;font-size:11px;color:var(--muted);">Node ${h.winner} gathered a majority vote and has successfully claimed the term leadership!</p>
        </div>
      </div>
    `;
  } else {
    outcomeHTML = `
      <div style="display:flex;align-items:center;gap:12px;">
        <span class="node-circle blocked" style="text-decoration:none;font-weight:700;">✗</span>
        <div>
          <b style="color:var(--danger);font-size:14px;">ELECTION FAILED / SPLIT VOTE</b>
          <p style="margin:4px 0 0;font-size:11px;color:var(--muted);">No node achieved the required majority vote threshold of ${majorityNeeded} nodes. Followers will time out again and retry pre-voting.</p>
        </div>
      </div>
    `;
  }
  funnelHTML += `
    <div class="funnel-step" style="border-color:${h.status === 'success' ? 'var(--good)' : 'var(--line)'}; background: ${h.status === 'success' ? 'rgba(46,204,113,0.03)' : 'var(--surface)'}">
      <div class="funnel-step-header">
        <div class="funnel-step-title">
          <span class="explainer-num" style="width:20px;height:20px;font-size:10px;">6</span>
          Stage 6: Final Outcome
        </div>
        <span class="funnel-step-badge" style="background:${h.status === 'success' ? 'var(--good-dim)' : 'var(--danger-dim)'};color:${h.status === 'success' ? 'var(--good)' : 'var(--danger)'};">
          ${h.status.toUpperCase()}
        </span>
      </div>
      ${outcomeHTML}
    </div>
  `;

  // ━━━━━━━━ STAGE 7: SUMMARY FUNNEL CHART ━━━━━━━━
  const finalLeader = h.winner !== null && h.status === 'success' ? [h.winner] : [];
  
  funnelHTML += `
    <div class="funnel-step" style="border-color: var(--accent); background: linear-gradient(180deg, var(--surface) 0%, rgba(108, 92, 231, 0.01) 100%);">
      <div class="funnel-step-header">
        <div class="funnel-step-title" style="color: var(--accent); font-weight:700;">
          <span class="explainer-num" style="width:20px;height:20px;font-size:11px;background:var(--accent-dim);color:var(--accent);border-color:rgba(108,92,231,0.25);">✓</span>
          Visual Pipeline Funnel Summary
        </div>
        <span class="funnel-step-badge" style="background:var(--accent-dim);color:var(--accent);border-color:rgba(108,92,231,0.2);">Funnel Overview</span>
      </div>
      <p style="margin:0 0 8px;color:var(--muted);font-size:11px;line-height:1.4;">
        Comparison of nodes that timed out, nodes that successfully gathered a pre-vote majority, and the final elected leader:
      </p>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 4px;">
        <!-- Pre-Candidates Column -->
        <div style="border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column;">
          <div style="font-weight:700;font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:8px;display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:6px;">
            <span>1. Pre-Candidates</span>
            <span style="color:#f39c12;font-weight:800;font-size:11px;">${preCands.length}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;overflow-y:auto;max-height:120px;padding-right:2px;">
            ${preCands.map(id => `<span class="vote-micro-pill" style="background:rgba(243, 156, 18, 0.08);color:#f39c12;border-color:rgba(243, 156, 18, 0.15);padding:2px 5px;font-size:9px;">N${id}</span>`).join('') || '<span style="color:var(--muted);font-size:10px;">None</span>'}
          </div>
        </div>
        
        <!-- Candidates Column -->
        <div style="border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column;">
          <div style="font-weight:700;font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:8px;display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:6px;">
            <span>2. Candidates</span>
            <span style="color:#9b59b6;font-weight:800;font-size:11px;">${cands.length}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;overflow-y:auto;max-height:120px;padding-right:2px;">
            ${cands.map(id => `<span class="vote-micro-pill" style="background:rgba(155, 89, 182, 0.08);color:#9b59b6;border-color:rgba(155, 89, 182, 0.15);padding:2px 5px;font-size:9px;">N${id}</span>`).join('') || '<span style="color:var(--muted);font-size:10px;">None</span>'}
          </div>
        </div>

        <!-- Leader Column -->
        <div style="border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column;">
          <div style="font-weight:700;font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:8px;display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:6px;">
            <span>3. Elected Leader</span>
            <span style="color:var(--good);font-weight:800;font-size:11px;">${finalLeader.length}</span>
          </div>
          <div style="display:flex;align-items:center;justify-content:center;flex-grow:1;min-height:40px;">
            ${finalLeader.map(id => `<span class="vote-micro-pill granted" style="font-weight:700;padding:4px 10px;font-size:11px;box-shadow:0 0 10px var(--good-dim);border-color:rgba(46,204,113,0.3);">Node ${id} ★</span>`).join('') || '<span class="vote-micro-pill denied" style="font-weight:700;padding:4px 10px;font-size:10px;">Split Vote ✗</span>'}
          </div>
        </div>
      </div>
    </div>
  `;

  document.getElementById('election-funnel').innerHTML = funnelHTML;
}

function fmt(x) { return x == null ? '-' : Number(x).toFixed(3); }

setInterval(fetchSnapshot, 500);
fetchSnapshot();

