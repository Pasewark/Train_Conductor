(function () {
  "use strict";

  var POLL_MS = 2000;
  var EXPERIMENT_POLL_MS = 4000;
  var POLL_BACKOFF_FACTOR = 2;
  var POLL_BACKOFF_MAX_MS = 30000;
  var POLL_BACKOFF_JITTER_RATIO = 0.2;
  var POLL_DEGRADED_THRESHOLD = 2;
  var POLL_OFFLINE_THRESHOLD = 5;

  var isOpen = false;
  var pollTimer = null;
  var pollInFlight = false;
  var pollFailureCount = 0;
  var lastLogCount = 0;
  var userScrolledUp = false;
  var experimentPollTimer = null;
  var experimentPollInFlight = false;
  var experimentPollFailureCount = 0;
  var experimentLastValue = null;
  var experimentValueEl = null;
  var experimentBannerEl = null;
  var connectionStateEl = null;

  var CONNECTION_STATE_LABELS = {
    connected: "Connected",
    reconnecting: "Reconnecting",
    degraded: "Degraded",
    offline: "Offline"
  };

  function ensureFavicon() {
    var faviconHref = "/public/favicon.png";
    var link = document.querySelector("link[rel~='icon']");
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.appendChild(link);
    }
    if (link.getAttribute("href") !== faviconHref) {
      link.setAttribute("href", faviconHref);
    }
  }

  function computeBackoffDelay(baseDelayMs, failureCount) {
    if (failureCount <= 0) return baseDelayMs;
    var scaledDelay = baseDelayMs * Math.pow(POLL_BACKOFF_FACTOR, failureCount);
    var cappedDelay = Math.min(scaledDelay, POLL_BACKOFF_MAX_MS);
    var jitterRange = cappedDelay * POLL_BACKOFF_JITTER_RATIO;
    var jitter = (Math.random() * 2 - 1) * jitterRange;
    return Math.max(baseDelayMs, Math.round(cappedDelay + jitter));
  }

  function stateFromFailureCount(failureCount) {
    if (failureCount >= POLL_OFFLINE_THRESHOLD) return "offline";
    if (failureCount >= POLL_DEGRADED_THRESHOLD) return "degraded";
    if (failureCount > 0) return "reconnecting";
    return "connected";
  }

  function setConnectionState(state, detail) {
    if (!connectionStateEl) return;
    var safeState = CONNECTION_STATE_LABELS[state] ? state : "reconnecting";
    var label = CONNECTION_STATE_LABELS[safeState];
    connectionStateEl.className = "debug-connection debug-conn-" + safeState;
    connectionStateEl.textContent = label;
    connectionStateEl.title = detail ? String(detail) : label;
  }

  function updateExperimentFromState(state) {
    if (!experimentValueEl || !experimentBannerEl) return;
    if (!state || typeof state !== "object") return;
    var exp = state.experiment ? String(state.experiment) : "";
    if (!exp) {
      if (!experimentLastValue) {
        experimentValueEl.textContent = "waiting...";
        experimentBannerEl.title = "Waiting for experiment";
      }
      return;
    }
    if (exp === experimentLastValue) return;

    // Auto-reload when server detects an experiment transition
    if (state.experiment_changed
        && experimentLastValue
        && state.experiment_changed !== experimentLastValue) {
      console.log("[debug-panel] Experiment changed to " + state.experiment_changed + ", reloading");
      window.location.reload();
      return;
    }

    experimentLastValue = exp;
    experimentValueEl.textContent = exp;
    experimentBannerEl.title = exp;
  }

  function fetchExperiment() {
    return fetch("/api/debug")
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        var ct = resp.headers.get("content-type") || "";
        if (ct.indexOf("json") === -1) throw new Error("Not JSON: " + ct);
        return resp.json();
      })
      .then(function (data) {
        updateExperimentFromState(data.state);
      });
  }

  function scheduleExperimentPoll(delayMs) {
    if (experimentPollTimer) {
      clearTimeout(experimentPollTimer);
    }
    experimentPollTimer = setTimeout(function () {
      experimentPollTimer = null;
      runExperimentPoll();
    }, delayMs);
  }

  function runExperimentPoll() {
    if (experimentPollInFlight) {
      scheduleExperimentPoll(EXPERIMENT_POLL_MS);
      return;
    }
    experimentPollInFlight = true;
    fetchExperiment()
      .then(function () {
        experimentPollFailureCount = 0;
        experimentPollInFlight = false;
        scheduleExperimentPoll(EXPERIMENT_POLL_MS);
      })
      .catch(function (err) {
        experimentPollFailureCount += 1;
        experimentPollInFlight = false;
        var retryMs = computeBackoffDelay(EXPERIMENT_POLL_MS, experimentPollFailureCount);
        console.warn("[debug-panel] experiment fetch error:", err.message, "| retry in " + retryMs + "ms");
        scheduleExperimentPoll(retryMs);
      });
  }

  function startExperimentPolling() {
    if (experimentPollTimer || experimentPollInFlight) return;
    experimentPollFailureCount = 0;
    runExperimentPoll();
  }

  function initExperimentBanner() {
    var existing = document.getElementById("experiment-banner");
    if (existing) {
      experimentBannerEl = existing;
      experimentValueEl = existing.querySelector(".experiment-value");
      startExperimentPolling();
      return;
    }

    experimentBannerEl = document.createElement("div");
    experimentBannerEl.id = "experiment-banner";
    experimentBannerEl.innerHTML =
      '<span class="experiment-value">waiting...</span>';
    document.body.appendChild(experimentBannerEl);
    experimentValueEl = experimentBannerEl.querySelector(".experiment-value");
    startExperimentPolling();
  }

  function init() {
    // Guard against double-init
    if (document.getElementById("debug-panel-toggle")) return;

    console.log("[debug-panel] Initializing debug panel");
    ensureFavicon();

    // --- Toggle button ---
    var toggleBtn = document.createElement("button");
    toggleBtn.id = "debug-panel-toggle";
    toggleBtn.title = "Toggle debug panel";
    toggleBtn.textContent = "\u2699";
    document.body.appendChild(toggleBtn);

    // --- Panel ---
    var panel = document.createElement("div");
    panel.id = "debug-panel";
    panel.innerHTML =
      '<div class="debug-header">' +
        '<div class="debug-header-left">' +
          '<h3>Debug</h3>' +
          '<span id="debug-connection-state" class="debug-connection debug-conn-reconnecting">Reconnecting</span>' +
        '</div>' +
        '<button class="debug-close" title="Close">\u2715</button>' +
      '</div>' +
      '<div class="debug-section-title">State</div>' +
      '<div class="debug-state"><table><tbody id="debug-state-body"></tbody></table></div>' +
      '<div class="debug-section-title">Logs</div>' +
      '<div class="debug-logs" id="debug-logs-container"></div>';
    document.body.appendChild(panel);

    var closeBtn = panel.querySelector(".debug-close");
    var stateBody = document.getElementById("debug-state-body");
    var logsContainer = document.getElementById("debug-logs-container");
    connectionStateEl = document.getElementById("debug-connection-state");

    // --- Scroll tracking ---
    logsContainer.addEventListener("scroll", function () {
      var atBottom = logsContainer.scrollHeight - logsContainer.scrollTop - logsContainer.clientHeight < 40;
      userScrolledUp = !atBottom;
    });

    // --- Toggle logic ---
    function openPanel() {
      isOpen = true;
      panel.classList.add("open");
      toggleBtn.style.display = "none";
      setConnectionState("reconnecting", "Connecting to /api/debug");
      startPolling();
    }
    function closePanel() {
      isOpen = false;
      panel.classList.remove("open");
      toggleBtn.style.cssText = "display:flex !important";
      stopPolling();
    }

    toggleBtn.addEventListener("click", function () {
      if (isOpen) closePanel();
      else openPanel();
    });
    closeBtn.addEventListener("click", closePanel);

    // --- Formatting ---
    function formatTs(unix) {
      var d = new Date(unix * 1000);
      var h = String(d.getHours()).padStart(2, "0");
      var m = String(d.getMinutes()).padStart(2, "0");
      var s = String(d.getSeconds()).padStart(2, "0");
      var ms = String(d.getMilliseconds()).padStart(3, "0");
      return h + ":" + m + ":" + s + "." + ms;
    }

    function escapeHtml(str) {
      var div = document.createElement("div");
      div.textContent = str;
      return div.innerHTML;
    }

    // --- Render ---
    function formatValue(v) {
      if (v === null || v === undefined) return "\u2014";
      if (Array.isArray(v) || (typeof v === "object" && v !== null)) {
        try {
          return JSON.stringify(v);
        } catch (e) {
          return String(v);
        }
      }
      return String(v);
    }

    function renderState(state) {
      if (!state || typeof state !== "object") return;

      // Separate session keys from monitor.* keys
      var sessionKeys = [];
      var monitorKeys = [];
      var keys = Object.keys(state);
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        if (k.indexOf("monitor.") === 0) {
          monitorKeys.push(k);
        } else {
          sessionKeys.push(k);
        }
      }

      var html = "";

      // Session section
      if (sessionKeys.length > 0) {
        html += "<tr><td colspan=\"2\" class=\"debug-state-header\">Session</td></tr>";
        for (var i = 0; i < sessionKeys.length; i++) {
          var k = sessionKeys[i];
          var v = formatValue(state[k]);
          html += "<tr><td>" + escapeHtml(k) + "</td><td title=\"" + escapeHtml(v) + "\">" + escapeHtml(v) + "</td></tr>";
        }
      }

      // Monitor Init Args section
      if (monitorKeys.length > 0) {
        html += "<tr><td colspan=\"2\" class=\"debug-state-header\">Monitor Init Args</td></tr>";
        for (var i = 0; i < monitorKeys.length; i++) {
          var k = monitorKeys[i];
          var displayKey = k.substring(8); // Strip "monitor." prefix
          var v = formatValue(state[k]);
          html += "<tr><td>" + escapeHtml(displayKey) + "</td><td title=\"" + escapeHtml(v) + "\">" + escapeHtml(v) + "</td></tr>";
        }
      }

      stateBody.innerHTML = html;
    }

    function renderLogs(logs) {
      if (!logs || !logs.length) return;

      var startIdx = lastLogCount;
      if (startIdx >= logs.length) return;

      var fragment = document.createDocumentFragment();
      for (var i = startIdx; i < logs.length; i++) {
        var entry = logs[i];
        var level = entry.level || "info";
        var div = document.createElement("div");
        div.className = "debug-log-entry debug-level-" + level;
        div.innerHTML =
          '<span class="debug-log-ts">' + formatTs(entry.ts) + '</span>' +
          '<span class="debug-log-badge">' + escapeHtml(level) + '</span>' +
          '<span class="debug-log-msg">' + escapeHtml(entry.msg) + '</span>';
        fragment.appendChild(div);
      }
      logsContainer.appendChild(fragment);
      lastLogCount = logs.length;

      if (!userScrolledUp) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
      }
    }

    // --- Polling ---
    function fetchDebug() {
      return fetch("/api/debug")
        .then(function (resp) {
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          var ct = resp.headers.get("content-type") || "";
          if (ct.indexOf("json") === -1) throw new Error("Not JSON: " + ct);
          return resp.json();
        })
        .then(function (data) {
          renderState(data.state);
          renderLogs(data.logs);
          updateExperimentFromState(data.state);
        });
    }

    function schedulePoll(delayMs) {
      if (pollTimer) {
        clearTimeout(pollTimer);
      }
      pollTimer = setTimeout(function () {
        pollTimer = null;
        runPoll();
      }, delayMs);
    }

    function runPoll() {
      if (!isOpen) return;
      if (pollInFlight) {
        schedulePoll(POLL_MS);
        return;
      }
      if (pollFailureCount > 0) {
        setConnectionState("reconnecting", "Retrying /api/debug");
      }
      pollInFlight = true;
      fetchDebug()
        .then(function () {
          pollFailureCount = 0;
          pollInFlight = false;
          setConnectionState("connected", "Live updates active");
          if (isOpen) schedulePoll(POLL_MS);
        })
        .catch(function (err) {
          pollFailureCount += 1;
          pollInFlight = false;
          var retryMs = computeBackoffDelay(POLL_MS, pollFailureCount);
          var state = stateFromFailureCount(pollFailureCount);
          setConnectionState(
            state,
            "Consecutive failures: " + pollFailureCount + ". Retry in " + retryMs + "ms"
          );
          console.warn("[debug-panel] fetch error:", err.message, "| retry in " + retryMs + "ms");
          if (isOpen) schedulePoll(retryMs);
        });
    }

    function startPolling() {
      if (pollTimer || pollInFlight) return;
      pollFailureCount = 0;
      runPoll();
    }

    function stopPolling() {
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
      pollFailureCount = 0;
    }

    console.log("[debug-panel] Ready");
  }

  // Try to init now if body exists, otherwise wait for DOMContentLoaded,
  // and also retry after a delay in case React replaces the DOM.
  function tryInit() {
    if (document.body) {
      init();
      initExperimentBanner();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryInit);
  } else {
    tryInit();
  }
  // Retry after React has had time to mount, in case elements were removed
  setTimeout(tryInit, 2000);
  setTimeout(tryInit, 5000);
})();
