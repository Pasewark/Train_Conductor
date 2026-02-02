# Training Monitor

Drop-in monitoring for ML training runs. An LLM watches your metrics, reads your code, and tells you when something looks wrong — all delivered to your phone via Telegram.

```python
import training_monitor as tm

tm.init(
    experiment_name="grpo_v2",
    config={"lr": 1e-4, "model": "gemma-3-4b", "batch_size": 32},
    openai_model="gpt-5.2",
    analysis_interval_min=10.0,
    telegram_chat=True,
)

for step in range(1000):
    metrics = train_step()
    tm.log(metrics, step)

tm.close()
```

That's the entire integration. A background process collects GPU/CPU stats, persists everything to SQLite, and periodically asks an LLM to analyze your run. If something looks wrong, you get an alert on your phone. You can message back to ask follow-up questions, just like chatting with any LLM.

## Installation

```bash
pip install "git+https://github.com/USER/REPO.git@v0.1.0"
```

## What it does

When you call `tm.init()`, a monitor subprocess starts in the background. It:

1. **Receives metrics** over ZMQ from your training loop (non-blocking, won't slow training)
2. **Polls GPU/CPU/RAM** via `nvidia-smi` and `psutil`
3. **Snapshots your code** and generates a summary so the LLM understands what's being trained
4. **Auto-generates metric descriptions** by reading your code to figure out what each logged metric actually measures
5. **Persists everything** to SQLite (survives crashes and restarts)
6. **Runs periodic LLM analysis** where the model inspects your code with tool calls, reviews metrics and system state, and flags issues
7. **Sends analysis to Telegram** (or Pushover) — alerts for problems, optionally every analysis
8. **Accepts your replies** — message the bot on Telegram to ask questions about your run, with full conversational context

### Code-aware analysis

The monitor doesn't just look at numbers. Before each analysis, the LLM has access to read-only tools that let it explore your training code:

- `snapshot_manifest` — list all files in the snapshot
- `snapshot_search` — grep for patterns (literal or regex) across the codebase
- `snapshot_read` — read specific line ranges from any file
- `snapshot_read_many` — read multiple file slices in one call

This means the analysis can reference your actual loss function, learning rate schedule, data loading logic, and anything else in your code. Analyses cite specific files and line numbers.

### Automatic metric descriptions

On first run, the LLM reads through your code to find all `tm.log()` calls and generates natural-language descriptions of each metric (e.g., `"kl_penalty"` → `"KL divergence between the policy and reference model, scaled by beta_kl"`). These descriptions are included in every analysis prompt so the LLM always knows what it's looking at. Descriptions are cached per code snapshot, so they're only regenerated when your code changes. If new metric keys appear mid-run, descriptions are generated incrementally.

### Interactive Telegram chat

Enable `telegram_chat=True` and you can message the bot directly:

> **You:** How's the loss trend looking?
>
> **Bot:** Loss decreased from 0.81 to 0.54 over steps 0–80 (Δ=-0.27). The decline is steady with no spikes. Grad norms are stable around 1.0. Looking healthy.

> **You:** Should I lower the learning rate?
>
> **Bot:** Current lr is 1e-4 and loss is still decreasing steadily with no signs of plateauing. I'd keep it as-is for now and revisit if loss flattens over the next ~20 steps.

The bot has full context: your config, all metrics, system state, code summary, metric descriptions, and the entire conversation history (previous analyses + your messages). Follow-ups work naturally. The LLM can also use code inspection tools when answering your questions, so it can look up implementation details on the fly.

All chat exchanges are logged and included in subsequent periodic analyses, so the model stays aware of your concerns.

## Logging metrics

```python
# Log whatever keys you want — the LLM sees all of them
tm.log({"loss": 0.42, "lr": 1e-4, "grad_norm": 1.2}, step=100)

# Multiple calls per step are merged automatically
tm.log({"rollout_reward": 0.7, "kl_penalty": 0.02}, step=100)
tm.log({"train_loss": 0.35, "step_time": 120.5}, step=100)
# → The LLM sees one entry for step 100 with all four metrics

# Force an immediate analysis after this log
tm.log(metrics, step=100, force_analysis=True)
```

Multiple `tm.log()` calls for the same step are merged into a single entry in the analysis prompt. This is useful when different phases of your training loop (rollout, training, evaluation) each log their own metrics for the same step.

## Configuration

### `tm.init()` parameters

| Parameter | Default | Description |
|---|---|---|
| `experiment_name` | *required* | Name for this run (used as directory name) |
| `config` | `None` | Dict of hyperparameters — included in LLM analysis context |
| `training_command` | `None` | Command used to launch training (defaults to `sys.argv`) |
| `openai_model` | `"gpt-4o"` | Model for analysis (any OpenAI-compatible API) |
| `openai_base_url` | `None` | Custom API base URL (for local models, vLLM, etc.) |
| `reasoning_effort` | `None` | `"low"` / `"medium"` / `"high"` for reasoning models |
| `analysis_interval_min` | `5.0` | Minutes between automatic analyses |
| `gpus` | `None` (all) | List of GPU indices to monitor, e.g. `[0, 1]` |
| `notify_all_analyses` | `True` | Send every analysis to Telegram, not just alerts |
| `telegram_chat` | `False` | Enable interactive Telegram chat |
| `idle_timeout_min` | `None` | Pause analyses after N minutes with no new metrics |
| `start_monitor` | `"auto"` | `"auto"` (rank 0 only), `True`, or `False` |
| `root_dir` | `"ai_logger"` | Base output directory |
| `sig_figs` | `5` | Significant figures for metrics in LLM prompt |
| `zmq_port` | `5555` | Port for metrics streaming (auto-finds open port if taken) |
| `max_prompt_len` | `8000` | Maximum token budget for the analysis prompt |
| `max_conversation_history_tokens` | `5000` | Token budget for conversation history in prompts |
| `regenerate_code_analysis` | `False` | Force regeneration of code summary and metric descriptions |

### Output structure

```
ai_logger/grpo_v2/
├── config.json              # experiment config snapshot
├── training_monitor.db      # SQLite: all metrics, analyses, conversation history
├── training_monitor.log     # monitor process logs
├── analyses.jsonl           # analysis history (append-only)
├── code_summary.md          # LLM-generated summary of your training code
├── metric_descriptions.json # auto-generated metric name → description mapping
└── code/                    # snapshot of your training code (read-only for LLM tools)
```

## Notifications

### Telegram (recommended)

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

To set up: message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → copy the token. Then message your bot and visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat ID.

### Pushover

```bash
export PUSHOVER_USER_KEY="your-user-key"
export PUSHOVER_APP_TOKEN="your-app-token"
```

Telegram takes priority if both are configured.

## Standalone monitor

You can run the monitor as its own process and send metrics from your training script separately:

```bash
training-monitor \
    --experiment grpo_v2 \
    --openai-model gpt-4o \
    --analysis-interval-min 5 \
    --gpus 0,1,2,3 \
    --notify-all-analyses \
    --telegram-chat
```

Or equivalently with `python -m training_monitor ...`.

## Distributed training

When `start_monitor="auto"` (the default), only rank 0 spawns the monitor subprocess. This is detected via `LOCAL_RANK`, `SLURM_LOCALID`, or `RANK` environment variables. All ranks can call `tm.log()` — metrics are sent over ZMQ to the single monitor process.

## Recovery

If the monitor crashes and restarts, it recovers recent metrics, system data, and conversation history from SQLite automatically (default: last 60 minutes, configurable via `--recovery-max-age-min`). Code summaries and metric descriptions are cached per code snapshot hash, so they don't need to be regenerated.

## Code snapshots

The monitor snapshots your training code at startup using a manifest file (`snapshot_files.txt` by default). List files and directories to include, one per line:

```
# snapshot_files.txt
train.py
model/
config/
utils.py
```

The snapshot is stored in the run directory and used by the LLM's read-only tools throughout the run. A SHA-256 hash of the snapshot is used to cache the code summary and metric descriptions across experiments that share the same code.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for the LLM |
| `TELEGRAM_BOT_TOKEN` | For Telegram | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | For Telegram | Your chat ID |
| `PUSHOVER_USER_KEY` | For Pushover | Pushover user key |
| `PUSHOVER_APP_TOKEN` | For Pushover | Pushover app token |

## Requirements

- Python ≥ 3.10
- `pyzmq`, `requests`, `psutil`
- An OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.)
- `nvidia-smi` on the machine for GPU monitoring