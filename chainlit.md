# Training Monitor Web Chat

A real-time web mirror of the Training Monitor conversation with two-way messaging. Messages from the running monitor appear here as they are processed, and you can send messages back to the monitor from this interface.

## Key Features

- **Real-time sync** — messages appear as the monitor processes them
- **Two-way chat** — type messages that get routed to the running monitor
- **Model/reasoning selection** — use the profile dropdown to change model and reasoning effort
- **File attachments** — upload images, PDFs, or text files for context
- **Debug panel** — click the gear icon (bottom-right) to view internal state and logs

## Configuration

| Variable | Description | Default |
|---|---|---|
| `TM_DB_PATH` | Path to the training monitor SQLite database | resolved from `TM_ROOT_DIR`/`TM_PROJECT`/`TM_EXPERIMENT` |
| `TM_EXPERIMENT` | Experiment name (used for DB resolution and display) | — |
| `TM_WEB_POLL_INTERVAL` | Seconds between poll cycles | `1.0` |
| `TM_WEB_HISTORY_LIMIT` | Max messages loaded on initial connect | all |
| `TM_WEB_MODEL_OPTIONS` | Comma-separated list of models for the settings picker | auto-detected |
