# Excel Relevance Retrieval API

This API runs retrieval with trained Stage 1 and Stage 2-v2 checkpoints and
supports asynchronous jobs for spreadsheet retrieval experiments.

## Endpoints and response fields

- `POST /api/v1/retrieval/jobs` creates a job.
- `GET /api/v1/retrieval/jobs/{job_id}` polls its status.
- Job responses include `job_id`, `status`, `created_at`, `updated_at`,
  `poll_url`, `result`, and `error`.
- Status is one of `queued`, `running`, `succeeded`, or `failed`.
- Every URL in `excel_urls` must use `https://`.
- Successful results contain `result.query`, `result.results[].excel_url`,
  `result.results[].sheets[].sheet_name`,
  `result.results[].sheets[].sheet_index`, and `result.errors[]`.
- `results` may be empty.
- If some URLs fail, the job can still finish as `succeeded`; individual
  failures are recorded in `result.errors`.
- Job creation returns `201 Created` with a `Location: <poll_url>` header.
- Unknown or expired job IDs return `404` with `JOB_NOT_FOUND`.

## Files

- `app.py`: endpoints and in-memory job management.
- `retrieval_runtime.py`: spreadsheet parsing and checkpoint inference.

## Dependencies

Install the repository requirements and the API dependencies:

```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx openpyxl
```

## Configuration and checkpoints

Set the repository directory and the public URL used in polling links:

```bash
export REPO_ROOT=/path/to/repository
export PUBLIC_BASE_URL=https://YOUR_HOST:8000
```

The default runtime resolves these files beneath `REPO_ROOT`:

| Input | Relative path |
|---|---|
| Stage 1 checkpoint | `best_model/classifier.pt` |
| Stage 2 checkpoint | `outputs/stage2_gtn_v2/stage2_gtn_v2_stable_lr15e5_ep50/best.pt` |
| Backbone | `best_model/backbone` |
| Tokenizer | `best_model` |
| Data | `data` |

For a different checkpoint layout, configure `RetrievalRuntime` in the API
startup code. Checkpoints must be supplied separately.

## Run

```bash
cd /path/to/repository/api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Example requests

### Create a job

```bash
curl -X POST "http://YOUR_HOST:8000/api/v1/retrieval/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "excel_urls": [
      "https://your-domain/path/file.xlsx"
    ],
    "query": "Which sheets contain regional sales and return rates for Q1 2024?"
  }'
```

### Poll the job

```bash
curl "http://YOUR_HOST:8000/api/v1/retrieval/jobs/<job_id>"
```

## Current limitations

- The job store is held in memory.
- Authentication is not implemented.
- An SSRF allowlist is not implemented.
- Job cancellation is not implemented.
- A configurable file-size limit is not implemented.
