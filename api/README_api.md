# Excel 相关性筛选 API（严格对齐 `API.md` 的 MVP）

这版是给论文实验用的“严格接口版”，目标是：

- 尽量按 `API.md` 的字段和状态机返回
- 使用你们已经训练好的 Stage1 + Stage2-v2 checkpoint
- 支持异步 Job 轮询
- 方便你们快速试验“retrieval + LLM”流程

## 已严格对齐的点

- `POST /api/v1/retrieval/jobs`
- `GET /api/v1/retrieval/jobs/{job_id}`
- `job_id / status / created_at / updated_at / poll_url / result / error`
- `status` 只使用：`queued / running / succeeded / failed`
- `excel_urls` 强制要求 `https://`
- 成功时返回：
  - `result.query`
  - `result.results[].excel_url`
  - `result.results[].sheets[].sheet_name`
  - `result.results[].sheets[].sheet_index`
  - `result.errors[]`
- `results` 允许为空数组
- 部分 URL 失败时仍返回 `succeeded`，并将失败明细写入 `result.errors`
- 创建任务时返回 `201 Created`，并带 `Location: <poll_url>` header
- `job_id` 不存在或过期时返回 `404 + JOB_NOT_FOUND`

## 文件

- `app.py`
- `retrieval_runtime.py`

## 推荐放置位置

```bash
/root/sheetagentresearch/sheetagent_paper/api/
```

## 依赖

```bash
conda activate agentsheet310
pip install fastapi uvicorn httpx openpyxl
```

## 环境变量

```bash
export REPO_ROOT=/root/sheetagentresearch/sheetagent_paper
export STAGE1_CKPT=/root/sheetagentresearch/sheetagent_paper/best_model/classifier.pt
export STAGE2_CKPT=/root/sheetagentresearch/sheetagent_paper/outputs/stage2_gtn_v2/stage2_gtn_v2_stable_lr15e5_ep50/best.pt
export BACKBONE_DIR=/root/sheetagentresearch/sheetagent_paper/best_model/backbone
export TOKENIZER_DIR=/root/sheetagentresearch/sheetagent_paper/best_model
export DATA_DIR=/root/sheetagentresearch/sheetagent_paper/data
export PUBLIC_BASE_URL=https://YOUR_HOST:8000
```

## 运行

```bash
cd /root/sheetagentresearch/sheetagent_paper/api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 测试

### 创建任务

```bash
curl -X POST "http://YOUR_HOST:8000/api/v1/retrieval/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "excel_urls": [
      "https://your-domain/path/file.xlsx"
    ],
    "query": "2024 年一季度各区域销售额与退货率在哪个表？"
  }'
```

### 轮询

```bash
curl "http://YOUR_HOST:8000/api/v1/retrieval/jobs/<job_id>"
```

## 注意

这是论文实验版，不是生产版：

- 使用内存 job store
- 未实现认证
- 未实现 SSRF 白名单
- 未实现取消任务
- 未实现文件大小上限配置

但主接口契约已经尽量按 `API.md` 对齐了。
