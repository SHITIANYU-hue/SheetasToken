from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyHttpUrl, Field, field_validator

from retrieval_runtime import (
    RetrievalRuntime,
    RetrievalRuntimeError,
    ErrorCode,
    RetrievalResult,
)

API_PREFIX = "/api/v1"
JOB_TTL_HOURS = 24


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class CreateJobBody(BaseModel):
    excel_urls: List[AnyHttpUrl]
    query: str

    @field_validator("excel_urls")
    @classmethod
    def validate_excel_urls(cls, value: List[AnyHttpUrl]) -> List[AnyHttpUrl]:
        if not value:
            raise ValueError("excel_urls 不能为空")
        for url in value:
            if url.scheme.lower() != "https":
                raise ValueError("excel_urls 中的每个地址都必须是 https://")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class SheetRef(BaseModel):
    sheet_name: str
    sheet_index: int


class ResultItem(BaseModel):
    excel_url: str
    sheets: List[SheetRef]


class ErrorItem(BaseModel):
    excel_url: str
    code: str
    message: str


class RetrievalResultBody(BaseModel):
    query: str
    results: List[ResultItem] = Field(default_factory=list)
    errors: List[ErrorItem] = Field(default_factory=list)


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    poll_url: str
    expires_at: str


class JobError(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    poll_url: str
    result: Optional[RetrievalResultBody] = None
    error: Optional[JobError] = None


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    poll_url: str
    expires_at: datetime
    request: CreateJobBody
    result: Optional[RetrievalResultBody] = None
    error: Optional[JobError] = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def build_poll_url(base_url: str, job_id: str) -> str:
    return f"{base_url.rstrip('/')}{API_PREFIX}/retrieval/jobs/{job_id}"


app = FastAPI(title="Excel Relevance Retrieval API", version="2.1-mvp-strict")
JOB_STORE: Dict[str, JobRecord] = {}
JOB_LOCK = asyncio.Lock()
RUNTIME: Optional[RetrievalRuntime] = None


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = "; ".join([err.get("msg", "参数校验失败") for err in exc.errors()]) or "参数校验失败"
    return error_response(400, "VALIDATION_ERROR", message)


@app.on_event("startup")
async def startup_event() -> None:
    global RUNTIME
    RUNTIME = RetrievalRuntime.from_defaults()
    RUNTIME.load()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def run_job(job_id: str) -> None:
    assert RUNTIME is not None
    async with JOB_LOCK:
        job = JOB_STORE[job_id]
        job.status = JobStatus.running
        job.updated_at = utc_now()

    result_body = RetrievalResultBody(query=job.request.query)

    try:
        for excel_url in map(str, job.request.excel_urls):
            try:
                retrieval_result: RetrievalResult = await RUNTIME.retrieve_from_url(
                    excel_url=excel_url,
                    query=job.request.query,
                )
                if retrieval_result.sheets:
                    result_body.results.append(
                        ResultItem(
                            excel_url=excel_url,
                            sheets=[SheetRef(**sheet) for sheet in retrieval_result.sheets],
                        )
                    )
            except RetrievalRuntimeError as exc:
                result_body.errors.append(
                    ErrorItem(
                        excel_url=excel_url,
                        code=exc.code.value,
                        message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                result_body.errors.append(
                    ErrorItem(
                        excel_url=excel_url,
                        code=ErrorCode.INTERNAL_ERROR.value,
                        message=str(exc),
                    )
                )

        async with JOB_LOCK:
            job = JOB_STORE[job_id]
            job.result = result_body
            job.status = JobStatus.succeeded
            job.updated_at = utc_now()

    except Exception as exc:  # noqa: BLE001
        async with JOB_LOCK:
            job = JOB_STORE[job_id]
            job.status = JobStatus.failed
            job.error = JobError(
                code=ErrorCode.INTERNAL_ERROR.value,
                message="任务执行失败" if not str(exc) else str(exc),
            )
            job.updated_at = utc_now()


@app.post(f"{API_PREFIX}/retrieval/jobs")
async def create_job(body: CreateJobBody) -> JSONResponse:
    base_url = "http://221.220.242.224:8000"
    now = utc_now()
    expires_at = now + timedelta(hours=JOB_TTL_HOURS)
    job_id = make_job_id()
    poll_url = build_poll_url(base_url, job_id)

    record = JobRecord(
        job_id=job_id,
        status=JobStatus.queued,
        created_at=now,
        updated_at=now,
        poll_url=poll_url,
        expires_at=expires_at,
        request=body,
    )

    async with JOB_LOCK:
        JOB_STORE[job_id] = record

    asyncio.create_task(run_job(job_id))

    payload = CreateJobResponse(
        job_id=job_id,
        status=JobStatus.queued,
        created_at=isoformat_z(now),
        poll_url=poll_url,
        expires_at=isoformat_z(expires_at),
    ).model_dump()

    return JSONResponse(
        status_code=201,
        content=payload,
        headers={"Location": poll_url},
    )


@app.get(f"{API_PREFIX}/retrieval/jobs/{{job_id}}")
async def get_job(job_id: str) -> JSONResponse:
    async with JOB_LOCK:
        record = JOB_STORE.get(job_id)

    if record is None or record.expires_at < utc_now():
        return error_response(404, ErrorCode.JOB_NOT_FOUND.value, "任务不存在或已过期")

    payload = JobResponse(
        job_id=record.job_id,
        status=record.status,
        created_at=isoformat_z(record.created_at),
        updated_at=isoformat_z(record.updated_at),
        poll_url=record.poll_url,
        result=record.result,
        error=record.error,
    ).model_dump()

    return JSONResponse(status_code=200, content=payload)
