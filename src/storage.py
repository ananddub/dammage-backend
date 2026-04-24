"""MinIO + MongoDB init and helpers."""
from __future__ import annotations

import io
from datetime import datetime, timezone

from minio import Minio
from minio.error import S3Error
from pymongo import MongoClient
from pymongo.database import Database

from .config import (
    ADMIN_SEED_EMAILS,
    ADMINS_COLL,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_PUBLIC_HOST,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MONGO_COLL,
    MONGO_DB,
    MONGO_URI,
)


def init_minio() -> Minio | None:
    try:
        mc = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
        if not mc.bucket_exists(MINIO_BUCKET):
            mc.make_bucket(MINIO_BUCKET)
        policy = (
            '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},'
            '"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::' + MINIO_BUCKET + '/*"]}]}'
        )
        try:
            mc.set_bucket_policy(MINIO_BUCKET, policy)
        except S3Error:
            pass
        return mc
    except Exception as e:
        print(f"[minio] disabled — {type(e).__name__}: {e}")
        return None


def init_mongo() -> Database | None:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client[MONGO_DB]
        db[MONGO_COLL].create_index([("location", "2dsphere")])
        db[MONGO_COLL].create_index([("time", -1)])
        # Seed admins from env once. `_id` is the email so upsert is idempotent.
        if ADMIN_SEED_EMAILS:
            now = datetime.now(timezone.utc)
            for email in ADMIN_SEED_EMAILS:
                db[ADMINS_COLL].update_one(
                    {"_id": email},
                    {"$setOnInsert": {"added_at": now, "added_by": "seed", "name": None}},
                    upsert=True,
                )
        return db
    except Exception as e:
        print(f"[mongo] disabled — {type(e).__name__}: {e}")
        return None


def upload(mc: Minio | None, key: str, data: bytes, content_type: str) -> str | None:
    if mc is None:
        return None
    try:
        mc.put_object(MINIO_BUCKET, key, io.BytesIO(data), length=len(data), content_type=content_type)
        return f"{MINIO_PUBLIC_HOST}/{MINIO_BUCKET}/{key}"
    except Exception as e:
        print(f"[minio] upload failed: {e}")
        return None
