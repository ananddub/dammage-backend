"""MinIO + MongoDB + MQTT init and helpers."""
from __future__ import annotations

import io
import json
from datetime import datetime

import paho.mqtt.client as mqtt
from minio import Minio
from minio.error import S3Error
from pymongo import MongoClient
from pymongo.database import Database

from .config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_PUBLIC_HOST,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MONGO_COLL,
    MONGO_DB,
    MONGO_URI,
    MQTT_HOST,
    MQTT_PASSWORD,
    MQTT_PORT,
    MQTT_RETAIN,
    MQTT_TOPIC,
    MQTT_USERNAME,
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


# ───────────────────────── MQTT ───────────────────────── #
def init_mqtt() -> mqtt.Client | None:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="dammage-backend")
        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_start()  # runs network IO in a background thread
        print(f"[mqtt] connecting to {MQTT_HOST}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f"[mqtt] disabled — {type(e).__name__}: {e}")
        return None


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def publish_report(client: mqtt.Client | None, doc: dict) -> bool:
    """Publish a report document to the shared MQTT_TOPIC.

    Every state change (upsert, status flip, soft-resolve) fires to the same
    topic. Subscribers receive the full fresh doc and filter client-side on
    coordinates / type / status. Retain is off by default — a single pinned
    message on a broadcast channel would be misleading.
    """
    if client is None:
        return False
    try:
        payload = json.dumps({**doc, "_id": str(doc.get("_id"))}, default=_json_default)
        info = client.publish(MQTT_TOPIC, payload, qos=0, retain=MQTT_RETAIN)
        return info.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception as e:
        print(f"[mqtt] publish failed: {e}")
        return False
