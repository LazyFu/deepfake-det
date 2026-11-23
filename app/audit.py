"""审计日志：JSONL，每条记录带前一条的哈希，形成哈希链。"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

ROOT = Path("data/app")
AUDIT_PATH = ROOT / "audit.jsonl"
GENESIS = "0" * 64


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class AuditLog:
    def __init__(self, path: Path = AUDIT_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.last = GENESIS
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.last = json.loads(line)["hash"]
                    except Exception:
                        continue

    def append(self, record: dict) -> dict:
        with self.lock:
            rec = dict(record)
            rec["prev_hash"] = self.last
            rec["hash"] = hashlib.sha256(
                (self.last + canonical(rec)).encode("utf-8")
            ).hexdigest()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.last = rec["hash"]
            return rec

    def verify(self) -> dict:
        """检查整条链是否完整，返回统计。"""
        if not self.path.exists():
            return {"ok": True, "n": 0}
        prev = GENESIS
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                body = {k: v for k, v in rec.items() if k != "hash"}
                expect = hashlib.sha256((prev + canonical(body)).encode()).hexdigest()
                if rec.get("prev_hash") != prev or rec.get("hash") != expect:
                    return {"ok": False, "n": n, "broken_at": n}
                prev = rec["hash"]
                n += 1
        return {"ok": True, "n": n}
