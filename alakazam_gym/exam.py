"""ExamClient — certify a policy in the frozen Webots oracle via the Forge API.

The exam is the sole scoreboard: dream/gym scores are training telemetry.
Today the exam accepts the 9-float controller family (genome9, or the compact
genome6 it expands from). ONNX policy submission is a planned extension.

Usage:
    from alakazam_gym import ExamClient
    c = ExamClient("http://localhost:8790", key=os.environ["FORGE_KEY"])
    job = c.submit_exam_only("my-cert-001", genome6=[...])   # returns spawn info
    result = c.wait(job["job_id"])                            # poll to done
    print(result["oracle"]["verdict"])
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ExamError(RuntimeError):
    pass


class ExamClient:
    def __init__(self, base_url: str, key: str, timeout_s: float = 30.0):
        self.base = base_url.rstrip("/")
        self.key = key
        self.timeout_s = timeout_s

    def _req(self, method: str, path: str, body=None):
        r = urllib.request.Request(self.base + path, method=method)
        r.add_header("Authorization", f"Bearer {self.key}")
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            r.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(r, data, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:
                detail = ""
            raise ExamError(f"{method} {path} -> {e.code}: {detail}") from e

    # -- jobs ---------------------------------------------------------------
    def submit_exam_only(self, job_id: str, genome6=None, genome9=None,
                         episodes: int = 20, note: str | None = None):
        """Exam-only certification (no training): gens=0 + your genome.
        Provide genome6 (6 floats) or genome9 (9 floats in the controller
        family layout [a,b,b,a,bias,bias,gt,gg,gth] — reduced to genome6)."""
        if genome9 is not None and genome6 is None:
            g = list(map(float, genome9))
            if len(g) != 9:
                raise ValueError("genome9 must be 9 floats")
            if g[3] != g[0] or g[2] != g[1] or g[5] != g[4]:
                raise ValueError(
                    "genome9 outside the controller family layout "
                    "[a,b,b,a,bias,bias,gt,gg,gth] — cannot reduce to genome6")
            genome6 = [g[0], g[1], g[4], g[6], g[7], g[8]]
        if genome6 is None:
            raise ValueError("provide genome6 or genome9")
        parent = {"genome6": [float(v) for v in genome6]}
        if note:
            parent["note"] = note
        spec = {"job_id": job_id,
                "train": {"pop": 0, "gens": 0, "T": 0, "seed": 0,
                          "parent": parent},
                "exam": {"episodes": int(episodes)}}
        return self._req("POST", "/jobs", spec)

    def submit_training(self, job_id: str, pop: int, gens: int, T: int,
                        seed: int, parent: dict | None = None,
                        episodes: int = 20):
        """Server-side dream training + exam (see the Forge API reference)."""
        train = {"pop": pop, "gens": gens, "T": T, "seed": seed}
        if parent is not None:
            train["parent"] = parent
        return self._req("POST", "/jobs", {
            "job_id": job_id, "train": train, "exam": {"episodes": episodes}})

    def get(self, job_id: str):
        return self._req("GET", f"/jobs/{job_id}")

    def wait(self, job_id: str, poll_s: float = 15.0, timeout_s: float = 4 * 3600):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            j = self.get(job_id)
            if j.get("status") in ("done", "failed"):
                return j
            time.sleep(poll_s)
        raise ExamError(f"timeout waiting for job {job_id}")
