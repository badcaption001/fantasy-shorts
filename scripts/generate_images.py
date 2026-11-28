#!/usr/bin/env python3
"""Generate scene images for ONE story with Cloudflare FLUX.2 klein 4B (free tier).

Modes:
  --mode dry-run   plan + cost estimate, no API call, no ledger write (default)
  --mode probe     ONE 256x256 test image
  --mode story     character reference portrait + up to N scene images (720x1280)
  --selftest       offline checks, no API call, no network

Safety design (all enforced in code):
  * Hard daily cap of 8,000 neurons (80% of the 10,000 free allowance), UTC day.
  * Usage ledger on a separate git branch ("state"). Reserve BEFORE spending.
  * Fail closed: if the ledger is missing, unreadable or invalid, nothing is generated.
  * The ledger is append-only: nothing ever resets or refunds below real attempts.
  * Costs are over-estimated on purpose (tiles rounded up, +10% safety factor).

Env: CF_ACCOUNT_ID, CF_API_TOKEN (GitHub Secrets), GITHUB_TOKEN (to push the ledger).
Secrets are never printed or written to any output file.
"""
import argparse
import base64
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import requests
from PIL import Image, ImageDraw, ImageStat

# ------------------------------------------------------------- constants ----
MODEL_ID = "@cf/black-forest-labs/flux-2-klein-4b"
API_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + MODEL_ID

# Cloudflare Workers AI pricing page, flux-2-klein-4b (checked 2026-11-27).
RATE_OUT_TILE = 26.05      # neurons per OUTPUT 512x512 tile
RATE_IN_TILE = 5.37        # neurons per INPUT  512x512 tile
TILE = 512
SAFETY = 1.10              # extra margin on every estimate
DAILY_ALLOWANCE = 10000.0  # Workers Free neurons per UTC day
HARD_CAP = 8000.0          # we never plan to use more than this per UTC day

SCENE_W, SCENE_H = 720, 1280
PORTRAIT_W = PORTRAIT_H = 512
REF_SEND = 448             # reference sent smaller than 512x512 (API limit wording)
PROBE_W = PROBE_H = 256
MAX_ATTEMPTS_PER_IMAGE = 3
BACKOFF = [2.0, 6.0, 18.0]
BLACKOUT_SECONDS = 300     # no live run within 5 min of the 00:00 UTC reset

LEDGER_FILE = "neuron_ledger.json"
LEDGER_SCHEMA = 1
STATE_BRANCH = "state"

FALLBACK_STYLE = ("cinematic fantasy illustration, painterly, dramatic lighting, "
                  "volumetric mist, rich teal and gold palette, no text, no watermark")
try:  # keep the same look as the story generator if it is available
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_stories import STYLE_SUFFIX as STORY_STYLE  # type: ignore
except Exception:  # noqa: BLE001
    STORY_STYLE = FALLBACK_STYLE
PROBE_PROMPT = ("a glowing blue crystal on a dark mossy rock, fantasy illustration, "
                "centered, soft light")

QUOTA_RE = re.compile(r"daily free allocation|used up your daily|10,?000 neurons|"
                      r"daily allocation|upgrade to .*paid", re.I)


# ------------------------------------------------------------ exceptions ----
class LedgerError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class ApiProblem(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind          # quota | rate | server | network | auth | client | empty
        self.retryable = kind in ("rate", "server", "network", "empty")


class RunStop(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------- cost math ----
def tiles(w, h):
    return math.ceil(w / TILE) * math.ceil(h / TILE)


def attempt_cost(w, h, ref_dims=None):
    """Conservative neurons for ONE attempt (tiles rounded up, plus safety)."""
    c = tiles(w, h) * RATE_OUT_TILE
    if ref_dims:
        c += tiles(*ref_dims) * RATE_IN_TILE
    return round(c * SAFETY, 2)


def likely_cost(w, h, ref_dims=None):
    """Area-based estimate (what Cloudflare most likely bills). Information only."""
    c = (w * h) / (TILE * TILE) * RATE_OUT_TILE
    if ref_dims:
        c += (ref_dims[0] * ref_dims[1]) / (TILE * TILE) * RATE_IN_TILE
    return round(c, 2)


def in_reset_blackout(now):
    secs = now.hour * 3600 + now.minute * 60 + now.second
    return secs >= 86400 - BLACKOUT_SECONDS or secs < BLACKOUT_SECONDS


def utc_day(now):
    return now.strftime("%Y-%m-%d")


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------- ledger ----
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def validate_ledger(led):
    if not isinstance(led, dict) or led.get("schema_version") != LEDGER_SCHEMA \
            or not isinstance(led.get("days"), dict):
        raise LedgerError("ledger has an unexpected structure")
    for day, d in led["days"].items():
        try:
            dt.date.fromisoformat(day)
        except ValueError:
            raise LedgerError(f"ledger has a bad day key: {day!r}")
        if not isinstance(d, dict) or not _num(d.get("external")) \
                or not isinstance(d.get("runs"), dict):
            raise LedgerError(f"ledger day {day} is malformed")
        for rid, r in d["runs"].items():
            if not isinstance(r, dict) or r.get("status") not in ("reserved", "settled") \
                    or not _num(r.get("reserved")) or not _num(r.get("spent")):
                raise LedgerError(f"ledger run {rid} on {day} is malformed")
            if r["status"] == "settled" and r["spent"] > r["reserved"] + 1e-6:
                raise LedgerError(f"ledger run {rid} spent more than it reserved")
    return led


def day_used(led, day):
    """Neurons counted for a day. An unsettled (crashed) run counts its full reservation."""
    d = led["days"].get(day)
    if not d:
        return 0.0
    used = d["external"]
    for r in d["runs"].values():
        used += r["spent"] if r["status"] == "settled" else r["reserved"]
    return round(used, 2)


def _guard(old, new, day, run, allow_external):
    """Append-only rule: an operation may only touch its own run (or add external usage)."""
    if old is None:
        return
    for k, v in old["days"].items():
        if k != day and new["days"].get(k) != v:
            raise LedgerError(f"refusing to alter another day's record ({k})")
    od, nd = old["days"].get(day), new["days"].get(day)
    if od is not None:
        if nd is None:
            raise LedgerError("refusing to delete a day record")
        if not allow_external and od["external"] != nd["external"]:
            raise LedgerError("refusing to change external usage here")
        if allow_external and nd["external"] < od["external"]:
            raise LedgerError("external usage can only increase")
        for rid, r in od["runs"].items():
            if rid != run and nd["runs"].get(rid) != r:
                raise LedgerError(f"refusing to alter another run's record ({rid})")


def _day(led, day):
    return led["days"].setdefault(day, {"external": 0.0, "runs": {}, "notes": []})


def op_init(day, assume_used):
    def fn(led):
        if led is not None:
            raise LedgerError("a ledger already exists; refusing to initialise it again")
        if not _num(assume_used):
            raise LedgerError("assume_used_today must be a non-negative number")
        return {"schema_version": LEDGER_SCHEMA, "days": {day: {
            "external": float(assume_used), "runs": {},
            "notes": [{"at": utc_now().isoformat(timespec="seconds"),
                       "neurons": float(assume_used), "note": "initial usage reported by owner"}]}}}
    return fn


def op_external(day, amount, note):
    def fn(led):
        if led is None:
            raise LedgerError("ledger is not initialised")
        if not _num(amount):
            raise LedgerError("external neurons must be a non-negative number")
        d = _day(led, day)
        d["external"] = round(d["external"] + float(amount), 2)
        d.setdefault("notes", []).append({"at": utc_now().isoformat(timespec="seconds"),
                                          "neurons": float(amount), "note": note})
        return led
    return fn


def op_reserve(day, run, amount, cap, mode):
    def fn(led):
        if led is None:
            raise LedgerError("ledger is not initialised")
        d = _day(led, day)
        if run in d["runs"]:
            raise LedgerError(f"run id {run} was already used")
        used = day_used(led, day)
        if used + amount > cap + 1e-9:
            raise BudgetExceeded(f"need {amount:.1f} but only {max(cap - used, 0):.1f} of the "
                                 f"{cap:.0f}-neuron daily cap is left (used {used:.1f})")
        d["runs"][run] = {"status": "reserved", "reserved": round(amount, 2), "spent": 0.0,
                          "mode": mode, "at": utc_now().isoformat(timespec="seconds")}
        return led
    return fn


def op_topup(day, run, amount, cap):
    def fn(led):
        d = led["days"].get(day) if led else None
        r = d["runs"].get(run) if d else None
        if not r or r["status"] != "reserved":
            raise LedgerError("cannot top up: run is not an open reservation")
        used = day_used(led, day)
        if used + amount > cap + 1e-9:
            raise BudgetExceeded(f"retry needs {amount:.1f} but only {max(cap - used, 0):.1f} "
                                 f"of the daily cap is left")
        r["reserved"] = round(r["reserved"] + amount, 2)
        return led
    return fn


def op_settle(day, run, spent):
    def fn(led):
        d = led["days"].get(day) if led else None
        r = d["runs"].get(run) if d else None
        if not r:
            raise LedgerError("cannot settle: run not found")
        if r["status"] == "settled":
            return led
        r["spent"] = round(spent, 2)
        r["reserved"] = round(max(r["reserved"], spent), 2)   # never settle below real attempts
        r["status"] = "settled"
        return led
    return fn


class StateStore:
    """The ledger lives in a git branch. Every change is pushed before we rely on it."""

    def __init__(self, directory, remote, branch=STATE_BRANCH, token=None):
        self.dir, self.remote, self.branch, self.token = directory, remote, branch, token
        self.fresh = False

    def _env(self):
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        if self.token and self.remote.startswith("https://github.com/"):
            basic = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
            env.update({"GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"})
        return env

    def _git(self, *args, cwd=None, check=True):
        p = subprocess.run(["git", *args], cwd=cwd, env=self._env(), capture_output=True,
                           text=True, timeout=120)
        if check and p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip().replace(self.token or "\0", "[REDACTED]")
            raise LedgerError(f"git {args[0]} failed: {err[:200]}")
        return p

    def open(self, init):
        if os.path.isdir(os.path.join(self.dir, ".git")):
            self._sync()
            return
        p = self._git("ls-remote", "--heads", self.remote, self.branch, check=False)
        if p.returncode != 0:
            raise LedgerError("cannot reach the repository's state branch (auth or network)")
        if p.stdout.strip():
            self._git("clone", "--depth", "1", "--branch", self.branch, self.remote, self.dir)
        elif init:
            os.makedirs(self.dir, exist_ok=True)
            self._git("init", "-b", self.branch, cwd=self.dir)
            self._git("remote", "add", "origin", self.remote, cwd=self.dir)
            self.fresh = True
        else:
            raise LedgerError("state branch not found. Run once with initialize_ledger=true.")

    def _sync(self):
        self._git("fetch", "origin", self.branch, cwd=self.dir)
        self._git("reset", "--hard", "FETCH_HEAD", cwd=self.dir)

    def load(self):
        path = os.path.join(self.dir, LEDGER_FILE)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                led = json.load(f)
        except (OSError, ValueError):
            raise LedgerError("ledger file is unreadable or corrupted")
        return validate_ledger(led)

    def commit(self, fn, message, day=None, run=None, allow_external=False):
        """Apply fn to the freshest ledger, push it, and only then return it."""
        for attempt in range(4):
            old = self.load()
            new = fn(copy.deepcopy(old) if old is not None else None)
            validate_ledger(new)
            _guard(old, new, day, run, allow_external)
            with open(os.path.join(self.dir, LEDGER_FILE), "w", encoding="utf-8") as f:
                json.dump(new, f, indent=2, sort_keys=True)
            self._git("add", LEDGER_FILE, cwd=self.dir)
            self._git("-c", "user.name=ledger-bot",
                      "-c", "user.email=ledger-bot@users.noreply.github.com",
                      "commit", "-m", message, cwd=self.dir)
            p = self._git("push", "origin", f"HEAD:refs/heads/{self.branch}", cwd=self.dir,
                          check=False)
            if p.returncode == 0:
                self.fresh = False
                return new
            if self.fresh:
                raise LedgerError("state branch appeared while initialising; run again")
            self._sync()               # someone else wrote first: re-read and re-check the budget
            time.sleep(0.2 * (attempt + 1))
        raise LedgerError("could not publish the ledger update after 4 tries")


# ----------------------------------------------------------------- plan ----
def load_story(path, index):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stories = data["stories"]
        return stories[index - 1]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise SystemExit(f"ERROR: cannot read story {index} from {path}. "
                         "Add the file from your generate-stories artifact (see README steps).")


def trim_prompt(p, limit=2000):
    p = re.sub(r"\s+", " ", p).strip()
    return p if len(p) <= limit else p[:limit].rsplit(" ", 1)[0]


def build_plan(story, mode, max_scenes):
    """Ordered list of images to make. The portrait (if any) always comes first."""
    if mode == "probe":
        return [dict(id="probe", kind="probe", w=PROBE_W, h=PROBE_H, prompt=PROBE_PROMPT,
                     use_ref=False)]
    items = []
    look = (story.get("character_look") or "").strip()
    if look:
        items.append(dict(id="character_ref", kind="portrait", w=PORTRAIT_W, h=PORTRAIT_H,
                          use_ref=False,
                          prompt=trim_prompt(f"Character reference portrait, upper body, facing "
                                             f"the camera, centered, plain dark background. "
                                             f"{look}. {STORY_STYLE}")))
    for sc in story["scenes"][:max_scenes]:
        base = sc.get("image_prompt_final") or sc["image_prompt"]
        use_ref = bool(look) and bool(sc.get("character_visible"))
        if use_ref:
            base = ("Keep the exact same character as in image 0: same face, hair and "
                    "clothing. " + base)
        items.append(dict(id=f"scene_{sc['scene']:02d}", kind="scene", w=SCENE_W, h=SCENE_H,
                          use_ref=use_ref, prompt=trim_prompt(base)))
    return items


def plan_costs(items):
    ref = (REF_SEND, REF_SEND)
    cons = sum(attempt_cost(i["w"], i["h"], ref if i["use_ref"] else None) for i in items)
    like = sum(likely_cost(i["w"], i["h"], ref if i["use_ref"] else None) for i in items)
    return round(cons, 2), round(like, 2)


def base_seed(story_id):
    return int(hashlib.sha256(story_id.encode()).hexdigest()[:8], 16) % 2147483647


# ------------------------------------------------------ Cloudflare client ----
def cf_request(session, account, token, item, seed, ref_bytes):
    fields = {"prompt": (None, item["prompt"]), "width": (None, str(item["w"])),
              "height": (None, str(item["h"])), "seed": (None, str(seed))}
    if item["use_ref"] and ref_bytes:
        fields["input_image_0"] = ("reference.jpg", ref_bytes, "image/jpeg")
    try:
        return session.post(API_URL.format(account=account),
                            headers={"Authorization": f"Bearer {token}"},
                            files=fields, timeout=(10, 180))
    except requests.RequestException as e:
        raise ApiProblem("network", f"network error: {type(e).__name__}")


def _clip(text, secret_values):
    t = str(text)[:200].replace("\n", " ")
    for s in secret_values:
        if s:
            t = t.replace(s, "[REDACTED]")
    return t


def parse_response(resp, secrets=()):
    """Return raw image bytes or raise ApiProblem."""
    st = resp.status_code
    body = resp.text if "image/" not in (resp.headers.get("content-type", "") or "") else ""
    if st == 200:
        if "image/" in (resp.headers.get("content-type", "") or ""):
            return resp.content
        try:
            data = resp.json()
        except ValueError:
            raise ApiProblem("empty", "200 response was not JSON")
        if isinstance(data, dict) and data.get("success") is False:
            msg = "; ".join(str(e.get("message", e)) for e in data.get("errors", [])) or "failed"
            kind = "quota" if QUOTA_RE.search(msg) else "server"
            raise ApiProblem(kind, _clip(msg, secrets))
        img = None
        if isinstance(data, dict):
            res = data.get("result")
            img = (res.get("image") if isinstance(res, dict) else None) or data.get("image")
        if not isinstance(img, str) or not img:
            raise ApiProblem("empty", "response had no image field")
        img = re.sub(r"^data:image/[a-z]+;base64,", "", img)
        try:
            return base64.b64decode(img, validate=False)
        except ValueError:
            raise ApiProblem("empty", "image field was not valid base64")
    msg = _clip(body, secrets)
    if st == 429:
        raise ApiProblem("quota" if QUOTA_RE.search(msg) else "rate", f"HTTP 429: {msg}")
    if st >= 500:
        raise ApiProblem("server", f"HTTP {st}: {msg}")
    if st in (401, 403):
        raise ApiProblem("auth", f"HTTP {st}: check the Cloudflare token permissions")
    if QUOTA_RE.search(msg):
        raise ApiProblem("quota", f"HTTP {st}: {msg}")
    raise ApiProblem("client", f"HTTP {st}: {msg}")


# ------------------------------------------------------------ validation ----
def ahash(im):
    g = im.convert("L").resize((8, 8), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / 64
    return sum(1 << i for i, v in enumerate(px) if v >= avg)


def hamming(a, b):
    return bin(a ^ b).count("1")


def encode_jpeg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92, subsampling=0)
    return buf.getvalue()


def validate_image(raw, want_w, want_h, seen):
    """Returns (rgb_image|None, jpeg_bytes|None, problems, notes, hash)."""
    notes = []
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:  # noqa: BLE001
        return None, None, ["not a decodable image"], notes, None
    im = im.convert("RGB")
    w, h = im.size
    probs = []
    if (w, h) != (want_w, want_h):
        if abs(w / h - want_w / want_h) > 0.01:
            probs.append(f"unexpected size {w}x{h}")
        else:
            im = im.resize((want_w, want_h), Image.LANCZOS)
            notes.append(f"resized from {w}x{h}")
    g = ImageStat.Stat(im.convert("L"))
    if g.stddev[0] < 8:
        probs.append("image is nearly flat or blank")
    if g.mean[0] < 6 or g.mean[0] > 249:
        probs.append("image is almost fully black or white")
    hsh = ahash(im)
    for name, other in seen.items():
        if hamming(hsh, other) <= 2:
            probs.append(f"near-duplicate of {name}")
    jpg = encode_jpeg(im)
    if len(jpg) < max(3000, int(want_w * want_h * 0.02)):
        probs.append("file is implausibly small")
    return im, jpg, probs, notes, hsh


def contact_sheet(files, path):
    thumbs = []
    for label, p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((180, 320))
        canvas = Image.new("RGB", (190, 345), (20, 20, 24))
        canvas.paste(im, (5, 20))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(240, 240, 240))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (190 * len(thumbs), 345), (20, 20, 24))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (190 * i, 0))
    sheet.save(path, "JPEG", quality=88)


# --------------------------------------------------------------- the run ----
class Run:
    def __init__(self, args, env, store, session, sleep, now_fn):
        self.args, self.env, self.store = args, env, store
        self.session, self.sleep, self.now_fn = session, sleep, now_fn
        self.cap = min(args.cap, HARD_CAP)
        self.run_key = f"{env.get('GITHUB_RUN_ID', 'local')}-{env.get('GITHUB_RUN_ATTEMPT', '1')}-" \
                       f"{uuid.uuid4().hex[:6]}"
        self.day = None
        self.spent = 0.0
        self.attempts = 0
        self.max_attempts = 0
        self.seen = {}
        self.secrets = [env.get("CF_API_TOKEN", ""), env.get("GITHUB_TOKEN", ""),
                        env.get("CF_ACCOUNT_ID", "")]

    def check_day(self):
        now = self.now_fn()
        if utc_day(now) != self.day:
            raise RunStop("day_changed", "UTC day changed during the run; stopping to keep the "
                                         "ledger accurate")
        if in_reset_blackout(now):
            raise RunStop("reset_window", "too close to the 00:00 UTC reset; stopping")

    def topup(self, amount):
        try:
            self.store.commit(op_topup(self.day, self.run_key, amount, self.cap),
                              f"ledger: top-up {self.run_key}", self.day, self.run_key)
        except BudgetExceeded as e:
            raise RunStop("budget_stop", str(e))

    def call_with_retries(self, item, ref_bytes, rec, bseed):
        ref_dims = (REF_SEND, REF_SEND) if (item["use_ref"] and ref_bytes) else None
        cost = attempt_cost(item["w"], item["h"], ref_dims)
        last = "no attempt"
        for k in range(MAX_ATTEMPTS_PER_IMAGE):
            if self.attempts >= self.max_attempts:
                rec["problems"].append("run attempt cap reached")
                return None
            self.check_day()
            if k > 0:
                self.topup(cost)               # retries are reserved BEFORE they run
            self.spent = round(self.spent + cost, 2)   # counted as spent the moment it starts
            self.attempts += 1
            rec["attempts"] += 1
            rec["neurons_conservative"] = round(rec["neurons_conservative"] + cost, 2)
            seed = (bseed + k * 7919) % 2147483647
            rec["seed"] = seed
            try:
                resp = cf_request(self.session, self.env["CF_ACCOUNT_ID"], self.env["CF_API_TOKEN"],
                                  item, seed, ref_bytes)
                raw = parse_response(resp, self.secrets)
            except ApiProblem as e:
                last = f"{e.kind}: {e}"
                rec["problems"].append(last)
                if e.kind == "quota":
                    raise RunStop("quota_stop", "Cloudflare reports the daily free allowance is used")
                if not e.retryable:
                    raise RunStop("failed", last)
                if k < MAX_ATTEMPTS_PER_IMAGE - 1 and self.attempts < self.max_attempts:
                    self.sleep(BACKOFF[min(k, len(BACKOFF) - 1)] * random.uniform(0.5, 1.5))
                continue
            im, jpg, probs, notes, hsh = validate_image(raw, item["w"], item["h"], self.seen)
            rec["notes"] += notes
            if probs:
                last = "invalid image: " + "; ".join(probs)
                rec["problems"].append(last)
                continue                        # a new seed on the next attempt
            self.seen[item["id"]] = hsh
            return im, jpg
        return None



def run_pipeline(args, env, store, session, sleep=time.sleep, now_fn=utc_now):
    """Returns (exit_code, manifest). Exit: 0 ok, 1 failed, 2 config, 3 stopped by budget/time."""
    now = now_fn()

    # Ledger initialization must not depend on stories.json.
    if args.mode == "init-ledger-only":
        story = None
        items = []
        cons, like = 0.0, 0.0
    else:
        story = None if args.mode == "probe" else load_story(
            args.stories, args.story_index
        )
        items = build_plan(story or {}, args.mode, args.max_scenes)
        cons, like = plan_costs(items)

    cap = min(args.cap, HARD_CAP)

    manifest = {
        "model": MODEL_ID,
        "mode": args.mode,
        "story_id": (story or {}).get("id", "ledger-init" if args.mode == "init-ledger-only" else "probe"),
        "title": (story or {}).get("title", "Ledger initialization" if args.mode == "init-ledger-only" else "probe"),
        "utc_day": utc_day(now),
        "cap_neurons": cap,
        "plan_images": len(items),
        "plan_cost_conservative": cons,
        "plan_cost_likely": like,
        "dimensions": {
            "scene": [SCENE_W, SCENE_H],
            "portrait": [PORTRAIT_W, PORTRAIT_H]
        },
        "items": [],
        "warnings": [],
        "status": "dry_run"
    }

    print(
        f"Plan: {len(items)} images | reserve {cons} neurons "
        f"(likely real cost ~{like}) | cap {cap:.0f}/day"
    )

    for item in items:
        print(
            f"  - {item['id']}: {item['w']}x{item['h']} "
            f"ref={item['use_ref']}"
        )

    if story and not (story.get("character_look") or "").strip():
        manifest["warnings"].append(
            "story has no character_look: no reference portrait"
        )

    # SAFE LEDGER INITIALIZATION ONLY
    # No stories, image planning, Cloudflare API calls or image generation.
    if args.mode == "init-ledger-only":
        if not args.init_ledger:
            print(
                "ERROR: Select initialize_ledger=true "
                "for init-ledger-only."
            )
            return 2, manifest

        try:
            st = StateStore(
                args.state_dir,
                args.state_remote,
                token=env.get("GITHUB_TOKEN")
            )

            st.open(init=True)

            st.commit(
                op_init(utc_day(now), args.assume_used),
                "ledger: initialise",
                utc_day(now),
                None
            )

        except LedgerError as e:
            manifest["status"] = "ledger_error"
            print(f"STOP: {e}")
            return 3, manifest

        manifest["status"] = "ledger_initialized"
        print(f"Ledger initialized on branch '{STATE_BRANCH}'.")
        print("No images generated.")
        print("No Cloudflare API calls made.")
        return 0, manifest

    # DRY RUN: NO API CALLS AND NO LEDGER WRITES
    if args.mode == "dry-run":
        print("DRY RUN: no API calls and no ledger writes.")
        return 0, manifest

    # LIVE MODES: SAFETY CHECKS BEFORE API CALLS
    if not env.get("CF_ACCOUNT_ID") or not env.get("CF_API_TOKEN"):
        print(
            "ERROR: CF_ACCOUNT_ID and CF_API_TOKEN "
            "must be set as GitHub Secrets."
        )
        return 2, manifest

    if args.cap > HARD_CAP:
        print(
            f"ERROR: --cap cannot exceed the "
            f"{HARD_CAP:.0f}-neuron hard cap."
        )
        return 2, manifest

    if in_reset_blackout(now):
        manifest["status"] = "reset_window"
        print(
            "STOP: within 5 minutes of the 00:00 UTC quota reset. "
            "Try again later."
        )
        return 3, manifest

    run = Run(args, env, store, session, sleep, now_fn)
    run.day = utc_day(now)
    manifest["run_key"] = run.run_key
    run.max_attempts = 2 * len(items)

    try:
        store.open(init=args.init_ledger)

        if args.init_ledger:
            store.commit(
                op_init(run.day, args.assume_used),
                "ledger: initialise",
                run.day,
                None
            )
        elif store.load() is None:
            raise LedgerError(
                "ledger file not found on the state branch; initialise it first"
            )

        if args.add_external > 0:
            store.commit(
                op_external(
                    run.day,
                    args.add_external,
                    "owner-reported external usage"
                ),
                "ledger: external usage",
                run.day,
                None,
                allow_external=True
            )

        led = store.commit(
            op_reserve(run.day, run.run_key, cons, cap, args.mode),
            f"ledger: reserve {run.run_key}",
            run.day,
            run.run_key
        )

    except BudgetExceeded as e:
        manifest["status"] = "budget_stop_before_start"
        print(f"STOP (nothing was generated or spent): {e}")
        return 3, manifest

    except LedgerError as e:
        manifest["status"] = "ledger_error"
        print(f"STOP (fail closed, nothing was generated): {e}")
        return 3, manifest

    manifest["ledger_used_after_reserve"] = day_used(led, run.day)

    print(
        f"Reserved {cons} neurons for run {run.run_key}; "
        f"ledger now {manifest['ledger_used_after_reserve']}/{cap:.0f} today"
    )

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    bseed = base_seed(manifest["story_id"])
    ref_bytes, saved, code, consecutive_failures = None, [], 0, 0

    try:
        for item in items:
            rec = {
                "id": item["id"],
                "kind": item["kind"],
                "width": item["w"],
                "height": item["h"],
                "used_reference": item["use_ref"],
                "attempts": 0,
                "status": "failed",
                "neurons_conservative": 0.0,
                "problems": [],
                "notes": [],
                "prompt": item["prompt"],
                "file": None
            }

            manifest["items"].append(rec)

            if item["use_ref"] and not ref_bytes:
                rec["problems"].append(
                    "reference portrait missing; scene skipped"
                )
                rec["status"] = "skipped"
                continue

            res = run.call_with_retries(item, ref_bytes, rec, bseed)

            if res is None:
                print(
                    f"FAILED {item['id']}: "
                    f"{rec['problems'][-1] if rec['problems'] else 'unknown'}"
                )
                consecutive_failures += 1
                code = 1

                if item["kind"] == "portrait":
                    raise RunStop(
                        "failed",
                        "reference portrait failed; stopping so no neurons "
                        "are spent on scenes without a consistent character"
                    )

                if consecutive_failures >= 2:
                    raise RunStop(
                        "failed",
                        "two images failed in a row; stopping (server looks unhealthy)"
                    )

                continue

            consecutive_failures = 0
            im, jpg = res

            path = os.path.join(out_dir, f"{item['id']}.jpg")

            with open(path, "wb") as f:
                f.write(jpg)

            rec.update(
                status="ok",
                file=os.path.basename(path),
                bytes=len(jpg),
                sha256=hashlib.sha256(jpg).hexdigest()
            )

            saved.append((item["id"], path))

            if item["kind"] == "portrait":
                small = im.resize((REF_SEND, REF_SEND), Image.LANCZOS)
                ref_bytes = encode_jpeg(small)

            print(
                f"ok {item['id']} ({rec['attempts']} attempt(s), "
                f"{rec['neurons_conservative']} n)"
            )

        manifest["status"] = "complete" if code == 0 else "partial_failures"

    except RunStop as e:
        manifest["status"] = e.status
        manifest["warnings"].append(str(e))
        print(f"STOP: {e}")

        code = 3 if e.status in (
            "budget_stop",
            "quota_stop",
            "day_changed",
            "reset_window"
        ) else 1

    finally:
        try:
            end = store.commit(
                op_settle(run.day, run.run_key, run.spent),
                f"ledger: settle {run.run_key}",
                run.day,
                run.run_key
            )
            manifest["ledger_used_final"] = day_used(end, run.day)

        except Exception as e:
            manifest["warnings"].append(
                f"settle failed ({type(e).__name__}); "
                "full reservation stays counted, which is safe"
            )
            print(
                "WARNING: could not settle the ledger; "
                "reservation stays counted (safe)."
            )

        manifest["neurons_spent_conservative"] = run.spent
        manifest["attempts_total"] = run.attempts

        with open(
            os.path.join(out_dir, "images_manifest.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        if saved:
            contact_sheet(
                saved,
                os.path.join(out_dir, "contact_sheet.jpg")
            )

    print(
        f"Done: status={manifest['status']} "
        f"attempts={run.attempts} "
        f"spent~{run.spent} neurons (conservative)"
    )

    return code, manifest



# -------------------------------------------------------------- selftest ----
def _synthetic_png(seed, w=720, h=1280):
    rnd = random.Random(seed)
    im = Image.new("RGB", (w, h), (rnd.randint(0, 80),) * 3)
    d = ImageDraw.Draw(im)
    for _ in range(60):
        x, y = rnd.randint(0, w), rnd.randint(0, h)
        d.ellipse([x, y, x + rnd.randint(20, 200), y + rnd.randint(20, 200)],
                  fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def selftest():
    assert attempt_cost(256, 256) == round(26.05 * 1.10, 2)
    assert tiles(720, 1280) == 6 and tiles(512, 512) == 1 and tiles(448, 448) == 1
    assert attempt_cost(720, 1280, (448, 448)) == round((6 * 26.05 + 5.37) * 1.10, 2)
    assert likely_cost(720, 1280) < attempt_cost(720, 1280)
    assert in_reset_blackout(dt.datetime(2026, 1, 1, 23, 58, tzinfo=dt.timezone.utc))
    assert in_reset_blackout(dt.datetime(2026, 1, 2, 0, 2, tzinfo=dt.timezone.utc))
    assert not in_reset_blackout(dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc))
    led = op_init("2026-01-01", 100)(None)
    assert day_used(led, "2026-01-01") == 100
    led = op_reserve("2026-01-01", "r1", 500, 8000, "story")(led)
    assert day_used(led, "2026-01-01") == 600
    try:
        op_reserve("2026-01-01", "r2", 7500, 8000, "story")(copy.deepcopy(led))
        raise AssertionError("cap not enforced")
    except BudgetExceeded:
        pass
    try:
        op_init("2026-01-01", 0)(led)
        raise AssertionError("re-init allowed")
    except LedgerError:
        pass
    led = op_settle("2026-01-01", "r1", 300)(led)
    assert day_used(led, "2026-01-01") == 400
    assert day_used(led, "2026-01-02") == 0          # a new day is a new key, not a reset
    assert "2026-01-01" in led["days"]
    d = tempfile.mkdtemp()
    try:
        for good in (True, False):
            im, jpg, probs, _, _ = validate_image(_synthetic_png(1) if good else b"nope", 720, 1280, {})
            assert (probs == []) == good
        blank = io.BytesIO()
        Image.new("RGB", (720, 1280), (10, 10, 10)).save(blank, "PNG")
        assert validate_image(blank.getvalue(), 720, 1280, {})[2]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("selftest OK (no API call, no network)")
    return 0


# ------------------------------------------------------------------ main ----
def parse_args(argv):
    ap = argparse.ArgumentParser(description="FLUX.2 klein image generation (free tier)")
    ap.add_argument(
    "--mode",
    choices=["dry-run", "init-ledger-only", "probe", "story"],
    default="dry-run")
    ap.add_argument("--stories", default="data/stories.json")
    ap.add_argument("--story-index", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=6)
    ap.add_argument("--out", default="out/images")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--state-remote", default=None)
    ap.add_argument("--cap", type=float, default=HARD_CAP)
    ap.add_argument("--init-ledger", action="store_true")
    ap.add_argument("--assume-used", type=float, default=0.0)
    ap.add_argument("--add-external", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if not 1 <= a.story_index <= 3:
        ap.error("--story-index must be 1, 2 or 3")
    if not 1 <= a.max_scenes <= 8:
        ap.error("--max-scenes must be between 1 and 8")
    if a.assume_used < 0 or a.add_external < 0 or a.cap <= 0:
        ap.error("neuron numbers must be positive")
    return a


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.selftest:
        return selftest()
    env = dict(os.environ)
    if not args.state_remote:
        repo = env.get("GITHUB_REPOSITORY", "")
        args.state_remote = f"https://github.com/{repo}.git" if repo else ""
    if args.mode != "dry-run" and not args.state_remote:
        print("ERROR: no state remote (set GITHUB_REPOSITORY or --state-remote).")
        return 2
    store = StateStore(args.state_dir, args.state_remote, token=env.get("GITHUB_TOKEN"))
    code, manifest = run_pipeline(args, env, store, requests.Session())
    summ = env.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(f"### Image run: {manifest['status']}\n"
                    f"- mode `{args.mode}`, story `{manifest.get('story_id')}`\n"
                    f"- plan {manifest['plan_images']} images, reserved "
                    f"~{manifest['plan_cost_conservative']} neurons (conservative)\n"
                    f"- spent ~{manifest.get('neurons_spent_conservative', 0)} neurons, "
                    f"ledger today {manifest.get('ledger_used_final', manifest.get('ledger_used_today', 'n/a'))}"
                    f"/{manifest['cap_neurons']:.0f}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
