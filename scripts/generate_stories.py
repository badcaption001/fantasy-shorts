#!/usr/bin/env python3
"""Generate 3 unique fantasy short-video stories with ONE Gemini request.

Usage:
  python scripts/generate_stories.py                 # generate -> out/stories.json
  python scripts/generate_stories.py --selftest      # offline checks, no API call
  python scripts/generate_stories.py --verify out/stories.json

Needs env var GEMINI_API_KEY (GitHub Secret). The key is never printed or saved.
Optional env: GEMINI_MODEL (force one model), STORY_SALT (change the seeds).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OUT_DIR = "out"
OUT_FILE = os.path.join(OUT_DIR, "stories.json")
ERR_FILE = os.path.join(OUT_DIR, "stories_error.json")
HISTORY_FILE = os.path.join("data", "story_history.json")  # optional, read-only

NUM_STORIES = 3
MAX_ATTEMPTS = 5            # hard cap on HTTP generation attempts per run
MAX_OUTPUT_TOKENS = 12000   # headroom in case the model spends tokens thinking
WORDS_PER_SEC = 2.45        # dramatic narration pace
SCENE_PAUSE_SEC = 0.3
DURATION_MIN, DURATION_MAX = 30.0, 60.0
SCENES_MIN, SCENES_MAX = 5, 9
SIMILARITY_ERROR, SIMILARITY_WARN = 0.35, 0.25
SCHEMA_VERSION = 1

# Cheapest first. The script keeps only the ones your key can actually use.
MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

# Added in code to every image prompt (costs no LLM tokens, keeps one look).
STYLE_SUFFIX = ("cinematic fantasy illustration, painterly, dramatic lighting, "
                "volumetric mist, rich teal and gold palette, vertical 9:16 "
                "composition, no text, no watermark")

BANNED = re.compile(
    r"\b(harry potter|hogwarts|gandalf|frodo|mordor|hobbit|game of thrones|"
    r"pok[eé]mon|disney|marvel|zelda|skyrim|narnia|star wars|pixar|"
    r"dungeons (?:&|and) dragons)\b", re.I)

# ---------------------------------------------------------------- seeds ----
SETTINGS = [
    "a drowned library under a frozen sea", "a caravan crossing a desert of glass",
    "a clockwork forest where the trees tick", "a floating market above storm clouds",
    "a village that only wakes at night", "a lighthouse at the edge of the map",
    "a labyrinth inside a dragon's skeleton", "a mountain monastery of silent mages",
    "a city built on a giant sleeping beast", "a moonlit orchard of glowing fruit",
    "a volcano forge of the last dwarves", "an island that drifts between worlds",
    "a mirror realm behind a cracked well", "a ruined sky-kingdom of falling bridges",
]
PROTAGONISTS = [
    "a young cartographer", "a retired dragon hunter", "a thief with a borrowed name",
    "a blind bell-ringer", "a witch's apprentice", "a disgraced knight", "a lost twin",
    "a talking raven", "a lantern keeper", "a child who casts no shadow",
    "a healer who hears memories", "a king in disguise",
]
TWISTS = [
    "the narrator is the monster", "the villain is the hero's future self",
    "the prize was the trap", "the whole world lives inside something small",
    "the dead were protecting the living", "the gift is secretly a debt",
    "the guide lied to save them", "time was running backward",
    "the hero is the legend everyone fears", "the sacrifice was already made",
]
TONES = ["eerie", "wondrous", "bittersweet", "tense", "melancholic", "triumphant",
         "haunting", "darkly whimsical"]


def pick_seeds(date_str, salt=""):
    rng = random.Random(f"{date_str}|{salt}")
    cols = [rng.sample(SETTINGS, NUM_STORIES), rng.sample(PROTAGONISTS, NUM_STORIES),
            rng.sample(TWISTS, NUM_STORIES), rng.sample(TONES, NUM_STORIES)]
    return [dict(setting=a, hero=b, twist=c, tone=d) for a, b, c, d in zip(*cols)]


# --------------------------------------------------------------- schema ----
def _s():
    return {"type": "STRING"}


SCENE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"description": _s(), "narration": _s(), "image_prompt": _s(),
                   "character_visible": {"type": "BOOLEAN"}},
    "required": ["description", "narration", "image_prompt", "character_visible"],
}
STORY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": _s(), "genre": _s(), "logline": _s(), "character_look": _s(),
        "scenes": {"type": "ARRAY", "items": SCENE_SCHEMA},
        "cta": _s(), "music_mood": _s(),
        "sound_effects": {"type": "ARRAY", "items": _s()},
        "youtube_title": _s(), "youtube_description": _s(),
        "instagram_caption": _s(),
        "hashtags": {"type": "ARRAY", "items": _s()},
    },
    "required": ["title", "genre", "logline", "character_look", "scenes", "cta",
                 "music_mood", "sound_effects", "youtube_title",
                 "youtube_description", "instagram_caption", "hashtags"],
}
RESPONSE_SCHEMA = {"type": "OBJECT",
                   "properties": {"stories": {"type": "ARRAY", "items": STORY_SCHEMA}},
                   "required": ["stories"]}

SCHEMA_HINT = (
    'Return JSON exactly shaped like: {"stories":[{"title":"","genre":"","logline":"",'
    '"character_look":"","scenes":[{"description":"","narration":"","image_prompt":"",'
    '"character_visible":true}],"cta":"","music_mood":"","sound_effects":[""],'
    '"youtube_title":"","youtube_description":"","instagram_caption":"","hashtags":[""]}]}')

SYSTEM_PROMPT = ("You are a screenwriter for faceless, narrated fantasy short videos. "
                 "Reply with JSON only.")

RULES = """Rules:
- Each story: 6-8 scenes and 95-120 spoken words in total including the cta (about 45 seconds).
- Scene 1 narration must open with a hook of at most 10 words: a shocking claim, a question or an impossible image.
- Clear beginning, middle and end, a twist the viewer cannot guess, and a last line that lingers.
- Narration is spoken aloud: short sentences, emotional and cinematic, no emojis, no stage directions, no sound words, numbers spelled out.
- Invent original names and worlds. No existing franchises, no real people, no gore, nothing sexual.
- image_prompt: self-contained English, 25-45 words, one concrete subject, camera angle, lighting, mood. Never ask for text, logos or real people. Set character_visible true only if the hero is in the frame.
- character_look: the hero's fixed appearance in under 25 words.
- cta: under 12 words, spoken, teases the next story and asks viewers to follow.
- sound_effects: 3-5 short suggestions. music_mood: 3-6 words.
- youtube_title under 70 characters, curiosity-driven. youtube_description under 250 characters. instagram_caption under 250 characters, ending with a question. hashtags: 8-12 without the # sign, mixing broad and niche."""


def build_prompt(seeds, avoid, feedback, schema_hint):
    lines = [f"Write {NUM_STORIES} original, completely different fantasy stories "
             "for vertical short videos.", RULES,
             "Use these seeds as inspiration (invent original names):"]
    for i, s in enumerate(seeds, 1):
        lines.append(f"Story {i}: setting={s['setting']}; hero={s['hero']}; "
                     f"twist={s['twist']}; tone={s['tone']}")
    if avoid:
        lines.append("Do not reuse these recent titles or ideas: " + "; ".join(avoid))
    if feedback:
        lines.append("Your previous answer had problems. Fix all of them: "
                     + " | ".join(feedback))
    if schema_hint:
        lines.append(schema_hint)
    return "\n".join(lines)


# -------------------------------------------------------------- helpers ----
WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOP = set("the a an and or but of to in on at for with from by is are was were be been "
           "it its as that this he she they them his her their you your i we our not no "
           "so if then than into over under out up down just very who whom what when "
           "where why how".split())


def words(t):
    return WORD_RE.findall(t or "")


def content_set(t):
    return {w.lower() for w in words(t) if w.lower() not in STOP and len(w) > 2}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def squash(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def clean_narration(t):
    """Remove things a TTS voice would read aloud by mistake."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", str(t or ""))
    t = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf*_#~`]", "", t)
    return squash(t)


def first_sentence(t):
    m = re.match(r"\s*(.+?[.!?\u2026])(\s|$)", t or "", re.S)
    return (m.group(1) if m else (t or "")).strip()


def norm_tags(tags):
    out, seen = [], set()
    for t in tags or []:
        t = re.sub(r"[^\w]", "", str(t))
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append("#" + t)
    return out


def narration_seconds(text):
    return len(words(text)) / WORDS_PER_SEC


def final_image_prompt(ip, look, visible):
    parts = []
    if visible and look and look.lower() not in ip.lower():
        parts.append(look.rstrip(". "))
    parts.append(ip.rstrip(". "))
    parts.append(STYLE_SUFFIX)
    return ", ".join(parts)[:900]


def extract_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        return json.loads(t[a:b + 1])
    raise ValueError("no JSON object found")


def redact(text, key=None):
    t = str(text)
    if key:
        t = t.replace(key, "[REDACTED]")
    return re.sub(r"(AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-]{20,})", "[REDACTED]", t)


# ----------------------------------------------------------- validation ----
def build_story(raw, idx, run_date):
    """Validate one raw story and add derived fields. Returns (story|None, errs, warns)."""
    errs, warns = [], []
    tag = f"story {idx}"
    if not isinstance(raw, dict):
        return None, [f"{tag}: not a JSON object"], warns

    def text(key, maxlen):
        v = squash(raw.get(key))
        if not v:
            errs.append(f"{tag}: '{key}' is missing or empty")
        if len(v) > maxlen:
            warns.append(f"{tag}: '{key}' trimmed to {maxlen} chars")
            v = v[:maxlen].rstrip()
        return v

    title, genre = text("title", 80), text("genre", 40)
    logline, music = text("logline", 200), text("music_mood", 80)
    yt_title, yt_desc = text("youtube_title", 100), text("youtube_description", 1000)
    ig_caption = text("instagram_caption", 2200)
    look = squash(raw.get("character_look"))[:300]

    cta = clean_narration(raw.get("cta"))
    if not cta:
        errs.append(f"{tag}: 'cta' is missing")
    elif len(words(cta)) > 16:
        errs.append(f"{tag}: cta is too long ({len(words(cta))} words, max 16)")

    sfx_raw = raw.get("sound_effects")
    sfx = [squash(x)[:60] for x in sfx_raw if squash(x)][:6] if isinstance(sfx_raw, list) else []
    if len(sfx) < 2:
        errs.append(f"{tag}: need at least 2 sound_effects")

    model_tags = norm_tags(raw.get("hashtags") if isinstance(raw.get("hashtags"), list) else [])
    if len(model_tags) < 5:
        errs.append(f"{tag}: need at least 5 hashtags")
    tags = model_tags[:13]
    for req in ("#Fantasy", "#Shorts"):
        if req.lower() not in {x.lower() for x in tags}:
            tags.append(req)

    scenes_raw = raw.get("scenes")
    if not isinstance(scenes_raw, list):
        errs.append(f"{tag}: 'scenes' must be a list")
        return None, errs, warns
    if not SCENES_MIN <= len(scenes_raw) <= SCENES_MAX:
        errs.append(f"{tag}: {len(scenes_raw)} scenes (need {SCENES_MIN}-{SCENES_MAX})")

    scenes = []
    for n, s in enumerate(scenes_raw, 1):
        if not isinstance(s, dict):
            errs.append(f"{tag} scene {n}: not an object")
            continue
        desc, narr = squash(s.get("description")), clean_narration(s.get("narration"))
        ip, vis = squash(s.get("image_prompt")), bool(s.get("character_visible"))
        nw, iw = len(words(narr)), len(words(ip))
        if not desc:
            errs.append(f"{tag} scene {n}: description missing")
        if not 3 <= nw <= 35:
            errs.append(f"{tag} scene {n}: narration has {nw} words (need 3-35)")
        if not 8 <= iw <= 90:
            errs.append(f"{tag} scene {n}: image_prompt has {iw} words (need 8-90)")
        scenes.append({
            "scene": n, "description": desc, "narration": narr,
            "image_prompt": ip, "character_visible": vis,
            "image_prompt_final": final_image_prompt(ip, look, vis),
            "est_seconds": round(nw / WORDS_PER_SEC + SCENE_PAUSE_SEC, 1),
        })
    if not scenes:
        return None, errs, warns

    hook = first_sentence(scenes[0]["narration"])
    hw = len(words(hook))
    if hw > 16:
        errs.append(f"{tag}: opening hook has {hw} words (max 16, aim for 10)")
    elif hw > 12:
        warns.append(f"{tag}: opening hook has {hw} words (aim for 10 or fewer)")

    full = " ".join([s["narration"] for s in scenes] + [cta]).strip()
    duration = narration_seconds(full) + SCENE_PAUSE_SEC * len(scenes)
    if not DURATION_MIN <= duration <= DURATION_MAX:
        errs.append(f"{tag}: estimated duration {duration:.0f}s is outside "
                    f"{DURATION_MIN:.0f}-{DURATION_MAX:.0f}s ({len(words(full))} words)")

    blob = " ".join([title, logline, full] + [s["image_prompt"] for s in scenes])
    m = BANNED.search(blob)
    if m:
        errs.append(f"{tag}: uses protected franchise name '{m.group(0)}'")

    if errs:
        return None, errs, warns
    sid = f"{run_date:%Y%m%d}-s{idx}-{hashlib.sha1(title.lower().encode()).hexdigest()[:6]}"
    story = {
        "id": sid, "title": title, "genre": genre, "logline": logline,
        "character_look": look, "hook": hook, "full_narration": full,
        "estimated_duration_sec": round(duration, 1), "scene_count": len(scenes),
        "scenes": scenes, "music_mood": music, "sound_effects": sfx, "cta": cta,
        "youtube": {"title": yt_title, "description": yt_desc},
        "instagram": {"caption": ig_caption}, "hashtags": tags,
    }
    return story, errs, warns


def cross_checks(stories, history_titles=()):
    """Uniqueness checks across the stories (works on final story dicts)."""
    errs, warns = [], []
    sets = [content_set(s["full_narration"]) for s in stories]
    for i in range(len(stories)):
        for j in range(i + 1, len(stories)):
            sim = jaccard(sets[i], sets[j])
            if sim >= SIMILARITY_ERROR:
                errs.append(f"stories {i+1} and {j+1} are too similar ({sim:.2f})")
            elif sim >= SIMILARITY_WARN:
                warns.append(f"stories {i+1} and {j+1} are fairly similar ({sim:.2f})")
            if squash(stories[i]["title"]).lower() == squash(stories[j]["title"]).lower():
                errs.append(f"stories {i+1} and {j+1} share the same title")
    seen = set()
    for n, s in enumerate(stories, 1):
        for sc in s["scenes"]:
            k = sc["image_prompt"].lower()
            if k in seen:
                errs.append(f"story {n}: duplicate image prompt in scene {sc['scene']}")
            seen.add(k)
    hooks = [" ".join(words(s["hook"])[:3]).lower() for s in stories]
    if len(set(hooks)) < len(hooks):
        warns.append("two stories open with the same first three words")
    if len({s["genre"].lower() for s in stories}) < 2:
        warns.append("all stories have the same genre")
    for n, s in enumerate(stories, 1):
        for h in history_titles:
            if jaccard(content_set(s["title"]), content_set(h)) >= 0.8:
                errs.append(f"story {n}: title too close to recent title '{h}'")
    return errs, warns


def process_response(obj, run_date, history):
    stories, errs, warns = [], [], []
    raw_list = obj.get("stories") if isinstance(obj, dict) else None
    if not isinstance(raw_list, list):
        return [], ["response has no 'stories' list"], warns
    if len(raw_list) != NUM_STORIES:
        errs.append(f"expected exactly {NUM_STORIES} stories, got {len(raw_list)}")
    for i, raw in enumerate(raw_list[:NUM_STORIES], 1):
        st, e, w = build_story(raw, i, run_date)
        errs += e
        warns += w
        if st:
            stories.append(st)
    if not errs:
        e, w = cross_checks(stories, history)
        errs += e
        warns += w
    return stories, errs, warns


# ------------------------------------------------------------ Gemini API ----
class Fatal(Exception):
    pass


def auth_headers(key):
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def pick_models(session, key):
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return [forced]
    try:
        r = session.get(f"{API_BASE}/models?pageSize=200", headers=auth_headers(key),
                        timeout=(10, 30))
        if r.status_code == 200:
            avail = {m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])}
            chosen = [m for m in MODEL_CANDIDATES if m in avail]
            if chosen:
                return chosen
    except (requests.RequestException, ValueError, KeyError):
        pass
    return list(MODEL_CANDIDATES)


def call_gemini(session, key, model, prompt, use_schema):
    cfg = {"temperature": 1.0, "maxOutputTokens": MAX_OUTPUT_TOKENS,
           "responseMimeType": "application/json"}
    if use_schema:
        cfg["responseSchema"] = RESPONSE_SCHEMA
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg}
    return session.post(f"{API_BASE}/models/{model}:generateContent",
                        headers=auth_headers(key), json=body, timeout=(10, 120))


def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            items = json.load(f)
        titles = [str(i.get("title", "")) if isinstance(i, dict) else str(i) for i in items]
        return [t for t in titles if t][-20:]
    except (OSError, ValueError, AttributeError):
        return []


def generate(key, run_date, salt, sleep=time.sleep):
    session = requests.Session()
    models = pick_models(session, key)
    print(f"Candidate models: {models}")
    seeds = pick_seeds(str(run_date), salt)
    avoid = load_history()
    use_schema, mi, attempts, feedback = True, 0, 0, []
    usage = {"prompt": 0, "output": 0, "thoughts": 0, "total": 0}
    last_err = "no attempt was made"
    log = []

    while attempts < MAX_ATTEMPTS and mi < len(models):
        attempts += 1
        model = models[mi]
        prompt = build_prompt(seeds, avoid, feedback, None if use_schema else SCHEMA_HINT)
        print(f"Attempt {attempts}/{MAX_ATTEMPTS} model={model} schema={use_schema}")
        try:
            r = call_gemini(session, key, model, prompt, use_schema)
        except requests.RequestException as e:
            last_err = f"network error: {type(e).__name__}"
            log.append(last_err)
            print(last_err)
            sleep(min(5 * attempts, 20))
            continue

        st, body = r.status_code, r.text or ""
        low = body.lower()
        if st != 200:
            snippet = redact(body[:200].replace("\n", " "), key)
            last_err = f"HTTP {st} from {model}: {snippet}"
            log.append(f"HTTP {st} {model}")
            print(last_err)
            if st == 400 and use_schema and re.search(r"schema|mime", low):
                use_schema = False
                print("Schema rejected; retrying in plain JSON mode.")
            elif st == 404 or (st == 403 and "model" in low):
                mi += 1
            elif st == 429:
                if "perday" in low or "per day" in low or "limit: 0" in low:
                    print(f"Daily free quota exhausted for {model}; trying next model.")
                    mi += 1
                else:
                    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
                    wait = min(float(m.group(1)) + 1 if m else 20, 60)
                    print(f"Rate limited; waiting {wait:.0f}s.")
                    sleep(wait)
            elif st in (500, 502, 503, 504):
                sleep(min(5 * 3 ** (attempts - 1), 45))
            else:
                raise Fatal(f"HTTP {st}: {snippet} (check the key and its permissions)")
            continue

        try:
            data = r.json()
        except ValueError:
            last_err = "response body was not JSON"
            continue
        um = data.get("usageMetadata", {})
        usage["prompt"] += um.get("promptTokenCount", 0)
        usage["output"] += um.get("candidatesTokenCount", 0)
        usage["thoughts"] += um.get("thoughtsTokenCount", 0)
        usage["total"] += um.get("totalTokenCount", 0)

        cands = data.get("candidates") or []
        if not cands:
            block = data.get("promptFeedback", {}).get("blockReason", "unknown")
            last_err = f"no candidates returned (blockReason={block})"
            feedback = ["Keep the content family-friendly and original."]
            print(last_err)
            continue
        finish = cands[0].get("finishReason")
        parts = cands[0].get("content", {}).get("parts", [])
        out_text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if finish == "MAX_TOKENS":
            last_err = "output was cut off (MAX_TOKENS)"
            feedback = ["Be more concise so the whole JSON fits."]
            print(last_err)
            continue
        try:
            obj = extract_json(out_text)
        except ValueError as e:
            last_err = f"invalid JSON: {type(e).__name__}"
            feedback = ["Return one valid JSON object and nothing else."]
            print(last_err)
            continue

        stories, errs, warns = process_response(obj, run_date, avoid)
        for w in warns:
            print("warning:", w)
        if errs:
            last_err = "validation failed: " + "; ".join(errs[:5])
            feedback = errs[:8]
            print(last_err)
            continue
        meta = {"schema_version": SCHEMA_VERSION, "model": model,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "attempts": attempts, "json_schema_mode": use_schema,
                "token_usage": usage, "seeds": seeds, "warnings": warns,
                "words_per_second_assumed": WORDS_PER_SEC}
        return {"meta": meta, "stories": stories}

    raise RuntimeError(f"{last_err} (attempts used: {attempts}, models tried: {models[:mi + 1]})")


# --------------------------------------------------------------- verify ----
STORY_FIELDS = ["id", "title", "genre", "logline", "hook", "full_narration",
                "estimated_duration_sec", "scene_count", "scenes", "music_mood",
                "sound_effects", "cta", "youtube", "instagram", "hashtags"]
SCENE_FIELDS = ["scene", "description", "narration", "image_prompt",
                "image_prompt_final", "est_seconds"]


def verify_file(path):
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    try:
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()
        data = json.loads(raw_text)
        add("JSON is valid", True)
    except (OSError, ValueError) as e:
        add("JSON is valid", False, type(e).__name__)
        return _report(checks)

    stories = data.get("stories") if isinstance(data, dict) else None
    add("exactly 3 stories", isinstance(stories, list) and len(stories) == NUM_STORIES,
        f"found {len(stories) if isinstance(stories, list) else 'none'}")
    if not (isinstance(stories, list) and stories):
        return _report(checks)

    missing, bad_dur, bad_count = [], [], []
    for n, s in enumerate(stories, 1):
        for f in STORY_FIELDS:
            if s.get(f) in (None, "", [], {}):
                missing.append(f"story {n}.{f}")
        for k in ("title", "description"):
            if not (s.get("youtube") or {}).get(k):
                missing.append(f"story {n}.youtube.{k}")
        if not (s.get("instagram") or {}).get("caption"):
            missing.append(f"story {n}.instagram.caption")
        for sc in s.get("scenes") or []:
            for f in SCENE_FIELDS:
                if sc.get(f) in (None, "", [], {}):
                    missing.append(f"story {n} scene {sc.get('scene')}.{f}")
        d = s.get("estimated_duration_sec") or 0
        if not DURATION_MIN <= d <= DURATION_MAX:
            bad_dur.append(f"story {n}: {d}s")
        if s.get("scene_count") != len(s.get("scenes") or []):
            bad_count.append(f"story {n}")
    add("all required fields present", not missing, ", ".join(missing[:6]))
    add("duration 30-60s and scene counts consistent", not bad_dur and not bad_count,
        ", ".join(bad_dur + bad_count))

    ids = [s.get("id") for s in stories]
    try:
        e, w = cross_checks(stories)
    except (KeyError, TypeError) as ex:
        e, w = [f"cannot compare stories ({type(ex).__name__})"], []
    add("stories are unique", not e and len(set(ids)) == len(ids), "; ".join(e))
    for x in w:
        print("note:", x)

    key = os.environ.get("GEMINI_API_KEY", "")
    leaked = (key and key in raw_text) or re.search(r"AIza[0-9A-Za-z_\-]{20,}", raw_text)
    add("no API key inside the output", not leaked)

    for n, s in enumerate(stories, 1):
        print(f"  story {n}: {s.get('id')} | {s.get('title')} | {s.get('genre')} | "
              f"{s.get('estimated_duration_sec')}s | {s.get('scene_count')} scenes")
    return _report(checks)


def _report(checks):
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({d})" if d and not ok else "")
             for name, ok, d in checks]
    print("\n".join(lines))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("### Story verification\n```\n" + "\n".join(lines) + "\n```\n")
    return 0 if all(ok for _, ok, _ in checks) else 1


# -------------------------------------------------------------- selftest ----
def selftest():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} done') == {"a": 2}
    assert first_sentence("Who wakes the sea? Nobody knows.") == "Who wakes the sea?"
    assert norm_tags(["#Fantasy", "fantasy", "dark magic!", ""]) == ["#Fantasy", "#darkmagic"]
    assert clean_narration("He ran (gasps) *fast* \U0001F525") == "He ran fast"
    assert redact("key=AIza" + "A" * 30 + " ok").count("REDACTED") == 1
    s1, s2 = pick_seeds("2026-01-01", ""), pick_seeds("2026-01-01", "")
    assert s1 == s2 and len({x["setting"] for x in s1}) == NUM_STORIES
    st, errs, _ = build_story({}, 1, dt.date(2026, 1, 1))
    assert st is None and errs
    _, errs, _ = process_response({"stories": []}, dt.date(2026, 1, 1), [])
    assert errs
    base = {"title": "Same", "genre": "dark", "hook": "Who rang the bell?",
            "full_narration": "ancient bell rings beneath frozen harbor while lanterns flicker",
            "scenes": [{"scene": 1, "image_prompt": "a bell"}]}
    e, _ = cross_checks([base, dict(base), dict(base)])
    assert any("similar" in x for x in e) and any("same title" in x for x in e)
    ok = lambda t: DURATION_MIN <= narration_seconds(t) <= DURATION_MAX  # noqa: E731
    assert ok(" ".join(["word"] * 100))
    print("selftest OK (no API call was made)")
    return 0


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", metavar="FILE")
    ap.add_argument("--salt", default=os.environ.get("STORY_SALT", ""))
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, UTC)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.verify:
        return verify_file(args.verify)

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("ERROR: GEMINI_API_KEY is not set. Add it as a GitHub Secret.")
        return 2
    run_date = (dt.date.fromisoformat(args.date) if args.date
                else dt.datetime.now(dt.timezone.utc).date())
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        result = generate(key, run_date, args.salt)
    except (Fatal, RuntimeError) as e:
        msg = redact(e, key)
        print("FAILED:", msg)
        with open(ERR_FILE, "w", encoding="utf-8") as f:
            json.dump({"error": msg, "time": dt.datetime.now(dt.timezone.utc).isoformat()}, f, indent=2)
        return 1

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    m = result["meta"]
    print(f"\nSaved {OUT_FILE} | model={m['model']} attempts={m['attempts']} "
          f"tokens={m['token_usage']}")
    for s in result["stories"]:
        print(f"- {s['id']} | {s['title']} | {s['genre']} | "
              f"{s['estimated_duration_sec']}s | {s['scene_count']} scenes")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(f"### Stories generated\nModel `{m['model']}`, attempts {m['attempts']}, "
                    f"tokens {m['token_usage']['total']}\n\n")
            for s in result["stories"]:
                f.write(f"- **{s['title']}** ({s['genre']}, {s['estimated_duration_sec']}s, "
                        f"{s['scene_count']} scenes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
