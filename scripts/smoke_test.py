import os, sys, json, base64, subprocess
import requests
from PIL import Image

OUT = "out"
os.makedirs(OUT, exist_ok=True)
results = {}


def secret(name):
    v = os.environ.get(name, "").strip()
    print(f"[{'ok' if v else 'MISSING'}] secret {name}")
    return v


CF_ACC = secret("CF_ACCOUNT_ID")
CF_TOK = secret("CF_API_TOKEN")
GEM = secret("GEMINI_API_KEY")


def err_text(r):
    try:
        return r.json().get("error", {}).get("message", r.text)[:250]
    except Exception:
        return r.text[:250]


# ---------- 1. Gemini free tier ----------
def test_gemini():
    if not GEM:
        return False, "no key"
    base = "https://generativelanguage.googleapis.com/v1beta"
    h = {"x-goog-api-key": GEM, "Content-Type": "application/json"}
    r = requests.get(f"{base}/models?pageSize=200", headers=h, timeout=30)
    print("Gemini list models -> HTTP", r.status_code)
    names = []
    if r.ok:
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].replace("models/", ""))
        print("Models with generateContent containing 'flash':",
              [n for n in names if "flash" in n][:40])
    else:
        print("  ", err_text(r))
        return False, "list models failed"

    wanted = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
              "gemini-3.5-flash", "gemini-2.5-flash"]
    candidates = [n for n in wanted if n in names][:4]
    working = []
    body = {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
            "generationConfig": {"maxOutputTokens": 20}}
    for n in candidates:
        rr = requests.post(f"{base}/models/{n}:generateContent",
                           headers=h, json=body, timeout=60)
        print(f"  test {n} -> HTTP {rr.status_code}",
              "" if rr.ok else "| " + err_text(rr))
        if rr.ok:
            working.append(n)
    print("WORKING TEXT MODELS ON YOUR KEY:", working)
    return bool(working), ", ".join(working) or "none worked"


# ---------- 2. Cloudflare FLUX ----------
IMG_PATH = os.path.join(OUT, "test_image.jpg")


def test_cloudflare():
    if not (CF_ACC and CF_TOK):
        return False, "missing secrets"
    url = (f"https://api.cloudflare.com/client/v4/accounts/{CF_ACC}"
           "/ai/run/@cf/black-forest-labs/flux-1-schnell")
    h = {"Authorization": f"Bearer {CF_TOK}"}
    prompt = ("cinematic fantasy illustration, a lone traveler before a "
              "glowing floating castle at dusk, mist, rich purple and gold "
              "palette, dramatic lighting, highly detailed")
    r = requests.post(url, headers=h, timeout=120,
                      json={"prompt": prompt, "steps": 4,
                            "width": 768, "height": 1344})
    print("Cloudflare (with width/height) -> HTTP", r.status_code)
    if not r.ok:
        print("  ", err_text(r))
        r = requests.post(url, headers=h, timeout=120,
                          json={"prompt": prompt, "steps": 4})
        print("Cloudflare (prompt+steps only) -> HTTP", r.status_code)
        if not r.ok:
            print("  ", err_text(r))
            return False, "image request failed"
    if r.headers.get("content-type", "").startswith("image/"):
        data = r.content
    else:
        data = base64.b64decode(r.json()["result"]["image"])
    with open(IMG_PATH, "wb") as f:
        f.write(data)
    w, hgt = Image.open(IMG_PATH).size
    print(f"Image saved: {w}x{hgt}, {len(data)//1024} KB")
    return True, f"{w}x{hgt}"


# ---------- 3. FFmpeg render ----------
def test_ffmpeg():
    if not os.path.exists(IMG_PATH):
        return False, "no image to render"
    vf = ("scale=1620:2880:force_original_aspect_ratio=increase,"
          "crop=1620:2880,"
          "zoompan=z='min(zoom+0.0015,1.15)':d=75:s=1080x1920:fps=25,"
          "format=yuv420p")
    out = os.path.join(OUT, "test_clip.mp4")
    p = subprocess.run(["ffmpeg", "-y", "-i", IMG_PATH, "-vf", vf,
                        "-c:v", "libx264", "-preset", "veryfast", out],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-600:])
        return False, "ffmpeg failed"
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,duration",
                        "-of", "json", out], capture_output=True, text=True)
    s = json.loads(q.stdout)["streams"][0]
    print("Rendered clip:", s)
    ok = s["width"] == 1080 and s["height"] == 1920
    return ok, f"{s['width']}x{s['height']}, {float(s['duration']):.1f}s"


for name, fn in [("gemini", test_gemini), ("cloudflare_image", test_cloudflare),
                 ("ffmpeg_render", test_ffmpeg)]:
    print(f"\n===== {name} =====")
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = (False, f"exception: {type(e).__name__}")
        print("EXCEPTION:", type(e).__name__, str(e)[:200])

print("\n===== SUMMARY =====")
lines = []
for k, (ok, note) in results.items():
    line = f"{'PASS' if ok else 'FAIL'}  {k}: {note}"
    print(line)
    lines.append(line)
summ = os.environ.get("GITHUB_STEP_SUMMARY")
if summ:
    with open(summ, "a") as f:
        f.write("### Smoke test\n```\n" + "\n".join(lines) + "\n```\n")
sys.exit(0 if all(ok for ok, _ in results.values()) else 1)
