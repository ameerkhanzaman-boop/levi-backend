"""
LEVI Backend Server
Fetches content from Fabo help centre and passes it to Groq AI
"""

import os
import urllib.request
import urllib.error
import html.parser
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

NAMED_PAGES = {
    "help-centre": "https://fabo.org/dca/helpcentre",
    "starter-kit": "https://fabo.org/community/fabo_starter_kit",
}

ALLOWED_URL_PREFIXES = [
    "https://fabo.org/",
]

DISCOVERED_LINKS = {}

LEVI_SYSTEM_PROMPT = """Identity & Goal: You are LEVI, the specialized help assistant for the Fabo learning platform. Fabo enables organizations to build internal learning sites and courses for their employees, including custom organizational landing pages, customizable course experiences, and interactive elements such as surveys. Your primary goal is to help Fabo learners and platform users succeed by clearly explaining how Fabo features work, guiding users through platform tasks step-by-step, troubleshooting issues using only the context the user can provide in chat, and generating high-quality learning content and copy when requested. You are empathetic, patient, and solution-oriented: you assume the user is acting in good faith, you never shame people for being confused, and you prioritize getting them unstuck as efficiently as possible while helping them understand what's happening.

Navigation Rules: You support multiple user types and adapt your guidance accordingly. Fabo users may be participants (taking courses), editors (building courses, pages), or admins (managing sites, users, permissions, access, and publishing). If the user's role affects the answer and it is not clear, you ask a single clarifying question such as: Are you using Fabo as a learner, course creator, or admin? Because Fabo is a logged-in platform, you do not have direct access to the user's organization, account, settings, courses, pages, analytics, or internal content unless the user provides that information explicitly in chat. You must never claim that you checked their account or that you see their dashboard. You must follow strict safety GDPR compliant boundaries: never ask for passwords, magic links, MFA codes, or anything that grants account access. Never invent features, settings, menu items, or limitations. If uncertain, present a small set of plausible explanations and provide a quick concrete verification method inside the UI.

Flow & Personality: Your tone is warm, professional, calm, and pragmatic. You never respond rudely, dismissively, or ambiguously. End most replies with a clear next step. Never request passwords or MFA codes, never claim to access the user's private workspace.

Scope: LEVI's scope is strictly limited to helping users with the Fabo platform. If a user asks for anything unrelated to Fabo, politely refuse and redirect back to Fabo.

IMPORTANT: You have been provided with real content fetched live from the Fabo help centre. Always use this content as your primary source of truth when answering questions. At the end of your response always include: Find out more information on: [the URL you fetched from]"""


class _ContentExtractor(html.parser.HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "aside", "iframe"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self):
        raw = "".join(self._chunks)
        lines = [line.strip() for line in raw.splitlines()]
        cleaned = []
        prev_blank = False
        for line in lines:
            if line == "":
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        return "\n".join(cleaned).strip()


class _LinkExtractor(html.parser.HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.links = []
        self._current_href = None
        self._current_label_chunks = []
        self._seen_urls = set()

    def _make_absolute(self, href):
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("/"):
            parts = self.base_url.split("/")
            domain = "/".join(parts[:3])
            return domain + href
        base_dir = self.base_url.rstrip("/").rsplit("/", 1)[0]
        return base_dir + "/" + href

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if not href or href.startswith("#") or href.startswith("mailto:"):
                self._current_href = None
                return
            full_url = self._make_absolute(href)
            if any(full_url.startswith(p) for p in ALLOWED_URL_PREFIXES):
                self._current_href = full_url
                self._current_label_chunks = []
            else:
                self._current_href = None

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_label_chunks.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href:
            label = " ".join("".join(self._current_label_chunks).split()).strip()
            url = self._current_href
            if url not in self._seen_urls and label:
                self._seen_urls.add(url)
                self.links.append({"url": url, "label": label})
            self._current_href = None
            self._current_label_chunks = []


def fetch_url(url, timeout=15):
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "levi-backend/1.0 (internal content reader)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            content_type = resp.headers.get_content_type() or "text/html"
            charset = resp.headers.get_content_charset() or "utf-8"

        if "html" not in content_type:
            return None, None, None

        parser = _ContentExtractor()
        parser.feed(raw.decode(charset, errors="replace"))
        text = parser.get_text()
        return text, raw, charset
    except Exception:
        return None, None, None


def discover_links():
    for shortcut, url in NAMED_PAGES.items():
        text, raw, charset = fetch_url(url)
        if not raw:
            continue
        html_str = raw.decode(charset or "utf-8", errors="replace")
        link_parser = _LinkExtractor(url)
        link_parser.feed(html_str)
        for link in link_parser.links:
            key = link["label"].lower()
            DISCOVERED_LINKS[key] = link["url"]
            DISCOVERED_LINKS[link["label"]] = link["url"]
    print(f"[levi-backend] Discovered {len(DISCOVERED_LINKS) // 2} links")


def find_best_match(query):
    if query in DISCOVERED_LINKS:
        return DISCOVERED_LINKS[query]
    query_lower = query.lower()
    if query_lower in DISCOVERED_LINKS:
        return DISCOVERED_LINKS[query_lower]
    for key, url in DISCOVERED_LINKS.items():
        if query_lower in key.lower():
            return url
    return None


def fetch_relevant_content(user_message):
    url = find_best_match(user_message)
    if not url:
        for shortcut, page_url in NAMED_PAGES.items():
            url = page_url
            break
    if not url:
        return None, None
    text, _, _ = fetch_url(url)
    return text, url


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        fetched_content, fetched_url = fetch_relevant_content(user_message)

        system_prompt = LEVI_SYSTEM_PROMPT
        if fetched_content:
            system_prompt += f"\n\n--- LIVE CONTENT FETCHED FROM FABO HELP CENTRE ---\nSource: {fetched_url}\n\n{fetched_content[:4000]}\n--- END OF FETCHED CONTENT ---"

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        req_data = json.dumps({
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }).encode("utf-8")

        req = urllib.request.Request(
            GROQ_URL,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        reply = result["choices"][0]["message"]["content"]
        return jsonify({
            "reply": reply,
            "source_url": fetched_url
        })

    except Exception as e:
        print(f"[levi-backend] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "links_discovered": len(DISCOVERED_LINKS) // 2})


if __name__ == "__main__":
    discover_links()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
