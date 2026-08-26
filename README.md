# Instagram Content Analyzer

An experimental CLI prototype exploring whether a public Instagram URL can
eventually be turned into accurate, structured, multimodal information.

**This repository is at Phase 4 of that experiment. It does not analyze
content yet — extraction is proven viable but reliability has not been
benchmarked.**

## Purpose

The long-term feasibility question is whether this pipeline can be made to
produce trustworthy structured output:

```
Instagram URL
  -> URL validation
  -> Instagram content extraction
  -> Extraction validation
  -> Audio / speech understanding
  -> Vision understanding
  -> OCR / on-screen text
  -> Multimodal synthesis
  -> Structured JSON
  -> Cleanup
  -> Evaluation
```

Each stage is built and validated as its own phase so that a failure can be
attributed to a specific step rather than to the pipeline as a whole.

## Current phase

**Phase 4 - Authenticated Extraction Experiment (complete).** Phases 1
(environment), 2 (URL validation), and 3 (content extraction) are also
complete.

### Phase 1 - Environment & Project Setup (done)

- Project layout (`extractor/`, `processor/`, `output/`, `temp/`, `tests/`).
- `analyzer.py`, an `argparse` CLI entry point.
- `config.py`, a single configuration layer reading the environment or a
  local `.env` file, resolving working directories and configuring logging.
- An environment self-check and a smoke test suite.

## Phase 2 - URL Validation (done)

`extractor/url_validator.py`: an offline gate that inspects a user-supplied
string before it ever reaches the extraction layer.

### What the validator does

Given one string, it answers five questions using only the text:

1. Is it a syntactically valid URL?
2. Does it belong to Instagram?
3. Is it HTTPS? (the project is HTTPS-only)
4. Does its path look like supported Instagram content?
5. What content type can be inferred from the URL, if any?

It returns a frozen `ValidationResult` dataclass:

| Field               | Meaning                                              |
| ------------------- | ---------------------------------------------------- |
| `valid`             | `True` / `False`                                     |
| `input_url`         | The string as supplied                               |
| `normalized_url`    | Canonical URL, or `None` on failure                  |
| `content_type_hint` | `reel`, `video`, `post_or_unknown`, or `None`        |
| `shortcode`         | The extracted content identifier, or `None`          |
| `error_code`        | Machine-readable failure code, or `None` on success  |
| `error_message`     | Human-readable explanation, or `None` on success     |

Normalization is textual only:

- host lowercased and rewritten to `www.instagram.com` (so `instagram.com`,
  `m.instagram.com` and `instagr.am` collapse to one form)
- `/reels/` rewritten to `/reel/`
- profile-scoped paths reduced (`/nasa/reel/<code>/` becomes `/reel/<code>/`)
- query parameters (`?igsh=...`) and `#fragments` dropped
- exactly one trailing slash
- the shortcode's letter case preserved - Instagram shortcodes are case
  sensitive

### What the validator does NOT do

This phase is completely offline. The validator never:

- makes a network request, resolves a hostname or opens a socket
- contacts Instagram, downloads media, or reads cookies
- follows redirects (which is why `/share/` links are rejected rather than
  resolved)
- executes or evaluates anything taken from the URL
- proves that the post exists, is public, or is still available
- determines the real media type of the content

### Supported URL patterns

| Pattern                                         | Content type hint |
| ----------------------------------------------- | ----------------- |
| `https://www.instagram.com/reel/<code>/`        | `reel`            |
| `https://www.instagram.com/reels/<code>/`       | `reel`            |
| `https://www.instagram.com/tv/<code>/`          | `video`           |
| `https://www.instagram.com/p/<code>/`           | `post_or_unknown` |
| `https://www.instagram.com/<user>/reel/<code>/` | `reel`            |
| `https://www.instagram.com/<user>/p/<code>/`    | `post_or_unknown` |

Accepted hosts: `instagram.com`, `www.instagram.com`, `m.instagram.com`,
`instagr.am`, `www.instagr.am`. Host matching is exact, so lookalikes such as
`www.instagram.com.evil.example` are rejected.

### Content type hint vs. actual media type

**A hint is not a fact.** The URL is the only evidence available in this
phase, and it is weak evidence:

- `/reel/<code>/` and `/tv/<code>/` reliably indicate video content.
- `/p/<code>/` is ambiguous. That same URL shape serves a single image, a
  single video, and a multi-item carousel. Reporting it as
  `post_or_unknown` is deliberate honesty: the URL cannot distinguish them.

The PRD's four experimental content types - reel, video, carousel and
single-image post - are therefore **not** fully resolvable here. Carousel and
single-image post are indistinguishable from a URL and are resolved during
extraction (Phase 3), which may also correct a hint that turns out to be
wrong.

### Example valid inputs

```
https://www.instagram.com/reel/Cx1_ab-Z/
https://www.instagram.com/p/Cx1_ab-Z/
https://www.instagram.com/tv/Cx1_ab-Z/
https://www.instagram.com/reel/Cx1_ab-Z          (trailing slash added)
https://www.instagram.com/p/Cx1_ab-Z/?igsh=abc   (query dropped)
https://www.instagram.com/p/Cx1_ab-Z/#comments   (fragment dropped)
https://instagram.com/reel/Cx1_ab-Z/             (host normalized)
https://WWW.INSTAGRAM.COM/reel/Cx1_ab-Z/         (case normalized)
https://www.instagram.com/nasa/reel/Cx1_ab-Z/    (profile prefix removed)
www.instagram.com/reel/Cx1_ab-Z/                 (scheme added)
```

### Example invalid inputs

| Input                                          | Error code             |
| ---------------------------------------------- | ---------------------- |
| `""` (empty or whitespace only)                | `EMPTY_INPUT`          |
| `not-a-url`                                    | `INVALID_URL`          |
| `https://`                                     | `INVALID_URL`          |
| `https://www.instagram.com:8443/p/Ab/`         | `INVALID_URL`          |
| `http://www.instagram.com/reel/Ab/`            | `INVALID_SCHEME`       |
| `ftp://www.instagram.com/p/Ab/`                | `INVALID_SCHEME`       |
| `https://youtube.com/watch?v=test`             | `NON_INSTAGRAM_DOMAIN` |
| `https://www.instagram.com.evil.example/p/Ab/` | `NON_INSTAGRAM_DOMAIN` |
| `https://www.instagram.com/`                   | `UNSUPPORTED_PATH`     |
| `https://www.instagram.com/nasa/`              | `UNSUPPORTED_PATH`     |
| `https://www.instagram.com/stories/nasa/123/`  | `UNSUPPORTED_PATH`     |
| `https://www.instagram.com/share/reel/AbCdEf/` | `UNSUPPORTED_PATH`     |
| `https://www.instagram.com/p/Ab/liked_by/`     | `UNSUPPORTED_PATH`     |

Error codes are deterministic and drawn from a closed set (`ErrorCode.ALL`).
Ordinary bad input never raises an exception.

### Not yet implemented

Still absent by design:

- Instagram scraping, downloading, `yt-dlp`, or Instagram cookies
- Metadata extraction of any kind
- OCR, speech-to-text, audio processing, computer vision
- LLM or multimodal calls
- Structured JSON output
- Telegram integration, a database, persistent media storage, or a UI

The CLI validates the URL and stops.

## Phase 3 - Instagram Content Extraction

The current phase adds the extraction layer: given a URL that passed Phase 2,
it attempts to retrieve the media file, caption, hashtags, media type and
available metadata, and reports honestly when it cannot.

> **Successful Phase 3 extraction does not mean Instagram extraction is
> reliable enough for production. Reliability is determined later by the
> 20-URL benchmark.** A single URL succeeding (or failing) proves only what
> happened for that URL, at that moment, from this network.

### Objective

Answer, with real data: *given a valid public Instagram URL, can we reliably
retrieve the media and metadata the later multimodal pipeline needs?* A failed
URL is legitimate experimental data, not a bug to paper over.

### yt-dlp as the primary extractor

Extraction is backed by [`yt-dlp`](https://github.com/yt-dlp/yt-dlp), driven
programmatically (never by shelling out). No paid Instagram API or scraping
service is used, and no API key is required.

The rest of the app talks to the `BaseExtractor` interface, not to yt-dlp
directly. If the benchmark shows yt-dlp is insufficient, a second backend can
be added without touching the CLI.

### Authentication modes

Extraction supports three modes via `EXTRACTION_MODE`:

| Mode              | What it does                                            |
| ----------------- | ------------------------------------------------------- |
| `public`          | No Instagram session (default). No login needed.        |
| `browser_cookies` | yt-dlp borrows a local browser's cookies (`COOKIES_FROM_BROWSER`). |
| `cookie_file`     | yt-dlp reads a local Netscape-format cookie file (`COOKIE_FILE`). |

Across every mode, cookie handling is minimal and safe: no password is ever
requested or stored, cookie files are never committed, and cookie/session
values are never read, printed or logged by this project. Only a browser
*name* or a file *path* is passed to yt-dlp.

**Why `cookie_file` exists.** Public extraction currently returns
`AUTH_REQUIRED` (Instagram blocks logged-out clients for many posts). On this
Windows setup, `browser_cookies` with Chrome fails *before* Instagram is even
reached, with `Failed to decrypt with DPAPI` - a Chrome cookie-encryption
issue, not something this project tries to bypass. `cookie_file` is the
experimental local fallback: export your logged-in session to a
Netscape-format cookie file and point `COOKIE_FILE` at it.

This is an experimental local authentication path, **not** production auth
infrastructure. The cookie file:

- must remain private - **never commit it, never share it**
- is git-ignored (see `.gitignore`: `secrets/`, `*_cookies.txt`, `*.cookies`)
- is only ever referenced by *path*; its contents are never touched by this
  app

Configure it in `.env`:

```
EXTRACTION_MODE=cookie_file
COOKIE_FILE=secrets/instagram_cookies.txt
```

If `cookie_file` mode is selected but the file is missing, unreadable or
empty, extraction fails cleanly with `COOKIE_FILE_MISSING`,
`COOKIE_FILE_UNREADABLE` or `COOKIE_FILE_INVALID` - never a misleading
`AUTH_REQUIRED`.

> This README documents only the *mechanism*. It never contains real cookie
> values, and the project never creates the cookie file for you.

### Supported content categories

The PRD's four categories - public reel, public video, public carousel, public
single-image post - are all attempted. Crucially, the media type is decided by
**actual extraction data**, not guessed from the URL:

| Situation                         | `media_type`             |
| --------------------------------- | ------------------------ |
| `/reel/` video                    | `reel`                   |
| `/tv/` or single `/p/` video      | `video`                  |
| Multiple items                    | `carousel`               |
| Single image                      | `image`                  |
| Type could not be established     | `unknown` (not guessed)  |

### Extraction result structure

`ExtractionResult` (see `extractor/models.py`) serialises cleanly to a dict /
JSON via `as_dict()`. Key fields:

- `success`, `source_url`, `normalized_url`
- `media_downloaded`, `media_path`, `media_files[]`
- `caption_extracted`, `caption`, `hashtags[]`
- `media_type_detected`, `media_type`, `content_type_hint`
- `metadata` (sanitised), `download_seconds`, `run_id`, `extraction_mode`
- `failure_category`, `failure_reason`, `error_detail`

Success is not all-or-nothing:

- **Media present, caption missing** -> `success=true`,
  `caption_extracted=false`, `failure_category=CAPTION_UNAVAILABLE` (a partial
  shortfall recorded separately).
- **Caption present, media missing** -> `success=false`,
  `media_downloaded=false` (not a success - the next stage needs the media).
- **Media present, type uncertain** -> `media_type="unknown"`,
  `media_type_detected=false` (no invented type).

### Failure categories

Every failure carries an explicit machine-readable category
(`extractor/errors.py`, `FailureCategory.ALL`):

`URL_VALIDATION`, `EXTRACTION`, `MEDIA_UNAVAILABLE`, `CAPTION_UNAVAILABLE`,
`UNSUPPORTED_MEDIA`, `AUTH_REQUIRED`, `RATE_LIMITED`, `NETWORK_ERROR`,
`DOWNLOAD_ERROR`, `METADATA_ERROR`, `COOKIE_FILE_MISSING`,
`COOKIE_FILE_UNREADABLE`, `COOKIE_FILE_INVALID`, `UNKNOWN`.

Classification is conservative: `AUTH_REQUIRED` and `RATE_LIMITED` are only
claimed when the underlying error text supports them. A failed extraction
never looks like a success with blank fields.

### Post-download validation

After yt-dlp reports a download, each file is validated before it is handed
onward: it must exist, be non-empty, be readable, and carry a recognised media
extension. Files that fail are dropped; if yt-dlp claimed a download that then
fails validation, the result is `DOWNLOAD_ERROR` rather than a false success.

### Temporary media policy

Downloaded media is temporary. Each run gets an isolated directory under
`temp/<run-id>/` so concurrent or repeated runs never collide. Nothing is
stored permanently and media is never committed to git.

- A **failed** run cleans up its own partial downloads immediately.
- A **successful** run keeps the media until the caller is done with it (the
  CLI prints the report, then cleans up unless `--keep-media` is given). This
  is so a future Phase 4 consumer can read the file before it is removed.

### Metadata & secrets

Only a whitelist of non-sensitive metadata fields is retained. Formats blobs,
HTTP headers, cookies and signed URLs are dropped, so nothing sensitive is
kept, logged or printed. yt-dlp's colourised error text is stripped of ANSI
codes and truncated.

### Not yet implemented

Still absent by design: OCR, speech-to-text, audio analysis, vision analysis,
multimodal LLM processing, summary generation, Telegram, a database, search,
and persistent media storage.

## Phase 4 — Authenticated Extraction Experiment (done)

### Why authenticated extraction was necessary

Phase 3 extraction in `public` mode (no Instagram session) consistently
returned `AUTH_REQUIRED` — Instagram blocks logged-out clients from
retrieving media for many posts. The extraction layer worked correctly in
reporting this, but without authentication, no media could be retrieved for
the posts tested.

Phase 4 investigated whether local, cookie-based authentication could
bypass this restriction.

### Chrome browser-cookie DPAPI failure

The first attempt used `EXTRACTION_MODE=browser_cookies` with
`COOKIES_FROM_BROWSER=chrome`. This failed *before* Instagram was even
contacted: yt-dlp raised `Failed to decrypt with DPAPI`, a Chrome
cookie-encryption issue on this Windows setup. This is a Chrome/OS
limitation, not a bug in this project or in yt-dlp. The project does not
attempt to bypass DPAPI.

### cookie_file mode

The working path was `EXTRACTION_MODE=cookie_file`, where a manually
exported Netscape-format cookie file (`secrets/instagram_cookies.txt`) is
passed by *path* to yt-dlp. The application never reads, prints, logs or
copies the cookie contents.

Configuration:

```
EXTRACTION_MODE=cookie_file
COOKIE_FILE=secrets/instagram_cookies.txt
```

### Security model

- The cookie file lives in `secrets/`, which is git-ignored.
- `*_cookies.txt`, `*.cookies` and `cookies.txt` are all git-ignored.
- `.env` is git-ignored.
- Cookie contents are never read, printed, logged or copied by this project.
- Only the file *path* is passed to yt-dlp.
- Unit tests use synthetic cookie fixtures with fake values — they never
  access the real cookie file.
- No credential is committed to the repository.

### Real test performed

A real Instagram post was extracted using `scripts/test_extraction.py`:

```bash
python scripts/test_extraction.py "https://www.instagram.com/p/DcWXVZlMwOB/"
```

### Observed result

```
Extraction mode: cookie_file
Cookie file configured: yes

[OK  ] https://www.instagram.com/p/DcWXVZlMwOB/
        media_downloaded : True
        caption_extracted: True
        media_type       : video (detected=True)
        download_seconds : 5.324

Summary: 1/1 succeeded.
```

The result is recorded in `output/extraction_benchmark.jsonl`.

### What Phase 4 proves

Authenticated cookie-file extraction is **technically viable**: for the
tested URL, yt-dlp successfully retrieved a real Instagram video, its
caption, and correctly detected the media type, in 5.324 seconds.

### What Phase 4 does NOT prove

> **Important:** A single successful extraction does not establish overall
> reliability.

Phase 4 does **not** prove:

- That extraction is ≥90% reliable across diverse content
- That all public content can be extracted
- That all Reels work
- That all carousels work
- That all image posts work
- That the 20-URL benchmark passes

Reliability is determined later by the 20-URL benchmark, after the
multimodal pipeline is implemented.

## Requirements

- Python 3.9 or newer (developed on 3.14)
- `yt-dlp` (installed via `requirements.txt`) for Phase 3 extraction

## Setup

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file (optional - the defaults work without
it):

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

`.env` is git-ignored. Never commit real API keys or cookie files.

## Running the CLI

```bash
python analyzer.py "https://www.instagram.com/reel/example/"
```

Successful extraction:

```
Instagram Content Analyzer
Phase: Instagram Content Extraction

Input URL:
https://www.instagram.com/reel/example/
Normalized URL:
https://www.instagram.com/reel/example/

URL validation: PASS

Extraction: PASS

Media downloaded: PASS
Caption extracted: PASS
Media type detected: PASS

Media type:
reel

Media path:
temp/<run-id>/<id>.mp4

Caption:
...

Hashtags:
#travel #sunset

No multimodal analysis is being performed yet.
```

Failed extraction (real example - current Instagram blocks logged-out clients):

```
URL validation: PASS

Extraction: FAIL

Media downloaded: FAIL
Caption extracted: FAIL
Media type detected: FAIL

...

Failure category:
AUTH_REQUIRED
Failure reason:
Instagram sent an empty media response ... use --cookies-from-browser ...
```

A rejected URL stops before extraction (exit code 3); a failed extraction
exits 4. Without a URL the CLI prints usage and exits 1.

Options:

```bash
python analyzer.py --help          # full option list
python analyzer.py --version       # project name and phase
python analyzer.py -v "<url>"      # DEBUG logging + sanitised diagnostics
python analyzer.py --keep-media "<url>"   # do not delete downloaded media
```

Logging is written to stderr, so stdout stays clean and predictable. Emoji in
captions are handled even on a legacy Windows console.

## Running the tests

`pytest` is **fully offline** - it mocks yt-dlp and never contacts Instagram:

```bash
pytest
```

### Live extraction test (separate, explicit)

Real Instagram extraction is deliberately kept out of `pytest`. Run it by hand
to gather benchmark data:

```bash
python scripts/test_extraction.py "https://www.instagram.com/reel/XXXX/"
python scripts/test_extraction.py url1 url2 url3
python scripts/test_extraction.py --keep-media "https://www.instagram.com/p/XXXX/"
python scripts/test_extraction.py --record "https://www.instagram.com/reel/XXXX/"
```

Each URL prints a row (success, media, caption, media type, failure category,
download time). `--record` appends one JSON line per URL to
`output/extraction_benchmark.jsonl` for the eventual 20-URL benchmark. A URL
that Instagram blocks is recorded as a real result, not hidden.

The runner honours the configured `EXTRACTION_MODE`. To try an authenticated
run with a local cookie file, set `.env`:

```
EXTRACTION_MODE=cookie_file
COOKIE_FILE=secrets/instagram_cookies.txt
```

then:

```bash
python scripts/test_extraction.py "https://www.instagram.com/reel/XXXX/"
```

It prints the extraction mode and `Cookie file configured: yes/no` - but never
the cookie contents.

## Configuration

All settings are read through `config.py`. Nothing else touches
`os.environ`, so the configuration surface stays in one file as the project
grows.

| Variable            | Default  | Purpose                                     |
| ------------------- | -------- | ------------------------------------------- |
| `LOG_LEVEL`            | `INFO`   | Console log verbosity                        |
| `OUTPUT_DIR`           | `output` | Where generated results are written          |
| `TEMP_DIR`             | `temp`   | Short-lived media scratch space              |
| `EXTRACTION_MODE`      | `public` | `public`, `browser_cookies` or `cookie_file` |
| `COOKIES_FROM_BROWSER` | empty    | Browser name for `browser_cookies` mode      |
| `COOKIE_FILE`          | empty    | Cookie-file path for `cookie_file` mode      |
| `ANTHROPIC_API_KEY`    | empty    | Placeholder for later phases                 |
| `OPENAI_API_KEY`       | empty    | Placeholder for later phases                 |

## Project structure

```
.
├── analyzer.py          # CLI entry point
├── config.py            # Configuration + logging
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── extractor/                 # Phase 2 validation + Phase 3 extraction
│   ├── __init__.py
│   ├── url_validator.py       # Phase 2: offline URL validation
│   ├── instagram_extractor.py # Phase 3: yt-dlp extraction backend
│   ├── models.py              # ExtractionResult, media/mode enums
│   ├── errors.py              # FailureCategory, ExtractionError
│   └── artifacts.py           # per-run temp directory manager
├── processor/                 # Phase 5+: audio, vision, OCR, synthesis
│   └── __init__.py
├── scripts/
│   └── test_extraction.py     # live extraction runner (explicit, networked)
├── output/                    # Generated results (git-ignored)
│   └── extraction_benchmark.jsonl  # Phase 4 real extraction results
├── temp/                      # Per-run downloaded media (git-ignored)
├── secrets/                   # Cookie files (git-ignored, NEVER committed)
└── tests/
    ├── __init__.py
    ├── test_setup.py          # project + CLI smoke tests (offline)
    ├── test_url_validator.py  # validator unit tests (offline)
    └── test_extractor.py      # extraction + cookie_file unit tests (offline, mocked yt-dlp)
```

## Exit codes

| Code | Meaning                                     |
| ---- | ------------------------------------------- |
| `0`  | Success                                     |
| `1`  | No URL provided (usage message printed)     |
| `2`  | Environment check failed                    |
| `3`  | URL failed validation                       |
| `4`  | Extraction failed                           |

## Status

Phases 1 (environment), 2 (URL validation), 3 (Instagram content extraction)
and 4 (authenticated extraction experiment) are **complete**.

**Proven:** Authenticated cookie-file extraction successfully retrieved a
real Instagram video, caption, and media type (5.324 s).

**Not proven:** Overall extraction reliability target of ≥90%. This is
determined by the 20-URL benchmark, which runs after the multimodal
pipeline is implemented.

**Remaining:** Phase 5 (multimodal analysis: ASR, OCR, vision, LLM
synthesis) has not been started.
