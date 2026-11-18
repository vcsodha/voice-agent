# Real-Time Voice Agent

Speech-to-text → Claude → text-to-speech, with barge-in (interrupt the
assistant mid-sentence, then say "continue" to make it pick up where you cut
it off) and RAG grounding over your own documents. Everything except the
Claude API call runs locally for free.

**Live demo:** https://vidisha-voice-agent.vercel.app

It runs on Vercel behind an access code, so strangers can't run up the API
bill. The page loads for everyone, but talking needs the code: ask Vidisha for
it, or run it yourself with the steps below.

## Architecture

```
 Browser mic ──PCM16/16kHz──▶ WebSocket (/ws) ──▶ VAD (turn detection)
                                                    │
                                            (turn ends)
                                                    ▼
                                          faster-whisper (STT)
                                                    │
                                                    ▼
                                     Chroma RAG lookup (local embeddings)
                                                    │
                                                    ▼
                                        Claude API (streamed tokens)
                                                    │
                                          per-sentence chunking
                                                    ▼
                                            Piper (TTS, local)
                                                    │
                                ◀── one WAV per sentence ┘
 Browser playback ◀─────────────────────────────────────
        (VAD also watches the mic while the assistant is
         speaking → barge-in cancels generation instantly)
```

## Cost

| Part | What it uses | Cost |
|------|--------------|------|
| Speech to text | faster-whisper, on your CPU | free |
| Text to speech | Piper, on your CPU | free |
| Voice activity detection | webrtcvad | free |
| Knowledge base | Chroma + its built-in local embedding model | free |
| LLM | Claude Haiku 5.5 through the API | $0.10 per million input tokens, $0.50 per million output tokens ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)) |

A spoken turn is a couple of thousand input tokens and under 400 output
tokens (`MAX_TOKENS` in `config.py`), so a 10-turn conversation costs well
under a cent. The API is pay-as-you-go and separate from a claude.ai
subscription: you add a few dollars of credit under Billing in the console.

## Run it

You need a Claude API key (https://console.anthropic.com, then "API keys").
Both options below start the same app on **http://localhost:8000**. Use
Chrome, Edge or Firefox and allow the microphone when asked.

### Option A: Docker (one command, any OS)

```bash
cp .env.example .env          # Windows: copy .env.example .env
# open .env in an editor and paste your key after ANTHROPIC_API_KEY=
docker compose up --build
```

The first build downloads the Python packages and every model (about 2 GB
in total, so it takes a few minutes). Starting it afterwards is quick. Wait
for the log line `Voice agent ready.`, then open the page.
Stop with `Ctrl+C`, and `docker compose down` to clean up.

### Option B: Python on your machine

Needs Python 3.11, 3.12 or 3.13 (`python3 --version`).

```bash
# 1. from the project folder, make an isolated environment
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

# 2. install the packages
pip install -r backend/requirements.txt

# 3. download the voice (about 60 MB) into models/
python -m piper.download_voices --download-dir models en_US-lessac-medium

# 4. add your API key
cp .env.example .env                  # Windows: copy .env.example .env
# open .env in an editor and paste your key after ANTHROPIC_API_KEY=

# 5. start the server (it must be run from the backend folder)
cd backend
uvicorn main:app --port 8000
```

The first start downloads the Whisper model and the embedding model (one
time, roughly 150 MB and 80 MB). Wait for `Voice agent ready.`, then open
http://localhost:8000. Stop with `Ctrl+C`.

On later days you only need: activate the venv, `cd backend`,
`uvicorn main:app --port 8000`.

## Try it

1. Click the mic. The status line says **listening**.
2. Ask: *"What is this project?"* The answer comes from
   `data/knowledge/about_this_agent.txt`.
3. Test interrupt and resume: ask *"Explain in detail how the internet
   works."* and let it get through two or three sentences. Then cut in with
   a word like **"wait"**. It stops straight away and stays quiet. Now say
   **"continue"** and it picks up from the last sentence you heard.
   Ask something different instead and it moves on to the new question.

Headphones help a lot: with laptop speakers the mic can hear the assistant
and interrupt it by mistake.

## Put it online (Vercel)

Vercel can run this whole app, page and voice server together, for free on
the Hobby plan. It runs the `Dockerfile` as a container function (a beta
feature) and supports WebSockets. Read "What to expect" before you rely on it.

**1. Put the code on GitHub** (from the project folder; `.gitignore` already
keeps `.env`, the models and the venv out):

```bash
git init
git add .
git status                      # check that .env is NOT in this list
git commit -m "Real-time voice agent"
gh repo create voice-agent --public --source=. --push
```

No `gh`? Make an empty repo on github.com, then `git remote add origin <its url>`
and `git push -u origin main`.

**2. Import it into Vercel.** On vercel.com choose Add New, then Project, then
pick the repo. Vercel finds the `Dockerfile` and uses the Container preset
(`vercel.json` also asks for it). Before you deploy, add these environment
variables:

| Name | Value |
|------|-------|
| `ANTHROPIC_API_KEY` | your key |
| `PORT` | `8000` (tells Vercel which port the server listens on) |
| `ACCESS_CODE` | a long random phrase, only people with the link can talk |
| `MAX_TURNS_PER_DAY` | `40` (everyone together) |
| `MAX_TURNS_PER_SESSION` | `15` |
| `MAX_CONNECTIONS` | `2` |
| `LOG_TRANSCRIPTS` | `0` (so visitors' words aren't kept in your logs) |

**3. Share the link with the code on the end:**
`https://your-project.vercel.app/?code=THE_CODE`. Without the code the page
loads but the assistant refuses to connect.

**4. Set a spend limit in the Anthropic console** (Settings, then Limits). The
limits above already keep usage tiny (a turn costs well under a cent), but a
limit there is the real safety net.

What the limits do: the access code stops strangers who find the URL.
`MAX_TURNS_PER_DAY` caps the total number of questions across all visitors per
day, `MAX_TURNS_PER_SESSION` caps one visit, and `MAX_CONNECTIONS` caps how many
people can talk at the same moment (speech recognition is the heavy part).
Leave them out and nothing is limited, which is what you want on your laptop.

**What to expect on the free plan**

- A session ends after about 5 minutes (Hobby limit for a WebSocket). The page
  says so and you tap Start to begin again; the conversation memory starts over.
- The server has 1 CPU and 2 GB of memory, so a reply takes a few seconds
  longer than on a laptop. After 5 quiet minutes it stops, and the next visitor
  waits for it to start again (the models are inside the image, so nothing is downloaded).
- Hobby includes 4 CPU-hours a month. My estimate is that this covers a bit more
  than a thousand questions; `MAX_TURNS_PER_DAY=40` keeps you under it. If you go
  over, Vercel pauses the project for the rest of the 30 days instead of billing you.
- This is running on a Hobby plan (deployed and tested in November 2026). The
  container runtime is still a beta feature, so if a deploy fails, read the
  build log and the runtime log in the Vercel dashboard first.
- Hobby plans are for personal, non-commercial use, which a portfolio project is.
- Vercel keeps the built image in its container registry and charges $0.10 per
  GB of storage. The image is about 1.2 GB, so expect cents a month at most.
- The limits (`MAX_TURNS_PER_DAY`, `MAX_CONNECTIONS`) are counted inside each
  running server. If Vercel starts a second copy under load, each copy counts
  on its own, which is another reason to set the spend limit in step 4.
- If the build fails with "exceeds the maximum uncompressed size limit", the
  image is bigger than your project's function limit. New projects allow large
  functions already (that is a project setting, not an environment variable you
  can add). To see what is big, add `VERCEL_ANALYZE_BUILD_OUTPUT=1` and read the build log.
- If the build works but the site shows 500 and the runtime log says the speech
  model cannot be found ("outgoing traffic has been disabled"), the models are
  not where the server looks. The `Dockerfile` pins those folders (`HF_HOME`, and
  `HOME` in the start command) and checks them while building, so keep both lines.
- If visitors see a Vercel login page instead of your app, open Settings, then
  Deployment Protection in the Vercel project and turn Vercel Authentication
  off for the production domain. Share the short project address
  (`your-project.vercel.app`), not the long one with random letters.
- To pick a nicer address: in the project, open Domains, click Edit on the
  `.vercel.app` domain, type a new name and choose "Redirect old domain to new".

Prefer another host? Any host that runs a Docker image works: build it, set
the same environment variables and open port 8000. If the page itself lives
somewhere else, put the server's address in `frontend/config.js` (`backendUrl`).

## If something goes wrong

| What you see | Fix |
|--------------|-----|
| `Setup problem:` when starting | Read the message under it. It tells you whether the API key or the voice file is missing, and how to fix it. |
| `ModuleNotFoundError` | The venv isn't active, or you're not in the `backend` folder. |
| `pip install` fails | Check `python3 --version` is 3.11 to 3.13. |
| Browser can't use the mic | Open it as `http://localhost:8000` (not an IP address) and allow the mic in the address-bar lock icon. |
| An orange `Error:` message on the page | `401` = wrong API key. `404`/model error = wrong `CLAUDE_MODEL`. "credit balance too low" = add credit in the console. |
| It cuts you off when you pause to think | Raise `SILENCE_MS_TO_END_TURN` in `backend/config.py`. |
| It interrupts itself | Use headphones, or raise `MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN`. |
| Port 8000 is taken | Use `--port 8001` (or change `8000:8000` in `docker-compose.yml`) and open that port. |

## Your own knowledge base

Put `.txt` or `.md` files in `data/knowledge/` and restart the server. The
folder is the source of truth: on every start the old index is cleared and
rebuilt from the files, so edited or deleted files don't leave stale answers
behind. A good one to try is a text file about yourself, your projects and
your experience, so people can talk to your portfolio.

## Interrupt and resume

While the assistant talks the mic stays open. If you talk over it (300 ms of
sustained speech, see `MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN`):

1. the server cancels the running Claude/TTS task and tells the browser to
   stop playing,
2. it saves `{question, said_so_far}` where `said_so_far` is only the
   sentences the browser **actually finished playing** (the browser sends a
   `chunk_played` message after each sentence, because the server sends
   audio much faster than it plays, so "sent" is not the same as "heard"),
3. your next sentence is transcribed (`intent.py`, small keyword checks
   with a length limit):
   - a "continue" request: the agent asks Claude to pick up right after
     what you heard.
   - a "wait" / "stop" / "hold on": not a question, so it stays quiet and
     keeps the paused answer for a later "continue".
   - anything else is a new question, and the part you heard stays in the
     history marked `[interrupted]` so context isn't lost.

## Tests

```bash
pip install -r backend/requirements-dev.txt
cd backend
pytest
```

119 automated tests, no models or API key needed (Whisper, Piper, Chroma and
Claude are replaced by small fakes in `tests/conftest.py`):

| File | What it covers |
|------|----------------|
| `test_intent.py` | continue / pause / new-question phrases, false-positive guards |
| `test_vad.py` | turn-end timing (700 ms), barge-in threshold (300 ms), noise blips, plus the real `webrtcvad` library |
| `test_chunking.py` | splitting the streamed reply into speakable sentences |
| `test_rag.py` | chunking, retrieval formatting, ingestion, clearing stale chunks |
| `test_llm.py` | system prompt, RAG context and history sent to Claude |
| `test_tts.py` | WAV output, and not crashing when Piper makes no audio |
| `test_limits.py` | the public-link protections: access code, people at once, questions per session and per day, hidden transcripts |
| `test_setup.py` | the startup checks for the API key and the voice file |
| `test_session.py` | the real `/ws` endpoint driven with scripted speech: normal turn, interrupt while generating / while playing, "wait" then "continue", resume, new topic after interrupt, double interrupt, reset, disconnect, late playback acks, error recovery |

The tests don't cover real microphone audio, Whisper accuracy, or how the VAD
thresholds feel in your room. Those need manual testing; tune the constants
in `backend/config.py` as you go.

## Make the page yours

`frontend/config.js` holds your name (shown top right as "By ...") and the link
to the GitHub repo (the link stays hidden until you fill it in). The wording is
in `frontend/index.html`, the look is in `frontend/style.css` (the colours are
the few variables at the top). While you talk, the captions show the sentences
you have heard (white), the one playing now (highlighted) and the ones still to
come (dimmer). If you cut the assistant off, the sentences you did not hear
disappear, so the captions only ever show what you heard. The page works out
sentences with the same rule as the server, so both agree.

The font (Bricolage Grotesque, SIL Open Font License) is kept in
`frontend/fonts/`, so the page does not load anything from Google.

## Settings (`backend/config.py`)

- `WHISPER_MODEL_SIZE`: `tiny.en` (fastest), `base.en` (default), `small.en` (more accurate)
- `SILENCE_MS_TO_END_TURN`, `VAD_AGGRESSIVENESS`: how eagerly it decides you've finished talking
- `MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN`: raise this if noise triggers false interrupts
- `CLAUDE_MODEL`, `MAX_TOKENS`: for example `claude-sonnet-5-5` for smarter (and pricier) answers

Settings can also go in `.env` (for example `WHISPER_MODEL_SIZE=tiny.en`). That works for the three voice-detection knobs too: `VAD_AGGRESSIVENESS=3`, `SILENCE_MS_TO_END_TURN=900`, `MIN_SPEECH_MS_TO_TRIGGER_BARGE_IN=500`.

## Project layout

```
backend/   main.py (websocket server + interrupt/resume), stt.py, tts.py,
           vad.py, rag.py, llm.py, intent.py, config.py, tests/
frontend/  index.html (the page), style.css, app.js (mic, playback, captions,
           the waveform), pcm-worklet.js (mic resampler), config.js (your name,
           GitHub link), fonts/ (the page font and its license)
data/knowledge/   documents the assistant can answer from
models/    Piper voice (created by the download step)
```

## Known limitations

- "Continue" detection is a keyword list, not a model. It handles common
  phrases but will miss unusual wording ("pick up where you left off").
  Swapping `intent.is_continue` for a small Claude call is the obvious upgrade.
- Talking again during the ~1 s while Whisper is transcribing your previous
  sentence cancels that transcription, so that sentence is lost.
- One full WAV is generated per sentence. For very long sentences, streaming
  raw audio chunks from Piper would lower the delay before speech starts.
- Single process and in-memory sessions. The only protection on the websocket
  is the optional shared access code, which is enough for a portfolio demo but
  is not real user authentication.
