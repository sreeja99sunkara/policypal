# Deploying PolicyPal as a Hugging Face Docker Space

This guide puts the PolicyPal demo on a Hugging Face Space: one fixed policy,
uploads disabled, and rate limits on. You need a Hugging Face account, a Groq
API key, Git, and Git LFS.

Each step is labelled with how far it was checked:

- **[verified locally]** was run on the development machine with Docker
  Desktop, using the same bundle and image the Space builds.
- **[unverified]** describes the Hugging Face website or service, which was
  not used while writing this. Menu names may differ slightly.

---

## 0. Build the bundle  [verified locally]

From the repo root:

```bash
python scripts/make_space_bundle.py
```

This writes `dist/hf-space/`, which is git-ignored, and replaces it on every
run. The bundle holds only what the Space needs:

- `app/`, `streamlit_app.py` and `docker/` (the start script and the
  requirements lists);
- `requirements.txt`, re-encoded as UTF-8;
- the one policy PDF;
- a generated `Dockerfile` (the `demo` stage last, models baked in);
- a `README.md` with the Space's YAML header;
- `.gitattributes`.

The script fails if `.env`, `eval/`, `tests/`, `notebooks/`, `venv/`, `.git/`,
another PDF or a `gsk_…` string ends up in the bundle.

Optional local check, which builds and runs the bundle the same way the Space
does (no build args, no volumes):

```bash
cd dist/hf-space
docker build -t policypal-hfspace .
docker run --rm -p 7860:7860 --env-file ../../.env \
  -e DEMO_MODE=true -e DEFAULT_POLICY_PDF=/app/demo/policy.pdf policypal-hfspace
# open http://localhost:7860 and wait for "Backend connected"
```

## a. Create the Space  [unverified]

1. Go to https://huggingface.co/new-space.
2. Fill in the form:
   - **Owner:** your account.
   - **Space name:** for example `policypal`.
   - **License:** your choice.
   - **Visibility:** Public.
3. **Select the Space SDK:** Docker, then the **Blank** template.
4. **Space hardware:** CPU basic (2 vCPU, 16 GB RAM, free).
5. Click **Create Space**.

To check that free hardware is available on your account:

- The hardware list on the new-space form shows which tiers are free; CPU
  basic should be marked free.
- After creating the Space, open **Settings → Space hardware**; it shows the
  current tier and any cost.
- Your account's billing page, https://huggingface.co/settings/billing, shows
  whether a payment method is needed. CPU basic should not need one. If the
  form asks for payment for CPU basic, stop and check before continuing.

## b. Secret and variables  [unverified]

In the Space: **Settings → Variables and secrets**.

1. **New secret:** name `GROQ_API_KEY`, value your Groq key.
   - It must be a **secret**, not a variable. Secrets are hidden after saving
     and are not copied when someone duplicates the Space. Variables can be
     read by anyone who can see the Space.
   - The container receives it as an environment variable at run time; it is
     never in the image or the repository.
2. **New variable** for each of these:

   | Name | Value |
   |---|---|
   | `DEMO_MODE` | `true` |
   | `DEFAULT_POLICY_PDF` | `/app/demo/policy.pdf` |
   | `GROQ_MODEL` | `openai/gpt-oss-120b` |
   | `DEMO_DAILY_QUERY_CAP` | `150` (or lower to protect your Groq quota) |

   The image already defaults `DEMO_MODE` and `DEFAULT_POLICY_PDF` to these
   values; setting them on the Space makes them visible and easy to change.

   Optional variables:
   - `RATE_LIMIT_PER_MINUTE` (default 10) and `RATE_LIMIT_PER_DAY` (default
     100), both per visitor.
   - `MAX_CONCURRENT_QUERIES` (default 1).
   - `QUERY_WAIT_SECONDS` (default 30).

   Do **not** set `PORT`: the Dockerfile sets it to 7860, which matches
   `app_port` in the README header.

Changing a secret or variable restarts the Space.

Also set a spending limit on the Groq account (console.groq.com), since the
daily cap only bounds what this one process answers.

## c. Access token and push  [unverified]

**Create a token.** Open https://huggingface.co/settings/tokens and choose
**Create new token**. Prefer a **fine-grained** token that has *write* access
to this one Space only (under Repositories, add the Space and tick write).
Otherwise use a **Write** token. Copy the token once and store it in your
password manager, not in a file.

**Push the bundle.** These commands use Git Bash on Windows. Replace `<user>`
and `<space>`. Run them outside the PolicyPal repo, for example in
`C:\Users\<you>\Downloads`.

```bash
git lfs install                                   # once per machine
git clone https://huggingface.co/spaces/<user>/<space> hf-space-repo
# copy the bundle over the clone (this replaces its README.md and .gitattributes)
cp -r policypal/dist/hf-space/. hf-space-repo/
cd hf-space-repo
git add -A
git status                                        # check: no .env, eval/, tests/
git commit -m "Deploy PolicyPal demo"
git -c credential.helper= push
```

- The clone is public and needs no token.
- `-c credential.helper=` turns off Git's credential store (Git Credential
  Manager on Windows) for this one push. Git then asks in the terminal:
  - **Username:** your Hugging Face username.
  - **Password:** paste the token. It is not shown on screen.

  The token is not written to a file, to the remote URL, or to your shell
  history. Git LFS uploads the PDF in the same push and may ask a second time.
- **Do not:**
  - put the token in the remote URL (`https://user:token@…`): Git saves it in
    `.git/config`;
  - run `export HF_TOKEN=…` on the command line: it lands in shell history;
  - run `hf auth login`: it saves the token to a file.
- If the push is rejected with a message about binary files, check
  `git lfs ls-files`; it should list the PDF. Hugging Face now stores large
  files with Xet. Its documentation says LFS pushes still work; if not,
  install `git-xet` as the error message suggests. (unverified)
- **To update the Space later:**
  1. Re-run `python scripts/make_space_bundle.py`.
  2. Delete everything in `hf-space-repo` except `.git/`, so files removed
     from the bundle don't linger.
  3. Copy the bundle in again, then commit and push as above.

## d. Build and run logs  [unverified on Hugging Face; log lines verified locally]

On the Space page, the status badge next to the name shows **Building**, then
**Running**, or an error. Click it, or **Logs**, to open the log viewer. It has
**Build** and **Container** tabs.

**Build.** Locally the build took 25–33 minutes with a warm pip cache, of which 24–27 minutes was downloading the models on a slow connection; the image is 5.4 GB. On
Hugging Face it starts from nothing, so expect longer: it downloads about
1 GB of Python wheels and bakes in about 1.2 GB of models. The build log
should end with the step that prints `models cached`, then image export.

**Healthy start.** The Container log shows these lines, in this order and
measured locally on the bundle image:

```
You can now view your Streamlit app in your browser.     <- UI is up, backend still loading
INFO:     Waiting for application startup.
... "message": "Default policy indexed", "pdf": "/app/demo/policy.pdf", "chunk_count": 486, "seconds": 43.1 ...
All services ready.
INFO:     Application startup complete.
```

Until `All services ready.` the page shows "The backend is starting …
Refresh in a minute." Locally the UI was usable 76–80 s after the
container started.

**If something fails:**

| Symptom | Likely cause and fix |
|---|---|
| Build fails while downloading wheels or models | Network hiccup: **Settings → Factory rebuild**. If it repeats, read the first error in the Build log. |
| Build fails with `--mount` or `--chmod` errors | The builder lacks BuildKit features; report it, since the Dockerfile needs `# syntax=docker/dockerfile:1`. (unverified) |
| Status stays **Starting**, page never loads | Port mismatch: `app_port: 7860` in README.md must match `PORT` (do not override `PORT`). |
| "The backend is starting" for more than 5 minutes | Container log: look for a Python traceback after `Waiting for application startup.` |
| Log shows `DEFAULT_POLICY_PDF not found` | Variable path is wrong; use `/app/demo/policy.pdf`. |
| Indexing fails or reports 0 chunks, or the PDF is ~130 bytes | The PDF was pushed as an LFS pointer, not the file. Check `git lfs ls-files` and push again. |
| Questions fail with "Failed to process your question" | Missing or wrong `GROQ_API_KEY` secret, or Groq quota exhausted. Container log shows the Groq error. |
| Space shows **Runtime error** | Open the Container log. **Restart Space** to retry; **Factory rebuild** rebuilds the image. |

## e. Sleep, wake and pause  [unverified on Hugging Face]

- **Sleep:** Spaces on free CPU basic hardware sleep after a period without
  visitors; Hugging Face currently documents 48 hours. A visit wakes the Space.
- **Wake time:** a wake is a cold start. The models are baked into the image,
  so nothing is downloaded, but the policy is re-indexed because the disk is
  not persistent. Locally:
  - cold start to a usable UI: 76–80 s;
  - of which indexing: about 43 s.

  Hugging Face's own scheduling adds to this, so expect roughly 2–3 minutes
  before the first answer.
- **Counters reset on wake or restart:** the rate-limit counters and the daily
  cap are in memory, so they start again from zero.
- **Pause:** **Settings → Pause Space** stops it completely; visitors see a
  paused page and it does not wake on visits. Click **Restart** there to bring
  it back. Pause it if the Groq key is being used more than you want, or while
  you are not showing it.

## f. Post-deploy checklist

- [ ] The Space status is **Running** and the Container log contains
      `All services ready.` and `"chunk_count": 486`.
- [ ] **Phone:** open the Space URL on a phone. The page loads, the sidebar
      (the `>` button) shows "Backend connected" and the demo note, and there
      is no upload box. (Not checked locally on a phone.)
- [ ] **Question 1:** "Is there a co-payment under this policy?" The answer
      mentions 10% and cites pages, and under **Sources** at least one item
      is marked "cited in answer".
- [ ] **Question 2:** "What is the room rent limit per day?" The answer cites
      page 6. Open a source and press **Show page**; the page image renders.
- [ ] **Upload refused:** the sidebar has no uploader. The API itself is not
      reachable from outside the Space (it listens on 127.0.0.1), so there is
      no public upload endpoint to try.
- [ ] **Rate-limit message:** ask 11 short questions within a minute. The
      11th should show "Too many questions: the limit is 10 per minute. Please
      wait N seconds and try again." This uses about 10 Groq calls. If everyone
      shares one limit (another phone or network is limited at the same time),
      the Space's proxy hides visitors' addresses; see "Rate limiting" in the
      main README. (unverified on Hugging Face)
- [ ] **Quota behaviour:** set the variable `DEMO_DAILY_QUERY_CAP` to `2` (the
      Space restarts), then ask three questions. The third should say "This
      demo has used its 2 questions for today. It resets at 00:00 UTC, in
      about N hours." Set it back to `150` afterwards.
- [ ] **Busy behaviour (optional):** ask from two devices at the same moment.
      The second waits for the first, up to 30 s, rather than failing.
- [ ] Add the Space URL to the "Live demo" section of the main README.
