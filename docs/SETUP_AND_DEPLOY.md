# Setup and deploy: the steps only you can do

Everything else is built and tested. These steps need your own accounts.

## 1. Get a Gemini API key (free, about 2 minutes)

1. Go to https://aistudio.google.com and sign in with your Google account.
2. Click **Get API key**, then **Create API key**. If asked, pick or create a Google Cloud
   project; the default one is fine.
3. Copy the key. It's a long string that starts with `AIza…`.
4. To use it locally, create `.streamlit/secrets.toml` in the project folder (it's gitignored)
   containing:
   ```toml
   GEMINI_API_KEY = "AIza...your key..."
   ```
5. Restart `streamlit run app.py`. The **Ask** view's "No Gemini API key" warning disappears,
   and the status line under each answer shows the model name instead of "offline".

> **Model name.** The app tries `gemini-2.5-flash`, then `gemini-3.1-flash-lite`, then
> `gemini-3.6-flash` and `gemini-3.8-flash`, and uses the first one that answers. Each attempt
> has a 20-second limit. If a model is overloaded (503), rate-limited (429), unavailable (404)
> or times out, the app moves straight on to the next one. The newer 3.x Flash models were
> overloaded or timing out when this was tested, which is why they come last. To try a
> specific model first, add `GEMINI_MODEL = "..."` to the secrets. The current list is at
> https://ai.google.dev/gemini-api/docs/models.
>
> **If Ask still says "Gemini is busy"**, the warning under the answer names each model and
> why it failed. An invalid key fails at once with "Gemini rejected the API key". Otherwise
> it's usually a temporary limit on the free tier: wait a minute and ask again. The app
> always falls back to a quick data answer, so a demo never breaks.

## 2. Put the code on GitHub

1. On https://github.com/new create a repository called `matchmate`. Make it **Public**, which
   Streamlit Community Cloud's free tier needs, and **don't** add a README, .gitignore or
   license.
2. In a terminal in the project folder, run:
   ```bash
   git init -b main
   git add .
   git commit -m "MatchMate: AI second-screen companion for live cricket"
   git remote add origin https://github.com/<your-username>/matchmate.git
   git push -u origin main
   ```
   `.gitignore` already excludes the 100 MB raw data, the virtualenv and `secrets.toml`. Your
   API key will **not** be pushed. Run `git status` before committing to double-check.

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Repository: `<your-username>/matchmate`, branch `main`, main file `app.py`.
4. Optionally pick a custom URL, such as `matchmate-ipl.streamlit.app`.
5. Open **Advanced settings**:
   - **Python version:** 3.12 or 3.13.
   - **Secrets:** paste `GEMINI_API_KEY = "AIza..."`.
6. Click **Deploy**. The first build takes 2–4 minutes because it installs `requirements.txt`.
7. Open the link **on your phone** and check that the scoreboard, chart and buttons fit the
   screen.
8. Paste the link at the top of `README.md`, then commit and push again.

If the build fails, open **Manage app → Logs**. The usual causes are a Python version below
3.11 (pick 3.12+) or a missing secret. The app still runs without a secret, in offline mode.

## 4. Record the 2-minute demo

Follow `docs/DEMO_SCRIPT.md`. Use OBS, or the Xbox Game Bar (Win+Alt+R) to record the screen.
Upload to YouTube as *Unlisted* or to Loom, then paste the link into the README.

## Optional: rebuild data and retrain

The processed data and the model are committed, so this is only needed if you want newer
matches after Cricsheet updates:
```bash
pip install -r requirements-train.txt
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train_model.py
pytest -q
```
