# HF Space Deployment — Step-by-Step

Judges test from a logged-out browser. Follow exactly.

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Owner: your HF username
3. Space name: `the-pivot` (or similar)
4. License: MIT
5. **SDK: Docker** (important — not Gradio/Streamlit)
6. **Space hardware: CPU basic** (free, fine for the env)
7. **Visibility: Public** (critical — private spaces fail validation)
8. Click **Create Space**

## 2. Push your repo contents into the Space

```bash
# On your local machine, in the meta_scaler folder:
git clone https://huggingface.co/spaces/<YOUR_HF_USERNAME>/the-pivot hf_deploy
cd hf_deploy

# Copy project files into the space repo
cp -r ../models.py ../scenarios ../server ../static ../openenv.yaml .
cp ../hf_space/Dockerfile .
cp ../hf_space/README.md .

git add .
git commit -m "Initial Pivot OpenEnv deployment"
git push
```

## 3. Watch the build

- Go to your Space page
- Click the **"Logs"** tab
- First build takes ~2–3 minutes
- When it says "Running", open the Space URL

## 4. Verify from a logged-out browser (CRITICAL)

1. Open an incognito / private window
2. Paste your Space URL (e.g. `https://huggingface.co/spaces/<USER>/the-pivot`)
3. You should see the dashboard
4. Try `/ui`, `/docs`, `/compare?scenario=b2c_saas&n_episodes=5`
5. If any of these 404 or require login — your submission fails. Fix before submitting.

## 5. Paste the Space URL into README.md

Replace the `<HF_SPACE_URL>` placeholder at the top of the main README.

## Common gotchas

- **Port 7860**: HF Spaces mandate this. Dockerfile already handles it.
- **Memory limit**: CPU basic = 16GB RAM. The env uses <500MB — plenty.
- **Stuck on "Building"**: Check logs for pip errors. Usually version mismatches.
- **Works locally but 404s on Space**: Make sure `server.app:app` is the right import path and that the working dir inside the container is `/app`.
