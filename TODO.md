# Stock Prediction Deployment TODO - Making Vercel-Ready (Not Netlify)

**Status:** Approved plan - Proceed with edits, local test, Vercel deploy.

## Breakdown Steps:

### 1. Update app.py for Production (Priority 1)
- [ ] Replace `if __name__ == "__main__": app.run(debug=True, port=5000)`
  with: `port = int(os.environ.get('PORT', 5000))`
  `debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'`
  `app.run(host='0.0.0.0', port=port, debug=debug)`
- [ ] Add `static_folder='static'` to Flask() if static assets exist.
- [ ] Ensure no training on cold starts (TRAIN_ON_DEMAND=false already).

### 2. Verify/Enhance Config Files
- [ ] requirements.txt: Add python-dotenv (local), confirm torch CPU.
- [ ] vercel.json: Headers for static/cache if needed.
- [ ] runtime.txt: python-3.10.12 ✓

### 3. Update Documentation
- [ ] README.md: Add gunicorn local test command, Vercel deploy steps.

### 4. Local Testing
- [ ] `pip install -r requirements.txt`
- [ ] `gunicorn --bind 0.0.0.0:$PORT stock-prediction.app:app -w 2`
- [ ] Test /run endpoint (no training if models exist).
- [ ] `flask --app stock-prediction.app run --host=0.0.0.0 --port=3000`

### 5. Deploy to Vercel
- [ ] `npm i -g vercel` (if not installed)
- [ ] `vercel login`
- [ ] `vercel --prod`
- [ ] Set env vars in Vercel dashboard: FLASK_DEBUG=false, TRAIN_ON_DEMAND=false

### 6. Post-Deploy
- [ ] Test live predictions.
- [ ] Monitor logs/functions duration.

**Current Progress:** 60% (configs done). Next: Edit app.py.

**Next Action:** Implement Step 1 - app.py production config.

