# Deploy to Railway (Python bot)

## Files
- bot.py — main entry
- requirements.txt — dependencies
- runtime.txt — pins Python to 3.12 (fixes pandas build issues)
- Procfile — forces Railway to run `python bot.py`

## Railway settings
1. Create a new project → Deploy from GitHub or upload this ZIP.
2. Set Variables (Environment):
   - BOT_TOKEN=...
   - (other variables if you use them) e.g. GSHEETS_SERVICE_ACCOUNT_JSON, DATA_DIR, etc.
3. Deploy.

If you previously enabled webhook, disable it (or switch to polling in code). 
