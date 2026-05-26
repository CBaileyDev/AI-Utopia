# Secrets

Do NOT commit secret contents to git. The `.gitignore` includes `secrets/`
except for this README and `.gitkeep`.

Required files (create locally before `docker compose up`):

- `anthropic_key` — single-line Anthropic API key (`sk-ant-...`)

Permissions:
```bash
chmod 600 secrets/anthropic_key
```
