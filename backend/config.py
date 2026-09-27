"""Configuration loader."""
import os
from pathlib import Path

# Load .env manually (no python-dotenv dep)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
