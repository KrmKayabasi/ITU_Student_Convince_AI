#!/usr/bin/env bash

set -euo pipefail

CONFIG_FILE="backend/orchestrator/config.py"
COMPOSE_FILE="docker-compose.yml"
README_FILE="backend/orchestrator/README.md"
ENV_FILE=".env"

echo "Disabling Gemini affective dialogue and proactive audio..."

# 1. Change Python defaults from True to False.
sed -i \
  's/_bool("GEMINI_AFFECTIVE_DIALOG", True)/_bool("GEMINI_AFFECTIVE_DIALOG", False)/g' \
  "$CONFIG_FILE"

sed -i \
  's/_bool("GEMINI_PROACTIVE_AUDIO", True)/_bool("GEMINI_PROACTIVE_AUDIO", False)/g' \
  "$CONFIG_FILE"

# 2. Add the variables to docker-compose.yml if they are missing.
python3 <<'PY'
from pathlib import Path

path = Path("docker-compose.yml")
text = path.read_text(encoding="utf-8")

affective_line = (
    "      - GEMINI_AFFECTIVE_DIALOG="
    "${GEMINI_AFFECTIVE_DIALOG:-false}"
)
proactive_line = (
    "      - GEMINI_PROACTIVE_AUDIO="
    "${GEMINI_PROACTIVE_AUDIO:-false}"
)

if "GEMINI_AFFECTIVE_DIALOG=" not in text:
    target = (
        "      - GEMINI_API_VERSION="
        "${GEMINI_API_VERSION:-v1alpha}"
    )

    if target not in text:
        raise RuntimeError(
            "Could not locate GEMINI_API_VERSION in docker-compose.yml"
        )

    replacement = (
        target
        + "\n"
        + affective_line
        + "\n"
        + proactive_line
    )

    text = text.replace(target, replacement, 1)

else:
    import re

    text = re.sub(
        r"^(\s*-\s*GEMINI_AFFECTIVE_DIALOG=).*$",
        r"\1${GEMINI_AFFECTIVE_DIALOG:-false}",
        text,
        flags=re.MULTILINE,
    )

    if "GEMINI_PROACTIVE_AUDIO=" not in text:
        affective_pattern = (
            r"^(\s*-\s*GEMINI_AFFECTIVE_DIALOG=.*)$"
        )

        text = re.sub(
            affective_pattern,
            r"\1\n"
            + proactive_line,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(
            r"^(\s*-\s*GEMINI_PROACTIVE_AUDIO=).*$",
            r"\1${GEMINI_PROACTIVE_AUDIO:-false}",
            text,
            flags=re.MULTILINE,
        )

path.write_text(text, encoding="utf-8")
PY

# 3. Create or update .env values.
touch "$ENV_FILE"

python3 <<'PY'
from pathlib import Path

path = Path(".env")
lines = path.read_text(encoding="utf-8").splitlines()

wanted = {
    "GEMINI_AFFECTIVE_DIALOG": "false",
    "GEMINI_PROACTIVE_AUDIO": "false",
}

found = set()
result = []

for line in lines:
    stripped = line.strip()

    replaced = False
    for key, value in wanted.items():
        if stripped.startswith(f"{key}="):
            result.append(f"{key}={value}")
            found.add(key)
            replaced = True
            break

    if not replaced:
        result.append(line)

for key, value in wanted.items():
    if key not in found:
        result.append(f"{key}={value}")

path.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")
PY

# 4. Update the documented default when present.
if [[ -f "$README_FILE" ]]; then
    sed -i \
      's/| `GEMINI_AFFECTIVE_DIALOG` \/ `GEMINI_PROACTIVE_AUDIO` | `true`/| `GEMINI_AFFECTIVE_DIALOG` \/ `GEMINI_PROACTIVE_AUDIO` | `false`/g' \
      "$README_FILE"
fi

echo
echo "Changes applied."
echo
echo "Resolved Compose configuration:"
docker compose config | grep -E \
  'GEMINI_(LIVE_MODEL|API_VERSION|AFFECTIVE_DIALOG|PROACTIVE_AUDIO)' \
  || true

echo
echo "Relevant Python defaults:"
grep -nE \
  'ENABLE_AFFECTIVE_DIALOG|ENABLE_PROACTIVE_AUDIO' \
  "$CONFIG_FILE"
