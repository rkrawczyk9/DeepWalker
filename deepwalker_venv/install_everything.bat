pip install -r requirements.txt
PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""R:\Code\DeepWalker\deepwalker_venv\install_choco.ps1""' -Verb RunAs}"
choco install ffmpeg