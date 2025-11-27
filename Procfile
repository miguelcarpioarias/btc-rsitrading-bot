release: python init_db.py
web: gunicorn app:server --workers 2 --worker-class sync --timeout 120 --bind 0.0.0.0:$PORT
worker: python -c "from app import scheduler; import time; [time.sleep(1) for _ in iter(int, 1)]"
