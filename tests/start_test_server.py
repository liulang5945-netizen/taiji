"""????????????"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ????
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    os.environ["MODEL_LOAD_TIMEOUT"] = "10"  # ????????
    from api.app import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
