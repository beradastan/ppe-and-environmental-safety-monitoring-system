from __future__ import annotations

import logging

from flask import Flask
from flask_cors import CORS

from backend.config_manager import CORS_ORIGINS, HOST, PORT, RESULTS_DIR, USE_DB
from backend.extensions import socketio
from backend.routes import events, pipeline, reports, settings
from backend.watcher import ResultsWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)
socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

app.register_blueprint(events.bp)
app.register_blueprint(reports.bp)
app.register_blueprint(pipeline.bp)
app.register_blueprint(settings.bp)

@socketio.on("connect")
def on_connect():
    logger.info("İstemci bağlandı.")

@socketio.on("disconnect")
def on_disconnect():
    logger.info("İstemci ayrıldı.")

def main() -> None:
    watcher = ResultsWatcher(RESULTS_DIR, socketio)
    watcher.start()

    if USE_DB:
        reports.start_report_scheduler()

    logger.info("Backend başlıyor → http://%s:%s", HOST, PORT)
    logger.info("Results dizini: %s", RESULTS_DIR)

    try:
        socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        watcher.stop()

if __name__ == "__main__":
    main()
