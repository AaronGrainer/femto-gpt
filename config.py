import logging
import logging.config
import sys
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
DATASET_DIR = Path(BASE_DIR, "dataset")
CHECKPOINT_DIR = Path(BASE_DIR, "checkpoint")

# Create Dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Config
BATCH_SIZE = 64
BLOCK_SIZE = 256
CHANNEL_SIZE = 32
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
HEAD_SIZE = N_EMBED // N_HEAD
DROPOUT = 0.2
TRAIN_TEST_SPLIT = 0.9
EPOCHS = 10
EVAL_ITER = 200
EVAL_INTERVAL = 500
LR = 3e-4
SEED = 8383


# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        }
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
# logger.handlers[0] = RichHandler(markup=True)
