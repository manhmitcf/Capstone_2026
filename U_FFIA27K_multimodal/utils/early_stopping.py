import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping monitor to halt training when monitored performance metric stops improving.
    """
    def __init__(self, patience: int = 40, delta: float = 0.0, verbose: bool = True) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.info("Early Stopping state has been reset.")

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered. Halting training.")
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False

    def is_triggered(self) -> bool:
        return self.early_stop
