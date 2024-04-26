from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

logger.info('test')