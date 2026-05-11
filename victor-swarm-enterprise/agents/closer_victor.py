import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class CloserVictor:
    """
    CloserVictor Agent.

    Responsible for closing deals and converting prospects into clients.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CloserVictor agent.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.state = "IDLE"
        self.deals_closed = 0
        logger.info("CloserVictor initialized.")

    def run(self) -> None:
        """
        Main execution loop for the agent.
        """
        try:
            self.state = "RUNNING"
            logger.info("CloserVictor started running.")
            # Simulation of closing deals
            self.deals_closed += 1
            logger.info("CloserVictor completed a cycle.")
            self.state = "IDLE"
        except Exception as e:
            self.state = "ERROR"
            logger.error(f"CloserVictor encountered an error: {e}", exc_info=True)

    def report_status(self) -> Dict[str, Any]:
        """
        Reports the current state and metrics of the agent.

        Returns:
            A dictionary containing the agent's status.
        """
        return {
            "agent_type": "CloserVictor",
            "state": self.state,
            "deals_closed": self.deals_closed,
            "config": self.config
        }
