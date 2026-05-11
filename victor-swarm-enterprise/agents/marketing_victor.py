import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class MarketingVictor:
    """
    MarketingVictor Agent.

    Responsible for managing marketing campaigns, generating brand awareness, and finding general audiences.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the MarketingVictor agent.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.state = "IDLE"
        self.campaigns_run = 0
        logger.info("MarketingVictor initialized.")

    def run(self) -> None:
        """
        Main execution loop for the agent.
        """
        try:
            self.state = "RUNNING"
            logger.info("MarketingVictor started running.")
            # Simulation of running marketing campaigns
            self.campaigns_run += 1
            logger.info("MarketingVictor completed a cycle.")
            self.state = "IDLE"
        except Exception as e:
            self.state = "ERROR"
            logger.error(f"MarketingVictor encountered an error: {e}", exc_info=True)

    def report_status(self) -> Dict[str, Any]:
        """
        Reports the current state and metrics of the agent.

        Returns:
            A dictionary containing the agent's status.
        """
        return {
            "agent_type": "MarketingVictor",
            "state": self.state,
            "campaigns_run": self.campaigns_run,
            "config": self.config
        }
