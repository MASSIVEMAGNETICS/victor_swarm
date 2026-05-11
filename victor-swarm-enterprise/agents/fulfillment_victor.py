import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class FulfillmentVictor:
    """
    FulfillmentVictor Agent.

    Responsible for delivering products or services to clients after a deal is closed.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the FulfillmentVictor agent.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.state = "IDLE"
        self.orders_fulfilled = 0
        logger.info("FulfillmentVictor initialized.")

    def run(self) -> None:
        """
        Main execution loop for the agent.
        """
        try:
            self.state = "RUNNING"
            logger.info("FulfillmentVictor started running.")
            # Simulation of fulfilling orders
            self.orders_fulfilled += 1
            logger.info("FulfillmentVictor completed a cycle.")
            self.state = "IDLE"
        except Exception as e:
            self.state = "ERROR"
            logger.error(f"FulfillmentVictor encountered an error: {e}", exc_info=True)

    def report_status(self) -> Dict[str, Any]:
        """
        Reports the current state and metrics of the agent.

        Returns:
            A dictionary containing the agent's status.
        """
        return {
            "agent_type": "FulfillmentVictor",
            "state": self.state,
            "orders_fulfilled": self.orders_fulfilled,
            "config": self.config
        }
