from prospector_victor import ProspectorVictor
from closer_victor import CloserVictor
from fulfillment_victor import FulfillmentVictor
from grow_victor import GrowVictor
from ad_victor import AdVictor
from reinvest_victor import ReinvestVictor
from support_victor import SupportVictor
from bounty_victor import BountyVictor
from freelance_victor import FreelanceVictor

all_bots = [
    ProspectorVictor(),
    CloserVictor(),
    FulfillmentVictor(),
    GrowVictor(),
    AdVictor(),
    ReinvestVictor(),
    SupportVictor(),
    BountyVictor(),
    FreelanceVictor(),
]

if __name__ == "__main__":
    print("Starting the Victor Swarm...")
    for bot in all_bots:
        bot.run()
        print("-" * 20) # Separator for readability
    print("All Victor bots have completed their current cycle.")
