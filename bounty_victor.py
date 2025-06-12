from victor_bot import VictorBot

class BountyVictor(VictorBot):
    def __init__(self):
        super().__init__(name="BountyVictor", job="Hunt real security vulnerabilities & claim bug bounty")

    def run(self):
        super().run()
        print("Scanning for vulnerabilities and reporting for bounties...")
