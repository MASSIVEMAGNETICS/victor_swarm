from victor_bot import VictorBot

class GrowVictor(VictorBot):
    def __init__(self):
        super().__init__(name="GrowVictor", job="Generate referrals and upsells")

    def run(self):
        super().run()
        print("Identifying happy clients for growth opportunities...")
