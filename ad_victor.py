from victor_bot import VictorBot

class AdVictor(VictorBot):
    def __init__(self):
        super().__init__(name="AdVictor", job="Run and tune paid ads")

    def run(self):
        super().run()
        print("Managing ad campaigns across platforms...")
