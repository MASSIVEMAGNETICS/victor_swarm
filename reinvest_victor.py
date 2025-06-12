from victor_bot import VictorBot

class ReinvestVictor(VictorBot):
    def __init__(self):
        super().__init__(name="ReinvestVictor", job="Automatically reinvest profits")

    def run(self):
        super().run()
        print("Analyzing profits and reinvesting into performing channels...")
