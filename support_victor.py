from victor_bot import VictorBot

class SupportVictor(VictorBot):
    def __init__(self):
        super().__init__(name="SupportVictor", job="Keep buyers happy and address issues")

    def run(self):
        super().run()
        print("Providing customer support and ensuring satisfaction...")
