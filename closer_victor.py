from victor_bot import VictorBot

class CloserVictor(VictorBot):
    def __init__(self):
        super().__init__(name="CloserVictor", job="Convert leads into cash")

    def run(self):
        super().run()
        print("Engaging leads and closing deals...")
