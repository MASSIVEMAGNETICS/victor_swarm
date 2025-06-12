from victor_bot import VictorBot

class ProspectorVictor(VictorBot):
    def __init__(self):
        super().__init__(name="ProspectorVictor", job="Scrape platforms for real prospects")

    def run(self):
        super().run()
        print("Scanning sources for leads...")
