from victor_bot import VictorBot

class FulfillmentVictor(VictorBot):
    def __init__(self):
        super().__init__(name="FulfillmentVictor", job="Deliver the product/service")

    def run(self):
        super().run()
        print("Fulfilling orders and delivering value...")
