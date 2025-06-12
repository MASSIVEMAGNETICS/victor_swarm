class VictorBot:
    def __init__(self, name, job):
        self.name = name
        self.job = job

    def run(self):
        print(f"{self.name} running: {self.job}")

    def report(self, result):
        pass
