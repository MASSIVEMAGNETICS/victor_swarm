import requests
import json # Added json import
from victor_bot import VictorBot

class ProspectorVictor(VictorBot):
    def __init__(self):
        super().__init__(name="ProspectorVictor", job="Scrape platforms for real prospects")
        self.prospects = []

    def run(self):
        super().run()
        print("Scanning sources for leads...")
        url = "https://jsonplaceholder.typicode.com/users"
        try:
            response = requests.get(url)
            response.raise_for_status()
            users = response.json()
            for user in users:
                prospect = {
                    "name": user.get("name"),
                    "email": user.get("email"),
                    "company": user.get("company", {}).get("name"),
                    "source": url
                }
                self.prospects.append(prospect)
            print(f"Successfully fetched {len(self.prospects)} prospects from {url}.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching prospects from {url}: {e}")

        self.report()

    def report(self, result=None): # Added result=None to match base class
        print(f"{self.name} reporting: Found {len(self.prospects)} prospects.")
        report_data = {
            "bot_name": self.name,
            "job": self.job,
            "prospects_count": len(self.prospects),
            "prospects": self.prospects # Storing all prospects
        }
        try:
            with open("prospector_report.json", "w") as f:
                json.dump(report_data, f, indent=4)
            print("ProspectorVictor report saved to prospector_report.json")
        except IOError as e:
            print(f"Error saving ProspectorVictor report: {e}")
