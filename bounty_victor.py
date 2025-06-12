import requests
import json # Added json import
from victor_bot import VictorBot

class BountyVictor(VictorBot):
    def __init__(self):
        super().__init__(name="BountyVictor", job="Hunt real security vulnerabilities & claim bug bounty")
        self.findings = []

    def run(self):
        super().run()
        print("Scanning for vulnerabilities and reporting for bounties...")
        target_sites = [
            "https://www.google.com",
            "https://www.wikipedia.org",
            "https://www.github.com",
            "https://www.example.com/nonexistentpage",
            "https://nonexistentdomain12345.com"
        ]

        for site_url in target_sites:
            cleaned_site_url = site_url.rstrip('/')
            robots_url = f"{cleaned_site_url}/robots.txt"

            finding_details = {
                "site": site_url,
                "robots_txt_url": robots_url,
                "robots_txt_found": False,
                "status_code": None,
                "error": None
            }
            try:
                response = requests.get(robots_url, timeout=5)
                finding_details["status_code"] = response.status_code
                if response.status_code == 200:
                    finding_details["robots_txt_found"] = True
            except requests.exceptions.Timeout:
                finding_details["error"] = "Timeout"
            except requests.exceptions.ConnectionError:
                finding_details["error"] = "ConnectionError"
            except requests.exceptions.RequestException as e:
                finding_details["error"] = str(e)
            self.findings.append(finding_details)

        print(f"Completed checks for {len(target_sites)} sites.")
        self.report()

    def report(self, result=None):
        print(f"--- {self.name} Report ---")
        print(f"Checked {len(self.findings)} sites for robots.txt.")
        for finding in self.findings:
            if finding['error']:
                print(f"Site: {finding['site']}, URL: {finding['robots_txt_url']}, Error: {finding['error']}")
            else:
                print(f"Site: {finding['site']}, URL: {finding['robots_txt_url']}, Found: {finding['robots_txt_found']}, Status: {finding['status_code']}")
        print(f"--- End of {self.name} Report ---")

        report_data = {
            "bot_name": self.name,
            "job": self.job,
            "findings_count": len(self.findings),
            "findings": self.findings
        }
        try:
            with open("bounty_report.json", "w") as f:
                json.dump(report_data, f, indent=4)
            print("BountyVictor report saved to bounty_report.json")
        except IOError as e:
            print(f"Error saving BountyVictor report: {e}")
