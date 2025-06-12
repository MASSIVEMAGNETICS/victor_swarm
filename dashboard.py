from flask import Flask
from prospector_victor import ProspectorVictor
from closer_victor import CloserVictor
from fulfillment_victor import FulfillmentVictor
from grow_victor import GrowVictor
from ad_victor import AdVictor
from reinvest_victor import ReinvestVictor
from support_victor import SupportVictor
from bounty_victor import BountyVictor
from freelance_victor import FreelanceVictor

app = Flask(__name__)

# Note: We are not instantiating the bot classes here,
# but using a dictionary for simplicity in this dashboard example.
# A more advanced dashboard might interact with live bot instances.
bots_data = [
    {"name": "ProspectorVictor", "job": "Scrape platforms for real prospects", "status": "active"},
    {"name": "CloserVictor", "job": "Convert leads into cash", "status": "active"},
    {"name": "FulfillmentVictor", "job": "Deliver the product/service", "status": "active"},
    {"name": "GrowVictor", "job": "Generate referrals and upsells", "status": "active"},
    {"name": "AdVictor", "job": "Run and tune paid ads", "status": "active"},
    {"name": "ReinvestVictor", "job": "Automatically reinvest profits", "status": "active"},
    {"name": "SupportVictor", "job": "Keep buyers happy and address issues", "status": "active"},
    {"name": "BountyVictor", "job": "Hunt real security vulnerabilities & claim bug bounty", "status": "active"},
    {"name": "FreelanceVictor", "job": "Find and complete freelance gigs", "status": "active"},
]

@app.route('/')
def home():
    html_content = "<h1>Victor Swarm Dashboard</h1>"
    html_content += "<h2>Bots Status:</h2>"
    html_content += "<ul>"
    for bot in bots_data:
        html_content += f"<li><b>{bot['name']}</b> ({bot['job']}): {bot['status']}</li>"
    html_content += "</ul>"
    return html_content

if __name__ == '__main__':
    app.run(debug=True, port=5001)
