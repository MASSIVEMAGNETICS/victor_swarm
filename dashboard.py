import json
import os
from flask import Flask

app = Flask(__name__)

def load_report_data(filename):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading report {filename}: {e}")
        return {"error": str(e), "filename": filename}

@app.route('/')
def home():
    bots_static_info = [
        {"name": "ProspectorVictor", "job": "Scrape platforms for real prospects"},
        {"name": "CloserVictor", "job": "Convert leads into cash"},
        {"name": "FulfillmentVictor", "job": "Deliver the product/service"},
        {"name": "GrowVictor", "job": "Generate referrals and upsells"},
        {"name": "AdVictor", "job": "Run and tune paid ads"},
        {"name": "ReinvestVictor", "job": "Automatically reinvest profits"},
        {"name": "SupportVictor", "job": "Keep buyers happy and address issues"},
        {"name": "BountyVictor", "job": "Hunt real security vulnerabilities & claim bug bounty"},
        {"name": "FreelanceVictor", "job": "Find and complete freelance gigs"},
    ]

    prospector_data = load_report_data("prospector_report.json")
    bounty_data = load_report_data("bounty_report.json")

    html_content = "<h1>Victor Swarm Dashboard</h1>"

    html_content += "<h2>Bot Activity Reports:</h2>"

    # Prospector Report
    html_content += "<h3>Prospector Victor Report</h3>"
    if prospector_data:
        if "error" in prospector_data:
            html_content += f"<p>Could not load report: {prospector_data['filename']} - {prospector_data['error']}</p>"
        else:
            html_content += f"<p>Total prospects found: {prospector_data.get('prospects_count', 'N/A')}</p>"
            if prospector_data.get('prospects'):
                html_content += "<ul>"
                for p in prospector_data['prospects'][:5]: # Display first 5 prospects
                    html_content += f"<li>{p.get('name', 'N/A')} ({p.get('email', 'N/A')}) - {p.get('company', 'N/A')}</li>"
                html_content += "</ul>"
                if len(prospector_data['prospects']) > 5:
                    html_content += f"<p>...and {len(prospector_data['prospects']) - 5} more.</p>"
            elif not prospector_data.get('prospects_count', 0) == 0 : # Added check for empty list vs count mismatch
                 html_content += "<p>Prospect list is empty or missing in report.</p>"
    else:
        html_content += "<p>No report available for ProspectorVictor yet. Run main.py to generate it.</p>"

    # Bounty Report
    html_content += "<h3>Bounty Victor Report</h3>"
    if bounty_data:
        if "error" in bounty_data:
            html_content += f"<p>Could not load report: {bounty_data['filename']} - {bounty_data['error']}</p>"
        else:
            html_content += f"<p>Sites checked: {bounty_data.get('findings_count', 'N/A')}</p>"
            if bounty_data.get('findings'):
                html_content += "<ul>"
                for f in bounty_data['findings'][:5]: # Display first 5 findings
                    status = "Found" if f.get('robots_txt_found') else "Not Found/Error"
                    error_msg = f" (Error: {f.get('error')})" if f.get('error') else ""
                    status_code_val = f.get('status_code')
                    status_code = f" (Status: {status_code_val})" if status_code_val is not None else "" # Check for None
                    html_content += f"<li>{f.get('site', 'N/A')}/robots.txt: {status}{status_code}{error_msg}</li>"
                html_content += "</ul>"
                if len(bounty_data['findings']) > 5:
                    html_content += f"<p>...and {len(bounty_data['findings']) - 5} more.</p>"
            elif not bounty_data.get('findings_count', 0) == 0: # Added check for empty list vs count mismatch
                html_content += "<p>Findings list is empty or missing in report.</p>"
    else:
        html_content += "<p>No report available for BountyVictor yet. Run main.py to generate it.</p>"

    html_content += "<hr><h2>All Bot Types (Static Info):</h2>"
    html_content += "<ul>"
    for bot_info in bots_static_info:
        html_content += f"<li><b>{bot_info['name']}</b> ({bot_info['job']})</li>"
    html_content += "</ul>"

    return html_content

if __name__ == '__main__':
    app.run(debug=True, port=5001)
