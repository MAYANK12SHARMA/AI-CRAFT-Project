import tempfile
import time
import os
import sweetviz as sv
import numpy as np

import base64

if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = DeprecationWarning


def generate_sweetviz_report(df):
    """
    Generates a Sweetviz report from the given DataFrame using a temporary file
    to avoid file locking issues. Returns a dictionary with key "html" containing
    the report's HTML content.
    """

    # Create a temporary file; delete=False allows us to re-open it later
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        temp_file_name = tmp.name

    # Generate the Sweetviz report and save it to the temporary file
    report = sv.analyze(df)
    report.show_html(temp_file_name, open_browser=False)

    # Wait briefly to ensure the file is fully written and released
    time.sleep(1)

    # Retry reading the file in case it's still locked
    retries = 5
    for attempt in range(retries):
        try:
            with open(temp_file_name, "r", encoding="utf-8") as f:
                html_content = f.read()
            break  # Successfully read the file
        except PermissionError:
            time.sleep(1)
    else:
        raise Exception("Failed to read the report file after multiple attempts.")

    # Clean up the temporary file
    os.remove(temp_file_name)
    return {"html": html_content}


# Inline helper function to generate a download link from the HTML content
def get_download_link_sweet(content, filename, ext):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/{ext};base64,{b64}" download="{filename}.{ext}">Download Report</a>'

