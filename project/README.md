# DataCanvas: AI-Powered Infographic Generator


DataCanvas is a powerful, user-friendly application that turns your raw data into meaningful, story-driven infographics. 
Using cutting-edge AI and visualization tools, DataCanvas empowers users to analyze datasets, generate narratives, 
and create interactive charts effortlessly.


# Features

1. Data Upload: Upload .csv or .xlsx files for seamless data processing.
2. Narrative Generation: AI-driven text generation provides a detailed narrative summarizing key insights and trends from your data.
3. Static Visualizations:
Trend Analysis
Category Comparison
Category Breakdown (Pie Chart)

Interactive Visualizations:
Generate Scatter Plots, Bar Charts, and Line Charts.
Dynamic column selection for customized visualizations.

PDF Report Creation:
Combines the AI-generated narrative and static visualizations into a professional-quality report.

Intuitive User Interface:
Built with Streamlit for a smooth, modern, and responsive user experience.


Getting Started
Prerequisites
Python: Version 3.8 or higher.
Dependencies: Listed in requirements.txt.

Installation
Clone the repository:

git clone https://github.com/ssharma-03/datacanvas.git
cd datacanvas

Install required dependencies:
pip install -r requirements.txt

Set up environment variables:

Create a .env file in the root directory.
Add your Groq API Key:
env
GROQ_API_KEY=your_api_key_here

Launch the app:
streamlit run app.py

Usage
Upload Data:
Upload a .csv or .xlsx file via the file uploader.
Generate Narrative & Static Visualization:
Provide a description for the infographic.
Select a visualization template (Trend Analysis, Comparison, or Breakdown).
Click the "Generate Narrative & Static Report" button to create visuals and downloadable reports.
Interactive Visualizations:
Use the "Generate Interactive Visualizations" button to dynamically create Scatter, Bar, or Line charts by selecting columns from dropdowns.
Download PDF:
Download your narrative and static visualizations in a PDF format using the "Download Report" button.

Project Structure

📁 datacanvas/

├── 📂 assets/                   # Static assets like logos or templates

├── 📂 data/                     # Sample datasets for testing

├── 📂 reports/                  # PDF reports generated by the app

├── app.py                       # Main application script

├── requirements.txt             # List of dependencies

├── .env                         # Environment variables

├── README.md                    # Documentation

Technologies Used
Frontend: Streamlit

Visualization:
Static Charts: Matplotlib

Interactive Charts: Plotly

AI Integration: Groq API for narrative generation

PDF Creation: ReportLab

Backend: Python


Future Plans
Add advanced visualizations like heatmaps, boxplots, and treemaps.
Enable support for additional file formats such as .json.
Enhance AI narratives to include predictive analytics and anomaly detection.
Add multi-language support for both narratives and UI.
