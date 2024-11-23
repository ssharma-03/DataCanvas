################################################# Initial Main Code ######################################################

# import os
# import streamlit as st
# import pandas as pd

# from scripts.visualization import create_bar_chart
# from scripts.ai_insights import generate_insights


# st.title("DataCanvas: AI-Powered Infographic Generator")

# uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.write("Preview of your data:", data.head())
    
#     x_column = st.selectbox("Select X-axis column", data.columns)
#     y_column = st.selectbox("Select Y-axis column", data.columns)
    
#     if st.button("Generate Chart"):
#         create_bar_chart(data, x_column, y_column, "chart.png")
#         st.image("chart.png", caption="Generated Chart")

#     if st.button("Generate Insights"):
#         insights = generate_insights(data.to_string())
#         st.text_area("AI Insights", insights)


################################################# with user insignts ######################################################


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from groq import Groq
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Load model and tokenizer globally to avoid reloading
# try:
#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#     st.success("Successfully connected to Groq API")
# except Exception as e:
#     client = None
#     st.error(f"Error connecting to Groq API: {e}")
# # Backend functions
# def process_data(file):
#     """Clean and summarize uploaded data."""
#     try:
#         df = pd.read_csv(file)
#     except Exception:
#         df = pd.read_excel(file)

#     # Example: Summarizing data trends
#     summary = {
#         "columns": list(df.columns),
#         "row_count": len(df),
#         "key_stats": df.describe(include="all").to_dict()
#     }
#     return df, summary

# def generate_narrative(description, data_summary):
#     """Generate narrative for infographic using Groq's Mixtral model."""
#     if client is None:
#         return "Error: Could not connect to Groq API. Check your API key."

#     try:
#         prompt = (
#             f"Create a clear and concise narrative for an infographic with the following description:\n"
#             f"'{description}'\n"
#             f"Data summary:\n{data_summary}\n\n"
#             "Please provide a structured analysis that highlights key insights and trends."
#         )
        
#         completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             model="mixtral-8x7b-32768",
#             temperature=0.7,
#             max_tokens=500,
#         )
        
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error generating narrative: {str(e)}"


# def generate_visualization(df, template_type):
#     """Generate visualizations based on template type."""
#     try:
#         if template_type == "Trend Analysis":
#             df.plot(x=df.columns[0], y=df.columns[1:], kind="line")
#             plt.title("Trend Analysis")
#             plt.xlabel("Time")
#             plt.ylabel("Values")
#         elif template_type == "Comparison":
#             df.plot(kind="bar")
#             plt.title("Category Comparison")
#             plt.xlabel("Categories")
#             plt.ylabel("Values")
#         elif template_type == "Breakdown":
#             df.iloc[0].plot(kind="pie", autopct="%1.1f%%")
#             plt.title("Category Breakdown")
#         plt.tight_layout()
#         plt.savefig("output_visual.png")
#         plt.close()
#         return "output_visual.png"
#     except Exception as e:
#         return f"Error generating visualization: {str(e)}"

# # Streamlit App
# st.title(" DataCanvas: AI-Powered Infographic Generator")
# st.markdown("### Transform your data into story-driven infographics.")

# # Upload Data
# uploaded_file = st.file_uploader("Upload your data file (CSV/Excel):")
# description = st.text_input("Describe the infographic you need:")
# template_type = st.selectbox("Choose a visualization template:", ["Trend Analysis", "Comparison", "Breakdown"])

# # Placeholders for interactive buttons and outputs
# narrative_placeholder = st.empty()
# visual_placeholder = st.empty()

# if uploaded_file:
#     with st.spinner("Processing your data..."):
#         df, summary = process_data(uploaded_file)
#         st.success("Data processed successfully!")
#         st.write("### Data Preview")
#         st.dataframe(df.head())

#         # Generate narrative
#         if st.button("Generate Narrative"):
#             with st.spinner("Generating narrative..."):
#                 narrative = generate_narrative(description, summary)
#                 narrative_placeholder.write("### Narrative")
#                 narrative_placeholder.write(narrative)

#         # Generate visualization
#         if st.button("Generate Visualization"):
#             with st.spinner("Creating visualization..."):
#                 visualization_result = generate_visualization(df, template_type)
#                 if visualization_result.endswith(".png"):
#                     visual_placeholder.image(visualization_result, caption="Generated Infographic")
#                 else:
#                     visual_placeholder.error(visualization_result)
# else:
#     st.warning("Please upload a file to proceed.")



########################################## with groq ####################################################
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from groq import Groq
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Load model and tokenizer globally to avoid reloading
# try:
#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#     st.success("Successfully connected to Groq API")
# except Exception as e:
#     client = None
#     st.error(f"Error connecting to Groq API: {e}")
# # Backend functions
# def process_data(file):
#     """Clean and summarize uploaded data."""
#     try:
#         df = pd.read_csv(file)
#     except Exception:
#         df = pd.read_excel(file)

#     # Example: Summarizing data trends
#     summary = {
#         "columns": list(df.columns),
#         "row_count": len(df),
#         "key_stats": df.describe(include="all").to_dict()
#     }
#     return df, summary

# def generate_narrative(description, data_summary):
#     """Generate narrative for infographic using Groq's Mixtral model."""
#     if client is None:
#         return "Error: Could not connect to Groq API. Check your API key."

#     try:
#         prompt = (
#             f"Create a clear and concise narrative for an infographic with the following description:\n"
#             f"'{description}'\n"
#             f"Data summary:\n{data_summary}\n\n"
#             "Please provide a structured analysis that highlights key insights and trends."
#         )
        
#         completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             model="mixtral-8x7b-32768",
#             temperature=0.7,
#             max_tokens=500,
#         )
        
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error generating narrative: {str(e)}"


# def generate_visualization(df, template_type):
#     """Generate visualizations based on template type."""
#     try:
#         if template_type == "Trend Analysis":
#             df.plot(x=df.columns[0], y=df.columns[1:], kind="line")
#             plt.title("Trend Analysis")
#             plt.xlabel("Time")
#             plt.ylabel("Values")
#         elif template_type == "Comparison":
#             df.plot(kind="bar")
#             plt.title("Category Comparison")
#             plt.xlabel("Categories")
#             plt.ylabel("Values")
#         elif template_type == "Breakdown":
#             df.iloc[0].plot(kind="pie", autopct="%1.1f%%")
#             plt.title("Category Breakdown")
#         plt.tight_layout()
#         plt.savefig("output_visual.png")
#         plt.close()
#         return "output_visual.png"
#     except Exception as e:
#         return f"Error generating visualization: {str(e)}"

# # Streamlit App
# st.title("AI-Powered Infographic Generator")
# st.markdown("### Transform your data into story-driven infographics.")

# # Upload Data
# uploaded_file = st.file_uploader("Upload your data file (CSV/Excel):")
# description = st.text_input("Describe the infographic you need:")
# template_type = st.selectbox("Choose a visualization template:", ["Trend Analysis", "Comparison", "Breakdown"])

# # Placeholders for interactive buttons and outputs
# narrative_placeholder = st.empty()
# visual_placeholder = st.empty()

# if uploaded_file:
#     with st.spinner("Processing your data..."):
#         df, summary = process_data(uploaded_file)
#         st.success("Data processed successfully!")
#         st.write("### Data Preview")
#         st.dataframe(df.head())

#         # Generate narrative
#         if st.button("Generate Narrative"):
#             with st.spinner("Generating narrative..."):
#                 narrative = generate_narrative(description, summary)
#                 narrative_placeholder.write("### Narrative")
#                 narrative_placeholder.write(narrative)

#         # Generate visualization
#         if st.button("Generate Visualization"):
#             with st.spinner("Creating visualization..."):
#                 visualization_result = generate_visualization(df, template_type)
#                 if visualization_result.endswith(".png"):
#                     visual_placeholder.image(visualization_result, caption="Generated Infographic")
#                 else:
#                     visual_placeholder.error(visualization_result)
# else:
#     st.warning("Please upload a file to proceed.")


############################################# with enhanced UI and all ###########################################

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import getSampleStyleSheet
# from transformers import pipeline
# from groq import Groq
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Connect to Groq API
# try:
#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#     st.success("Successfully connected to Groq API")
# except Exception as e:
#     client = None
#     st.error(f"Error connecting to Groq API: {e}")

# # Functions
# def process_data(file):
#     """Process uploaded data file."""
#     try:
#         df = pd.read_csv(file)
#     except Exception:
#         df = pd.read_excel(file)

#     summary = {
#         "columns": list(df.columns),
#         "row_count": len(df),
#         "key_stats": df.describe(include="all").to_dict(),
#     }
#     return df, summary

# def generate_narrative(description, data_summary):
#     """Generate narrative using Groq API."""
#     if client is None:
#         return "Error: Could not connect to Groq API."

#     try:
#         prompt = (
#             f"Generate a report narrative for an infographic with the following description:\n"
#             f"'{description}'\nData summary:\n{data_summary}\n\n"
#             "Provide structured sections:\n"
#             "- Executive Summary\n"
#             "- Key Insights and Trends\n"
#             "- Recommendations or conclusions\n"
#             "- Suggestions for visualization"
#         )

#         completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             model="mixtral-8x7b-32768",
#             temperature=0.7,
#             max_tokens=500,
#         )

#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error generating narrative: {str(e)}"

# def generate_visualization(df, template_type):
#     """Generate visualizations."""
#     try:
#         if template_type == "Trend Analysis":
#             df.plot(x=df.columns[0], y=df.columns[1:], kind="line")
#             plt.title("Trend Analysis")
#             plt.xlabel("Time")
#             plt.ylabel("Values")
#         elif template_type == "Comparison":
#             df.plot(kind="bar")
#             plt.title("Category Comparison")
#             plt.xlabel("Categories")
#             plt.ylabel("Values")
#         elif template_type == "Breakdown":
#             df.iloc[0].plot(kind="pie", autopct="%1.1f%%")
#             plt.title("Category Breakdown")
#         plt.tight_layout()
#         image_path = "output_visual.png"
#         plt.savefig(image_path)
#         plt.close()
#         return image_path
#     except Exception as e:
#         return f"Error generating visualization: {str(e)}"

# def create_pdf_report(narrative, visualization_path, output_filename="report.pdf"):
#     """Generate a PDF report with narrative and visualization."""
#     styles = getSampleStyleSheet()
#     doc = SimpleDocTemplate(output_filename, pagesize=letter)

#     elements = []

#     # Add narrative
#     narrative_sections = narrative.split("\n")
#     for section in narrative_sections:
#         elements.append(Paragraph(section, styles["Normal"]))
#         elements.append(Spacer(1, 12))

#     # Add visualization
#     if os.path.exists(visualization_path):
#         elements.append(Image(visualization_path, width=400, height=300))
#         elements.append(Spacer(1, 12))

#     # Save the PDF
#     doc.build(elements)
#     return output_filename

# # Streamlit App
# st.title("AI-Powered Infographic Generator")
# st.markdown("### Transform your data into story-driven infographics.")

# # File upload
# uploaded_file = st.file_uploader("Upload your data file (CSV/Excel):")
# description = st.text_input("Describe the infographic you need:")
# template_type = st.selectbox("Choose a visualization template:", ["Trend Analysis", "Comparison", "Breakdown"])

# # Generate narrative and visualization
# if uploaded_file:
#     with st.spinner("Processing your data..."):
#         df, summary = process_data(uploaded_file)
#         st.success("Data processed successfully!")
#         st.write("### Data Preview")
#         st.dataframe(df.head())

#         # Generate Narrative
#         if st.button("Generate Narrative & Report"):
#             with st.spinner("Generating narrative..."):
#                 narrative = generate_narrative(description, summary)

#                 if "Error" in narrative:
#                     st.error(narrative)
#                 else:
#                     st.write("### Narrative Report")
#                     st.write(narrative)

#                     # Generate Visualization
#                     with st.spinner("Creating visualization..."):
#                         visualization_result = generate_visualization(df, template_type)

#                         if visualization_result.endswith(".png"):
#                             st.image(visualization_result, caption="Generated Visualization")

#                             # Create PDF
#                             st.write("### Downloadable Report")
#                             report_filename = create_pdf_report(narrative, visualization_result)
#                             with open(report_filename, "rb") as pdf_file:
#                                 st.download_button(
#                                     label="Download Report",
#                                     data=pdf_file,
#                                     file_name="Infographic_Report.pdf",
#                                     mime="application/pdf",
#                                 )
#                         else:
#                             st.error(visualization_result)
# else:
#     st.warning("Please upload a file to proceed.")



############################################################################################


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit App Configuration
st.set_page_config(page_title="Enhanced Infographic Generator", layout="wide")

# Helper Functions
def process_data(file):
    """Process uploaded data file."""
    try:
        df = pd.read_csv(file)
    except Exception:
        df = pd.read_excel(file)
    return df

def generate_plotly_visualizations(df):
    """Generate interactive visualizations using Plotly."""
    st.subheader("Interactive Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Scatter Plot")
        x_col = st.selectbox("Select X-axis column", df.columns, key="scatter_x")
        y_col = st.selectbox("Select Y-axis column", df.columns, key="scatter_y")
        scatter_fig = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
        st.plotly_chart(scatter_fig)

    with col2:
        st.write("### Bar Chart")
        bar_col = st.selectbox("Select column for Bar Chart", df.columns, key="bar_chart")
        bar_fig = px.bar(df, x=bar_col, title="Bar Chart")
        st.plotly_chart(bar_fig)

    st.write("### Line Chart")
    x_col_line = st.selectbox("Select X-axis column for Line Chart", df.columns, key="line_x")
    y_col_line = st.selectbox("Select Y-axis column for Line Chart", df.columns, key="line_y")
    line_fig = px.line(df, x=x_col_line, y=y_col_line, title="Line Chart")
    st.plotly_chart(line_fig)

def generate_static_visualization(df, template_type):
    """Generate static visualizations using Matplotlib."""
    try:
        if template_type == "Trend Analysis":
            df.plot(x=df.columns[0], y=df.columns[1:], kind="line")
            plt.title("Trend Analysis")
            plt.xlabel("Time")
            plt.ylabel("Values")
        elif template_type == "Comparison":
            df.plot(kind="bar")
            plt.title("Category Comparison")
            plt.xlabel("Categories")
            plt.ylabel("Values")
        elif template_type == "Breakdown":
            df.iloc[0].plot(kind="pie", autopct="%1.1f%%")
            plt.title("Category Breakdown")
        plt.tight_layout()
        image_path = "static_visual.png"
        plt.savefig(image_path)
        plt.close()
        return image_path
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

def create_pdf_report(narrative, visualization_path, output_filename="report.pdf"):
    """Generate a PDF report with narrative and visualization."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_filename, pagesize=letter)

    elements = []

    # Add narrative
    narrative_sections = narrative.split("\n")
    for section in narrative_sections:
        elements.append(Paragraph(section, styles["Normal"]))
        elements.append(Spacer(1, 12))

    # Add visualization
    if os.path.exists(visualization_path):
        elements.append(Image(visualization_path, width=400, height=300))
        elements.append(Spacer(1, 12))

    # Save the PDF
    doc.build(elements)
    return output_filename

# Main Application
st.title("Enhanced AI-Powered Infographic Generator")
st.markdown("### Transform your data into story-driven infographics.")

# File Upload
uploaded_file = st.file_uploader("Upload your data file (CSV/Excel):")

if uploaded_file:
    with st.spinner("Processing your data..."):
        df = process_data(uploaded_file)
        st.success("Data uploaded successfully!")
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Static Visualization and Narrative
        description = st.text_input("Describe the infographic you need:")
        template_type = st.selectbox(
            "Choose a static visualization template:",
            ["Trend Analysis", "Comparison", "Breakdown"],
        )

        if st.button("Generate Narrative & Static Report"):
            st.write("Generating narrative and static visualization...")
            narrative = f"Placeholder narrative for '{description}' based on data."
            st.write("### Narrative")
            st.write(narrative)

            static_image = generate_static_visualization(df, template_type)
            if static_image.endswith(".png"):
                st.image(static_image, caption="Generated Static Visualization")

                # Generate PDF Report
                report_filename = create_pdf_report(narrative, static_image)
                with open(report_filename, "rb") as pdf_file:
                    st.download_button(
                        label="Download Report",
                        data=pdf_file,
                        file_name="Infographic_Report.pdf",
                        mime="application/pdf",
                    )
            else:
                st.error(static_image)

        # Generate Interactive Visualizations
        if st.button("Generate Interactive Visualizations"):
            generate_plotly_visualizations(df)
else:
    st.warning("Please upload a file to proceed.")
