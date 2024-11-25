import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
import time
from groq import Groq
from dotenv import load_dotenv
import os
import json
from gtts import gTTS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from io import BytesIO, StringIO
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import re
from typing import Dict, List, Tuple
from wordcloud import WordCloud

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Advanced Data Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 0rem; }
    .stApp { margin: 0; padding: 0; }
    .css-1d391kg { padding-top: 1rem; }
    .stSidebar .sidebar-content { background-color: #f8f9fa; }
    .visualization-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'story_generated' not in st.session_state:
    st.session_state.story_generated = False

# Initialize Groq client
def initialize_groq():
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return client
    except Exception as e:
        st.error(f"Error connecting to Groq API: {e}")
        return None

client = initialize_groq()

# Data Processing Functions
def process_data(file):
    """Clean and summarize uploaded data."""
    try:
        # Get file extension
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif file_extension == 'json':
            df = pd.read_json(file)
        elif file_extension == 'txt':
            df = process_text_file(file)
        else:
            raise ValueError("Unsupported file format")

        # Convert int64 columns to regular int
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')

        summary = {
            "columns": list(df.columns),
            "row_count": len(df),
            "key_stats": df.describe(include='all').to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        return df, summary
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

def process_text_file(file) -> pd.DataFrame:
    """Process text file and convert to DataFrame"""
    content = file.getvalue().decode('utf-8')
    
    # Try to detect the data structure
    lines = content.split('\n')
    data_type = detect_text_data_type(lines)
    
    if data_type == 'key_value':
        return process_key_value_text(lines)
    elif data_type == 'tabular':
        return process_tabular_text(lines)
    elif data_type == 'time_series':
        return process_time_series_text(lines)
    else:
        return process_unstructured_text(content)

def detect_text_data_type(lines: List[str]) -> str:
    """Detect the type of data in text file"""
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return 'unstructured'
    
    # Check for key-value pairs (e.g., "key: value" or "key=value")
    key_value_pattern = re.compile(r'^[\w\s]+[:=]\s*.*$')
    if all(key_value_pattern.match(line) for line in lines[:5]):
        return 'key_value'
    
    # Check for tabular data (consistent number of delimiters)
    delimiters = ['\t', ',', '|']
    for delimiter in delimiters:
        if all(line.count(delimiter) == lines[0].count(delimiter) for line in lines[:5]):
            return 'tabular'
    
    # Check for time series data (timestamp patterns)
    time_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}')
    if any(time_pattern.search(line) for line in lines[:5]):
        return 'time_series'
    
    return 'unstructured'

def process_key_value_text(lines: List[str]) -> pd.DataFrame:
    """Process key-value text data"""
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
        elif '=' in line:
            key, value = line.split('=', 1)
        else:
            continue
        data[key.strip()] = [value.strip()]
    return pd.DataFrame.from_dict(data)

def process_tabular_text(lines: List[str]) -> pd.DataFrame:
    """Process tabular text data"""
    # Try different delimiters
    delimiters = ['\t', ',', '|']
    max_cols = 0
    best_delimiter = None
    
    for delimiter in delimiters:
        cols = len(lines[0].split(delimiter))
        if cols > max_cols:
            max_cols = cols
            best_delimiter = delimiter
    
    if best_delimiter:
        # Use StringIO to create a file-like object
        text_io = StringIO('\n'.join(lines))
        return pd.read_csv(text_io, sep=best_delimiter)
    
    return pd.DataFrame([line.split() for line in lines])

def process_time_series_text(lines: List[str]) -> pd.DataFrame:
    """Process time series text data"""
    data = []
    time_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}')
    
    for line in lines:
        match = time_pattern.search(line)
        if match:
            timestamp = match.group()
            # Extract values after timestamp
            values = re.findall(r'[-+]?\d*\.\d+|\d+', line[match.end():])
            data.append([timestamp] + values)
    
    # Create DataFrame with appropriate columns
    if data:
        columns = ['timestamp'] + [f'value_{i+1}' for i in range(len(data[0])-1)]
        df = pd.DataFrame(data, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

def process_unstructured_text(content: str) -> pd.DataFrame:
    """Process unstructured text data"""
    # Perform basic text analysis
    words = content.lower().split()
    word_freq = pd.Series(words).value_counts()
    
    # Create a DataFrame with text metrics
    metrics = {
        'metric': [
            'total_words',
            'unique_words',
            'avg_word_length',
            'sentences',
            'paragraphs'
        ],
        'value': [
            len(words),
            len(set(words)),
            sum(len(word) for word in words) / len(words) if words else 0,
            len(re.findall(r'[.!?]+', content)),
            len(content.split('\n\n'))
        ]
    }
    
    # Add word frequency analysis
    freq_df = pd.DataFrame(word_freq.head(20)).reset_index()
    freq_df.columns = ['word', 'frequency']
    
    # Combine metrics and frequency analysis
    metrics_df = pd.DataFrame(metrics)
    return pd.concat([metrics_df, freq_df], axis=0)

def generate_analysis(description, data_summary, client):
    """Generate analysis using Groq's Mixtral model."""
    if client is None:
        return "Error: Groq API client not available."

    try:
        prompt = (
            f"Analyze this dataset and provide insights:\n"
            f"Description: {description}\n"
            f"Data summary: {json.dumps(data_summary, indent=2)}\n\n"
            "Please provide:\n"
            "1. Key trends\n"
            "2. Notable patterns\n"
            "3. Potential insights\n"
            "4. Recommendations"
        )
        
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000,
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def perform_statistical_analysis(data):
    """Perform comprehensive statistical analysis"""
    analysis = {
        'descriptive': data.describe(),
        'missing_values': data.isnull().sum(),
        'correlations': data.corr() if len(data.select_dtypes(include=[np.number]).columns) > 1 else None,
        'skewness': data.skew(numeric_only=True),
        'kurtosis': data.kurtosis(numeric_only=True)
    }
    return analysis

# Visualization Functions
def create_visualization(df, viz_type, columns=None):
    """Create different types of visualizations"""
    try:
        if viz_type == "Line Plot":
            fig = px.line(df, x=df.index, y=columns if columns else df.columns)
        elif viz_type == "Bar Plot":
            fig = px.bar(df, x=df.index, y=columns if columns else df.columns)
        elif viz_type == "Scatter Plot":
            if len(columns) >= 2:
                fig = px.scatter(df, x=columns[0], y=columns[1])
            else:
                st.warning("Scatter plot requires two columns")
                return None
        elif viz_type == "Box Plot":
            fig = px.box(df, y=columns if columns else df.columns)
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=columns[0] if columns else df.columns[0])
        elif viz_type == "Heatmap":
            corr = df.corr()
            fig = px.imshow(corr, aspect="auto")
        elif viz_type == "Area Plot":
            fig = px.area(df, x=df.index, y=columns if columns else df.columns)
        elif viz_type == "Violin Plot":
            fig = px.violin(df, y=columns if columns else df.columns)
        elif viz_type == "Sunburst":
            if columns and len(columns) >= 2:
                fig = px.sunburst(df, path=columns)
            else:
                st.warning("Sunburst plot requires at least two hierarchical columns")
                return None
        else:
            return None

        fig.update_layout(
            title=viz_type,
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_white",
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def create_animated_plot(data, column, duration=5):
    """Create an animated plot"""
    try:
        frames = []
        for i in range(1, len(data) + 1):
            frame = go.Frame(
                data=[go.Scatter(
                    x=data.index[:i],
                    y=data[column][:i],
                    mode='lines+markers'
                )],
                name=str(i)
            )
            frames.append(frame)

        fig = go.Figure(
            frames=frames,
            layout=go.Layout(
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [{
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": duration * 1000 / len(frames)},
                            "fromcurrent": True,
                            "transition": {"duration": 300}
                        }]
                    }]
                }]
            )
        )

        # Add initial data
        fig.add_trace(go.Scatter(
            x=[data.index[0]],
            y=[data[column].iloc[0]],
            mode='lines+markers',
            name=column
        ))

        fig.update_layout(
            title=f"Animated {column} Trend",
            xaxis_title="Time",
            yaxis_title=column,
            template="plotly_white",
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"Error creating animation: {e}")
        return None

def create_bar_chart_race(data, column, duration):
    """Create animated bar chart race"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        current_data = data[column].iloc[:frame]
        current_data.plot(kind='bar', ax=ax)
        ax.set_title(f"{column} Over Time")
        ax.set_ylim([data[column].min(), data[column].max()])
    
    frames = len(data)
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, 
        interval=duration*1000/frames, repeat=False
    )
    
    return anim

def create_scatter_animation(data, x_col, y_col, duration):
    """Create animated scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        ax.scatter(data[x_col].iloc[:frame], data[y_col].iloc[:frame])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlim([data[x_col].min(), data[x_col].max()])
        ax.set_ylim([data[y_col].min(), data[y_col].max()])
    
    frames = len(data)
    anim = animation.FuncAnimation(
        fig, animate, frames=frames,
        interval=duration*1000/frames, repeat=False
    )
    
    return anim

def save_animation(anim, filename):
    """Save animation as HTML"""
    html_string = anim.to_jshtml()
    with open(filename, 'w') as f:
        f.write(html_string)
    return filename

def create_text_visualization(text_data):
    """Create visualizations for text data"""
    try:
        # Word Cloud
        if 'word' in text_data.columns and 'frequency' in text_data.columns:
            word_freq = dict(zip(text_data['word'], text_data['frequency']))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Top words bar chart
            fig = px.bar(
                text_data.head(10), 
                x='word', 
                y='frequency',
                title='Top 10 Most Frequent Words'
            )
            st.plotly_chart(fig)
            
        # Text metrics visualization
        if 'metric' in text_data.columns and 'value' in text_data.columns:
            metrics_data = text_data[text_data['metric'].isin([
                'total_words', 'unique_words', 'avg_word_length', 
                'sentences', 'paragraphs'
            ])]
            
            fig = px.bar(
                metrics_data,
                x='metric',
                y='value',
                title='Text Analysis Metrics'
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error creating text visualization: {e}")

# Main Streamlit Interface
def main():
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["📊 Data Upload", "🔍 Analysis", "📈 Visualization", "📖 Story", "🎬 Animation"],
            icons=['upload', 'search', 'graph-up', 'book', 'film'],
            menu_icon="cast",
            default_index=0
        )
    
    if selected == "📊 Data Upload":
        st.title("Data Upload")
        
        # File upload section
        st.write("### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV, Excel, JSON, TXT)", 
            type=['csv', 'xlsx', 'json', 'txt']
        )
        
        if uploaded_file:
            with st.spinner("Processing data..."):
                df, summary = process_data(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    st.session_state.summary = summary
                    st.success("Data uploaded successfully!")
                    
                    # Data preview
                    st.write("### Data Preview")
                    st.dataframe(df.head())
                    
                    # Data summary
                    st.write("### Data Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", summary['row_count'])
                    with col2:
                        st.metric("Total Features", len(summary['columns']))
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    # Column information
                    st.write("### Column Information")
                    col_info = pd.DataFrame({
                        'Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_info)
                    
                    # For text files, show additional text analysis
                    if uploaded_file.name.endswith('.txt'):
                        st.write("### Text Analysis")
                        create_text_visualization(df)
    
    elif selected == "🔍 Analysis":
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("Please upload data first!")
            return
            
        st.title("Data Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Statistical Analysis", "Correlation Analysis", "Distribution Analysis", 
             "Time Series Analysis", "Custom Analysis"]
        )
        
        if analysis_type == "Statistical Analysis":
            st.write("### Statistical Summary")
            stats = perform_statistical_analysis(st.session_state.data)
            
            # Display descriptive statistics
            st.write("#### Descriptive Statistics")
            st.dataframe(stats['descriptive'])
            
            # Display missing values
            st.write("#### Missing Values Analysis")
            missing_df = pd.DataFrame(stats['missing_values'], columns=['Missing Count'])
            missing_df['Missing Percentage'] = (missing_df['Missing Count'] / 
                                              len(st.session_state.data) * 100)
            st.dataframe(missing_df)
            
            # Display skewness and kurtosis
            st.write("#### Distribution Metrics")
            dist_metrics = pd.DataFrame({
                'Skewness': stats['skewness'],
                'Kurtosis': stats['kurtosis']
            })
            st.dataframe(dist_metrics)
        
        elif analysis_type == "Correlation Analysis":
            st.write("### Correlation Analysis")
            numeric_data = st.session_state.data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                # Correlation matrix
                corr_matrix = numeric_data.corr()
                
                # Heatmap
                fig = px.imshow(corr_matrix,
                              labels=dict(color="Correlation"),
                              title="Correlation Heatmap")
                st.plotly_chart(fig)
                
                # Strong correlations
                st.write("#### Strong Correlations (|r| > 0.7)")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr))
                else:
                    st.info("No strong correlations found")
            else:
                st.warning("No numeric columns found for correlation analysis")
        
        elif analysis_type == "Distribution Analysis":
            st.write("### Distribution Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Select Column", st.session_state.data.columns)
            with col2:
                plot_type = st.selectbox("Select Plot Type", 
                                       ["Histogram", "Box Plot", "Violin Plot"])
            
            if st.session_state.data[column].dtype in ['int64', 'float64']:
                if plot_type == "Histogram":
                    fig = px.histogram(st.session_state.data, x=column,
                                     title=f"Distribution of {column}")
                elif plot_type == "Box Plot":
                    fig = px.box(st.session_state.data, y=column,
                                title=f"Box Plot of {column}")
                else:  # Violin Plot
                    fig = px.violin(st.session_state.data, y=column,
                                  title=f"Violin Plot of {column}")
                st.plotly_chart(fig)
                
                # Descriptive statistics
                st.write("#### Descriptive Statistics")
                stats = st.session_state.data[column].describe()
                st.dataframe(stats)
            else:
                st.warning("Selected column must be numeric for distribution analysis")
        
        elif analysis_type == "Time Series Analysis":
            st.write("### Time Series Analysis")
            
            # Time column selection
            time_col = st.selectbox("Select Time Column", st.session_state.data.columns)
            value_col = st.selectbox("Select Value Column", 
                                   st.session_state.data.select_dtypes(include=[np.number]).columns)
            
            try:
                # Convert to datetime
                time_data = pd.to_datetime(st.session_state.data[time_col])
                
                # Create time series plot
                fig = px.line(st.session_state.data, x=time_col, y=value_col,
                             title=f"{value_col} over Time")
                st.plotly_chart(fig)
                
                # Moving averages
                window_size = st.slider("Moving Average Window Size", 2, 30, 7)
                ma = st.session_state.data[value_col].rolling(window=window_size).mean()
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=time_data, y=st.session_state.data[value_col],
                                        name='Original'))
                fig2.add_trace(go.Scatter(x=time_data, y=ma,
                                        name=f'{window_size}-point Moving Average'))
                fig2.update_layout(title=f"Moving Average Analysis of {value_col}")
                st.plotly_chart(fig2)
                
            except Exception as e:
                st.error(f"Error in time series analysis: {str(e)}")
        
        elif analysis_type == "Custom Analysis":
            st.write("### Custom Analysis with AI")
            question = st.text_input("Ask a question about your data:")
            if question and client:
                with st.spinner("Generating analysis..."):
                    analysis = generate_analysis(question, st.session_state.summary, client)
                    st.write(analysis)
                    
                    # Generate relevant visualization if possible
                    try:
                        if any(keyword in question.lower() 
                              for keyword in ['trend', 'time', 'change']):
                            fig = create_visualization(st.session_state.data, "Line Plot")
                            st.plotly_chart(fig)
                        elif any(keyword in question.lower() 
                                for keyword in ['compare', 'difference']):
                            fig = create_visualization(st.session_state.data, "Bar Plot")
                            st.plotly_chart(fig)
                        elif any(keyword in question.lower() 
                                for keyword in ['relationship', 'correlation']):
                            fig = create_visualization(st.session_state.data, "Scatter Plot")
                            st.plotly_chart(fig)
                    except Exception as e:
                        st.warning(f"Could not generate visualization: {str(e)}")
            
        elif selected == "📈 Visualization":
            if 'data' not in st.session_state or st.session_state.data is None:
                st.warning("Please upload data first!")
                return
            
        st.title("Data Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Line Plot", "Bar Plot", "Scatter Plot", "Box Plot", 
             "Histogram", "Heatmap", "Area Plot", "Violin Plot", "Sunburst"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if viz_type in ["Line Plot", "Bar Plot", "Box Plot", "Area Plot", "Violin Plot"]:
                columns = st.multiselect("Select Columns", st.session_state.data.columns)
            elif viz_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis", st.session_state.data.columns)
                y_col = st.selectbox("Select Y-axis", st.session_state.data.columns)
                columns = [x_col, y_col]
            elif viz_type in ["Histogram"]:
                columns = [st.selectbox("Select Column", st.session_state.data.columns)]
            elif viz_type == "Sunburst":
                columns = st.multiselect("Select Hierarchical Columns (order matters)", 
                                       st.session_state.data.columns)
        
        with col2:
            if viz_type != "Heatmap":
                color_col = st.selectbox("Color by (optional)", 
                                       ["None"] + list(st.session_state.data.columns))
        
        if st.button("Generate Visualization"):
            with st.spinner("Creating visualization..."):
                fig = create_visualization(st.session_state.data, viz_type, columns)
                if fig:
                    st.plotly_chart(fig)
                    
                    # Add download button
                    if st.button("Download Visualization"):
                        filename = f"{viz_type.lower().replace(' ', '')}{int(time.time())}.html"
                        fig.write_html(filename)
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="Download HTML",
                                data=f,
                                file_name=filename,
                                mime="text/html"
                            )
    
    elif selected == "📖 Story":
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("Please upload data first!")
            return
            
        st.title("Data Story Generation")
        
        story_type = st.selectbox(
            "Select Story Type",
            ["Overview", "Trend Analysis", "Comparative Analysis", "Custom Story"]
        )
        
        if story_type == "Overview":
            if client:
                with st.spinner("Generating overview..."):
                    overview = generate_analysis("Generate a comprehensive overview", 
                                              st.session_state.summary, 
                                              client)
                    st.write(overview)
                    
                    # Generate summary visualizations
                    col1, col2 = st.columns(2)
                    with col1:
                        # Data types distribution
                        dtypes_count = st.session_state.data.dtypes.value_counts()
                        fig = px.pie(values=dtypes_count.values, 
                                   names=dtypes_count.index, 
                                   title="Data Types Distribution")
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Missing values heatmap
                        missing = st.session_state.data.isnull()
                        fig = px.imshow(missing, 
                                      title="Missing Values Heatmap")
                        st.plotly_chart(fig)
        
        elif story_type == "Trend Analysis":
            column = st.selectbox("Select Column for Trend Analysis", 
                                st.session_state.data.select_dtypes(include=[np.number]).columns)
            if st.button("Analyze Trends"):
                fig = create_visualization(st.session_state.data, "Line Plot", [column])
                st.plotly_chart(fig)
                
                if client:
                    trend_analysis = generate_analysis(
                        f"Analyze trends in {column}", 
                        {"column": column, "data": st.session_state.data[column].describe().to_dict()},
                        client
                    )
                    st.write("### Trend Analysis")
                    st.write(trend_analysis)
        
        elif story_type == "Comparative Analysis":
            col1, col2 = st.columns(2)
            with col1:
                column1 = st.selectbox("Select First Column", st.session_state.data.columns)
            with col2:
                column2 = st.selectbox("Select Second Column", st.session_state.data.columns)
                
            if st.button("Compare"):
                # Create comparison visualizations
                if st.session_state.data[column1].dtype in ['int64', 'float64'] and \
                   st.session_state.data[column2].dtype in ['int64', 'float64']:
                    fig = px.scatter(st.session_state.data, x=column1, y=column2,
                                   title=f"{column1} vs {column2}")
                    st.plotly_chart(fig)
                
                if client:
                    comparison = generate_analysis(
                        f"Compare {column1} and {column2}",
                        {"columns": [column1, column2]},
                        client
                    )
                    st.write("### Comparison Analysis")
                    st.write(comparison)
        
        elif story_type == "Custom Story":
            story_prompt = st.text_area("What aspects of the data would you like to explore?")
            if story_prompt and client:
                with st.spinner("Generating custom story..."):
                    story = generate_analysis(story_prompt, st.session_state.summary, client)
                    st.write(story)
    
    elif selected == "🎬 Animation":
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("Please upload data first!")
            return
            
        st.title("Animated Visualizations")
        
        animation_type = st.selectbox(
            "Select Animation Type",
            ["Trend Animation", "Bar Chart Race", "Scatter Plot Animation"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if animation_type in ["Trend Animation", "Bar Chart Race"]:
                column = st.selectbox(
                    "Select Column to Animate", 
                    st.session_state.data.select_dtypes(include=[np.number]).columns
                )
            else:  # Scatter Plot Animation
                x_col = st.selectbox(
                    "Select X-axis Column",
                    st.session_state.data.select_dtypes(include=[np.number]).columns
                )
                y_col = st.selectbox(
                    "Select Y-axis Column",
                    st.session_state.data.select_dtypes(include=[np.number]).columns
                )
        
        with col2:
            duration = st.slider("Animation Duration (seconds)", 1, 10, 5)
            
        if st.button("Generate Animation"):
            with st.spinner("Creating animation..."):
                try:
                    if animation_type == "Trend Animation":
                        fig = create_animated_plot(st.session_state.data, column, duration)
                        st.plotly_chart(fig)
                        
                        # Add download button for Plotly animation
                        if st.button("Download Animation"):
                            filename = f"trend_animation_{int(time.time())}.html"
                            fig.write_html(filename)
                            with open(filename, "rb") as f:
                                st.download_button(
                                    label="Download HTML",
                                    data=f,
                                    file_name=filename,
                                    mime="text/html"
                                )
                    
                    elif animation_type == "Bar Chart Race":
                        anim = create_bar_chart_race(
                            st.session_state.data, column, duration
                        )
                        
                        # Save and display animation
                        filename = f"bar_race_{int(time.time())}.html"
                        save_animation(anim, filename)
                        with open(filename, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                        
                        # Add download button
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="Download Animation",
                                data=f,
                                file_name=filename,
                                mime="text/html"
                            )
                    
                    elif animation_type == "Scatter Plot Animation":
                        anim = create_scatter_animation(
                            st.session_state.data, x_col, y_col, duration
                        )
                        
                        # Save and display animation
                        filename = f"scatter_animation_{int(time.time())}.html"
                        save_animation(anim, filename)
                        with open(filename, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                        
                        # Add download button
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="Download Animation",
                                data=f,
                                file_name=filename,
                                mime="text/html"
                            )
                    
                except Exception as e:
                    st.error(f"Error creating animation: {str(e)}")
        
        # Add animation tips
        with st.expander("Animation Tips"):
            st.markdown("""
            - *Trend Animation*: Best for time series data
            - *Bar Chart Race*: Great for comparing changing values over time
            - *Scatter Plot Animation*: Perfect for showing relationships between variables
            
            For best results:
            1. Choose numeric columns for animation
            2. Adjust duration based on your data size
            3. Download animations for sharing or embedding
            """)

if _name_ == "_main_":
    main()
