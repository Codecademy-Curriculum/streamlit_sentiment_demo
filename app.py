import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import re
from transformers import pipeline
import torch

def standardize_sentiment_label(label):
    """Standardize sentiment labels to uppercase"""
    if pd.isna(label):
        return 'NEUTRAL'
    
    label_str = str(label).lower()
    
    mapping = {
        'label_0': 'NEGATIVE',
        'label_1': 'NEUTRAL',  
        'label_2': 'POSITIVE',
        'negative': 'NEGATIVE',
        'positive': 'POSITIVE',
        'neutral': 'NEUTRAL'
    }
    
    return mapping.get(label_str, str(label).upper())

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

@st.cache_resource
def load_sentiment_model():
    """Load and cache the sentiment analysis model"""
    try:
        # Use a fast, lightweight model that works well on CPU
        classifier = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to a simpler model
        try:
            classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            return classifier
        except:
            return None

def preprocess_text(text):
    """Clean and preprocess text for better analysis"""
    if pd.isna(text):
        return ""
    
    # Convert to string and clean
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Limit length for model constraints
    return text[:512]

def analyze_sentiment_batch(texts, classifier, batch_size=16):
    """Analyze sentiment for multiple texts in batches"""
    results = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = classifier(batch)
            results.extend(batch_results)
        except Exception as e:
            st.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add empty results for failed batch
            results.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
    
    return results

def create_sentiment_metrics(df):
    """Create sentiment distribution metrics"""
    if df is None or len(df) == 0:
        st.warning("No data available for analysis.")
        return
    
    # Handle different label formats from different models
    # Map various sentiment labels to standard format
    label_mapping = {
        'LABEL_0': 'NEGATIVE',
        'LABEL_1': 'POSITIVE', 
        'LABEL_2': 'NEUTRAL',
        'NEGATIVE': 'NEGATIVE',
        'POSITIVE': 'POSITIVE',
        'NEUTRAL': 'NEUTRAL',
                # Add lowercase mappings
        'negative': 'NEGATIVE',
        'positive': 'POSITIVE',
        'neutral': 'NEUTRAL'
    }
    
    df['sentiment_mapped'] = df['sentiment'].map(label_mapping).fillna(df['sentiment'])
    sentiment_counts = df['sentiment_mapped'].value_counts()
    total = len(df)
    print(df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Analyzed", 
            f"{total:,}",
            help="Total number of texts analyzed"
        )
    
    with col2:
        positive_count = sentiment_counts.get('POSITIVE', 0)
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        st.metric(
            "Positive", 
            f"{positive_count:,} ({positive_pct:.1f}%)",
            delta=f"{positive_pct:.1f}%",
            delta_color="normal"
        )
    
    with col3:
        negative_count = sentiment_counts.get('NEGATIVE', 0)
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        st.metric(
            "Negative", 
            f"{negative_count:,} ({negative_pct:.1f}%)",
            delta=f"-{negative_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        neutral_count = sentiment_counts.get('NEUTRAL', 0)
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        st.metric(
            "Neutral", 
            f"{neutral_count:,} ({neutral_pct:.1f}%)",
            delta=f"{neutral_pct:.1f}%",
            delta_color="off"
        )

def create_visualizations(df):
    """Create visualizations using matplotlib with proper Streamlit integration"""
    if df is None or len(df) == 0:
        st.warning("No data available for visualization.")
        return None, None, None
    
    # Handle different label formats
    label_mapping = {
        'LABEL_0': 'NEGATIVE',
        'LABEL_1': 'NEUTRAL', 
        'LABEL_2': 'POSITIVE',
        'NEGATIVE': 'NEGATIVE',
        'POSITIVE': 'POSITIVE',
        'NEUTRAL': 'NEUTRAL',
                # Add lowercase mappings
        'negative': 'NEGATIVE',
        'positive': 'POSITIVE',
        'neutral': 'NEUTRAL'
    }
    
    df['sentiment_mapped'] = df['sentiment'].map(label_mapping).fillna(df['sentiment'])
    
    # Debug output
    st.write(f"Debug: Mapped sentiment counts: {df['sentiment_mapped'].value_counts()}")
    
    # Define colors
    color_map = {'POSITIVE': '#2E8B57', 'NEGATIVE': '#DC143C', 'NEUTRAL': '#4682B4'}
    
    # 1. Sentiment Distribution Pie Chart
    plt.style.use('default')  # Reset style
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    
    sentiment_counts = df['sentiment_mapped'].value_counts()
    
    if len(sentiment_counts) > 0:
        # Prepare data for pie chart
        labels = sentiment_counts.index.tolist()
        sizes = sentiment_counts.values.tolist()
        colors = [color_map.get(label, '#636EFA') for label in labels]
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        ax1.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        ax1.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
    
    fig1.tight_layout()
    
    # 2. Confidence Score Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='white')
    
    if 'confidence' in df.columns and len(df) > 0:
        # Create histogram for each sentiment
        sentiment_order = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        
        for sentiment in sentiment_order:
            if sentiment in df['sentiment_mapped'].values:
                subset = df[df['sentiment_mapped'] == sentiment]
                if len(subset) > 0:
                    ax2.hist(
                        subset['confidence'], 
                        bins=15, 
                        alpha=0.7, 
                        label=sentiment, 
                        color=color_map[sentiment],
                        edgecolor='black',
                        linewidth=0.5
                    )
        
        ax2.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Confidence Score Distribution by Sentiment', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('white')
    else:
        ax2.text(0.5, 0.5, 'No Confidence Data', ha='center', va='center', fontsize=16)
        ax2.set_title('Confidence Score Distribution', fontsize=16, fontweight='bold')
    
    fig2.tight_layout()
    
    # 3. Top Confident Predictions
    fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
    
    # Most confident positive
    positive_data = df[df['sentiment_mapped'] == 'POSITIVE']
    if not positive_data.empty:
        top_positive = positive_data.nlargest(min(5, len(positive_data)), 'confidence')
        if not top_positive.empty:
            y_pos = range(len(top_positive))
            bars = ax3_left.barh(y_pos, top_positive['confidence'], color='#2E8B57', alpha=0.8, edgecolor='black')
            ax3_left.set_yticks(y_pos)
            ax3_left.set_yticklabels([f'Text {i+1}' for i in y_pos])
            ax3_left.set_xlabel('Confidence Score', fontweight='bold')
            ax3_left.set_title('Most Confident Positive Predictions', fontweight='bold', pad=15)
            ax3_left.set_xlim(0, 1)
            ax3_left.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3_left.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                             f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            ax3_left.text(0.5, 0.5, 'No positive samples', ha='center', va='center', 
                         transform=ax3_left.transAxes, fontsize=12)
    else:
        ax3_left.text(0.5, 0.5, 'No positive samples', ha='center', va='center', 
                     transform=ax3_left.transAxes, fontsize=12)
    
    ax3_left.set_title('Most Confident Positive Predictions', fontweight='bold')
    ax3_left.set_facecolor('white')
    
    # Most confident negative  
    negative_data = df[df['sentiment_mapped'] == 'NEGATIVE']
    if not negative_data.empty:
        top_negative = negative_data.nlargest(min(5, len(negative_data)), 'confidence')
        if not top_negative.empty:
            y_pos = range(len(top_negative))
            bars = ax3_right.barh(y_pos, top_negative['confidence'], color='#DC143C', alpha=0.8, edgecolor='black')
            ax3_right.set_yticks(y_pos)
            ax3_right.set_yticklabels([f'Text {i+1}' for i in y_pos])
            ax3_right.set_xlabel('Confidence Score', fontweight='bold')
            ax3_right.set_title('Most Confident Negative Predictions', fontweight='bold', pad=15)
            ax3_right.set_xlim(0, 1)
            ax3_right.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3_right.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                              f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            ax3_right.text(0.5, 0.5, 'No negative samples', ha='center', va='center', 
                          transform=ax3_right.transAxes, fontsize=12)
    else:
        ax3_right.text(0.5, 0.5, 'No negative samples', ha='center', va='center', 
                      transform=ax3_right.transAxes, fontsize=12)
    
    ax3_right.set_title('Most Confident Negative Predictions', fontweight='bold')
    ax3_right.set_facecolor('white')
    
    fig3.tight_layout()
    
    return fig1, fig2, fig3

# Main App
st.markdown('<h1 class="main-header">📊 AI Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
Analyze sentiment in text data using state-of-the-art AI models from Hugging Face. 
Upload your data or enter text directly to get insights into emotional tone and sentiment patterns.
""")

# Sidebar
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Choose Model",
        [
            "cardiffnlp/twitter-roberta-base-sentiment-latest (Recommended)",
            "distilbert-base-uncased-finetuned-sst-2-english (Faster)"
        ]
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    batch_size = st.slider("Batch Size", 8, 32, 16, help="Larger batches are faster but use more memory")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, help="Minimum confidence to highlight results")
    
    # Load model
    if st.button("🚀 Load Model", type="primary"):
        with st.spinner("Loading AI model..."):
            st.session_state.classifier = load_sentiment_model()
            st.session_state.model_loaded = True
        
        if st.session_state.classifier:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Failed to load model")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📁 File Upload", "📊 Results & Analytics"])

with tab1:
    st.header("Single Text Analysis")
    
    # Text input
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste your text here...",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("Analyze Text", type="primary", disabled=not st.session_state.model_loaded)
    
    if analyze_button and user_text and st.session_state.classifier:
        with st.spinner("Analyzing sentiment..."):
            cleaned_text = preprocess_text(user_text)
            result = st.session_state.classifier(cleaned_text)[0]
            
            # Display results
            sentiment = result['label']
            confidence = result['score']
            
            # Map sentiment labels
            label_mapping = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'POSITIVE', 
                'LABEL_2': 'NEUTRAL',
                'NEGATIVE': 'NEGATIVE',
                'POSITIVE': 'POSITIVE',
                'NEUTRAL': 'NEUTRAL'
            }
            
            mapped_sentiment = label_mapping.get(sentiment, sentiment)
            
            # Color coding
            color_map = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '🔵'}
            
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Sentiment", f"{color_map.get(mapped_sentiment, '⚪')} {mapped_sentiment}")
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Progress bar for confidence
            st.progress(confidence)
            
            if confidence < confidence_threshold:
                st.warning(f"⚠️ Low confidence score ({confidence:.2%}). Results may be unreliable.")

with tab2:
    st.header("Bulk File Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with text data to analyze"
    )
    
    if uploaded_file:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Show preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Column selection
            text_column = st.selectbox(
                "Select the column containing text to analyze:",
                options=df.columns.tolist(),
                help="Choose which column contains the text you want to analyze"
            )
            
            # Optional: Date column for time series analysis
            date_columns = [col for col in df.columns if df[col].dtype == 'object']
            date_column = st.selectbox(
                "Select date column (optional):",
                options=["None"] + date_columns,
                help="Optional: Select a date column for time series analysis"
            )
            
            # Analysis button
            if st.button("🔍 Analyze File", type="primary", disabled=not st.session_state.model_loaded):
                if not st.session_state.classifier:
                    st.error("Please load the model first!")
                else:
                    with st.spinner(f"Analyzing {len(df)} texts... This may take a few minutes."):
                        # Prepare texts
                        texts = df[text_column].fillna("").apply(preprocess_text).tolist()
                        
                        # Analyze in batches
                        results = analyze_sentiment_batch(texts, st.session_state.classifier, batch_size)
                        
                        # Process results
                        df['sentiment'] = [r['label'] for r in results]
                        df['confidence'] = [r['score'] for r in results]
                        df['text_processed'] = texts
                        
                        # Handle date column
                        if date_column != "None":
                            try:
                                df['date'] = pd.to_datetime(df[date_column])
                            except:
                                st.warning("Could not parse date column. Time series analysis will be skipped.")
                        
                        st.session_state.analyzed_data = df
                        
                    st.success("✅ Analysis complete!")
                    st.balloons()
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.header("Results & Analytics")
    
    if st.session_state.analyzed_data is not None:
        df = st.session_state.analyzed_data
        
        # Metrics
        create_sentiment_metrics(df)
        
        st.markdown("---")
        
        # Visualizations using seaborn
        try:
            fig1, fig2, fig3 = create_visualizations(df)
            
            if fig1 is not None:
                # Display charts in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(fig1)
                
                with col2:
                    st.pyplot(fig2)
                
                if fig3 is not None:
                    st.pyplot(fig3)
            
        except Exception as e:
            st.error(f"Error creating visualizations: {e}")
        
        # Time series analysis (if date column available)
        # Time series analysis - SIMPLER VERSION
        if 'date' in df.columns:
            st.markdown("### 📈 Sentiment Over Time")
            
            try:
                # Use standardized sentiment labels
                df['sentiment_mapped'] = df['sentiment'].apply(standardize_sentiment_label)
                
                # Simple approach: count by date and sentiment
                df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')  # Convert to string for grouping
                
                time_data = df.groupby(['date_str', 'sentiment_mapped']).size().unstack(fill_value=0)
                
                st.write(f"Debug: Time data shape: {time_data.shape}")
                st.write("Debug: Time data:")
                st.dataframe(time_data)
                
                if not time_data.empty:
                    # Create plot
                    fig_time, ax_time = plt.subplots(figsize=(12, 6))
                    
                    colors = {'POSITIVE': '#2E8B57', 'NEGATIVE': '#DC143C', 'NEUTRAL': '#4682B4'}
                    
                    for sentiment in time_data.columns:
                        if sentiment in colors:
                            ax_time.plot(
                                range(len(time_data)), 
                                time_data[sentiment], 
                                marker='o', 
                                label=sentiment, 
                                color=colors[sentiment], 
                                linewidth=2
                            )
                    
                    # Set x-axis labels
                    ax_time.set_xticks(range(len(time_data)))
                    ax_time.set_xticklabels(time_data.index, rotation=45)
                    
                    ax_time.set_title("Sentiment Trends Over Time", fontsize=16, fontweight='bold')
                    ax_time.set_xlabel("Date", fontsize=12)
                    ax_time.set_ylabel("Count", fontsize=12)
                    ax_time.legend()
                    ax_time.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_time, use_container_width=True)
                    plt.close(fig_time)
                else:
                    st.warning("No time series data to display.")
            
            except Exception as e:
                st.error(f"Time series error: {e}")
                st.code(traceback.format_exc())
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        # Map sentiments for filtering
        label_mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'POSITIVE', 
            'LABEL_2': 'NEUTRAL',
            'NEGATIVE': 'NEGATIVE',
            'POSITIVE': 'POSITIVE',
            'NEUTRAL': 'NEUTRAL'
        }
        
        df['sentiment_mapped'] = df['sentiment'].map(label_mapping).fillna(df['sentiment'])
        
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment:",
                options=df['sentiment_mapped'].unique(),
                default=df['sentiment_mapped'].unique()
            )
        with col2:
            min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0)
        with col3:
            max_rows = st.number_input("Max Rows to Display:", 10, 1000, 100)
        
        # Apply filters
        filtered_df = df[
            (df['sentiment_mapped'].isin(sentiment_filter)) & 
            (df['confidence'] >= min_confidence)
        ].head(max_rows)
        
        # Display results
        display_columns = ['text_processed', 'sentiment_mapped', 'confidence']
        if 'date' in df.columns:
            display_columns = ['date'] + display_columns
        
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Download results
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👆 Upload and analyze a file in the 'File Upload' tab to see results here.")
        
        # Sample data section
        st.markdown("### 🎯 Try with Sample Data")
        if st.button("Load Sample Data"):
            sample_data = {
                'text': [
                    "I absolutely love this product! It's amazing!",
                    "This is terrible. Worst experience ever.",
                    "It's okay, nothing special but does the job.",
                    "Outstanding service and quality!",
                    "Complete waste of money. Very disappointed.",
                    "Average product, meets expectations.",
                    "Incredible! Exceeded all my expectations!",
                    "Poor quality and overpriced."
                ],
                'date': pd.date_range('2024-01-01', periods=8, freq='D')
            }
            sample_df = pd.DataFrame(sample_data)
            
            # Analyze sample data
            with st.spinner("Analyzing sample data..."):
                if st.session_state.classifier:
                    texts = sample_df['text'].apply(preprocess_text).tolist()
                    results = analyze_sentiment_batch(texts, st.session_state.classifier)
                    
                    sample_df['sentiment'] = [r['label'] for r in results]
                    sample_df['confidence'] = [r['score'] for r in results]
                    sample_df['text_processed'] = texts
                    
                    st.session_state.analyzed_data = sample_df
                    st.rerun()
                else:
                    st.error("Please load the model first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit, Seaborn, and Hugging Face Transformers<br>
    Perfect for learning AI deployment and sentiment analysis!
</div>
""", unsafe_allow_html=True)