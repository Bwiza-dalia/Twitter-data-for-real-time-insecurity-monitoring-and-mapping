import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os

# Set page configuration
st.set_page_config(
    page_title="NYC Security Incident Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        padding: 1rem;
    }
    .stDataFrame {
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .insights-card {
        background-color: #E1F5FE;
        border-left: 5px solid #0288D1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_security_data.csv')
        return df
    except FileNotFoundError:
        st.error("Processed data file not found. Please run the analysis script first.")
        return None

# Main header
st.markdown('<h1 class="main-header">NYC Security Incident Dashboard</h1>', unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None:
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month_name()
        df['day_of_week'] = df['date'].dt.day_name()

    # Sidebar filters
    st.sidebar.header('Filters')
    
    # Location filter
    locations = ['All'] + sorted(df['primary_location'].unique().tolist())
    selected_location = st.sidebar.selectbox('Select Location', locations)
    
    # Security level filter
    security_levels = ['All'] + sorted(df['security_level'].unique().tolist())
    selected_security = st.sidebar.selectbox('Select Security Level', security_levels)
    
    # Crime category filter
    categories = ['All'] + sorted(df['crime_category'].unique().tolist())
    selected_category = st.sidebar.selectbox('Select Crime Category', categories)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['primary_location'] == selected_location]
    if selected_security != 'All':
        filtered_df = filtered_df[filtered_df['security_level'] == selected_security]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['crime_category'] == selected_category]
    
    # Display filters applied
    st.sidebar.markdown("---")
    st.sidebar.write("Filters Applied:")
    st.sidebar.write(f"- Location: {selected_location}")
    st.sidebar.write(f"- Security Level: {selected_security}")
    st.sidebar.write(f"- Crime Category: {selected_category}")
    
    # Dashboard layout
    # Row 1: Summary metrics
    st.markdown('<h2 class="section-header">Summary Metrics</h2>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(filtered_df)}</div>
            <div class="metric-label">Total Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        high_count = len(filtered_df[filtered_df['security_level'] == 'high'])
        high_percentage = (high_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background-color: #FEECEC;">
            <div class="metric-value" style="color: #B91C1C;">{high_count}</div>
            <div class="metric-label">High Security Incidents ({high_percentage:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        moderate_count = len(filtered_df[filtered_df['security_level'] == 'moderate'])
        moderate_percentage = (moderate_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background-color: #FEF3C7;">
            <div class="metric-value" style="color: #D97706;">{moderate_count}</div>
            <div class="metric-label">Moderate Security Incidents ({moderate_percentage:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        low_count = len(filtered_df[filtered_df['security_level'] == 'low'])
        low_percentage = (low_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background-color: #ECFDF5;">
            <div class="metric-value" style="color: #059669;">{low_count}</div>
            <div class="metric-label">Low Security Incidents ({low_percentage:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Security Levels Map and Crime Categories
    st.markdown('<h2 class="section-header">Security Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Security level by location (Top 10)
        location_security = pd.crosstab(df['primary_location'], df['security_level'])
        location_security['total'] = location_security.sum(axis=1)
        top_locations = location_security.sort_values('total', ascending=False).head(10).index
        
        # Filter to top locations for better visualization
        location_security_filtered = location_security.loc[top_locations]
        
        # Create stacked bar chart
        fig = px.bar(
            location_security_filtered, 
            y=location_security_filtered.index,
            x=['high', 'moderate', 'low'] if all(level in location_security_filtered.columns for level in ['high', 'moderate', 'low']) else location_security_filtered.columns[:-1],
            title="Security Level by Top Locations",
            color_discrete_map={'high': '#DC2626', 'moderate': '#F59E0B', 'low': '#10B981', 'unknown': '#9CA3AF'},
            orientation='h'
        )
        fig.update_layout(legend_title="Security Level", barmode='stack', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Crime categories distribution
        crime_counts = filtered_df['crime_category'].value_counts()
        fig = px.pie(
            values=crime_counts.values,
            names=crime_counts.index,
            title="Crime Categories Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Time Analysis and Engagement
    st.markdown('<h2 class="section-header">Temporal Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'day_of_week' in filtered_df.columns:
            # Day of week analysis
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = filtered_df['day_of_week'].value_counts().reindex(day_order)
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Incidents by Day of Week",
                color_discrete_sequence=['#4F46E5']
            )
            fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Incidents", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'month' in filtered_df.columns:
            # Month analysis
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_counts = filtered_df['month'].value_counts().reindex(month_order)
            
            fig = px.line(
                x=month_counts.index,
                y=month_counts.values,
                title="Incidents by Month",
                markers=True,
                color_discrete_sequence=['#4F46E5']
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Number of Incidents", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Row 4: Word Cloud and Insights
    st.markdown('<h2 class="section-header">Content Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud from processed tweets
        if 'processed_tweet' in filtered_df.columns:
            # Combine all processed tweets
            all_text = ' '.join(filtered_df['processed_tweet'].dropna())
            
            if all_text.strip():  # Check if there's text to display
                # Generate and display word cloud
                plt.figure(figsize=(10, 5))
                wordcloud = WordCloud(
                    width=800, height=400, 
                    background_color='white', 
                    colormap='viridis', 
                    max_words=100,
                    contour_width=1, 
                    contour_color='steelblue'
                ).generate(all_text)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(plt)
            else:
                st.info("Not enough text data to generate a word cloud.")
        else:
            st.info("Processed tweet data not available for word cloud generation.")
    
    with col2:
        # Key insights section
        st.markdown('<div class="insights-card">', unsafe_allow_html=True)
        st.subheader("Key Insights")
        
        # Calculate insights dynamically
        if len(filtered_df) > 0:
            # Most common location
            top_location = filtered_df['primary_location'].value_counts().index[0]
            top_location_count = filtered_df['primary_location'].value_counts()[0]
            top_location_pct = (top_location_count / len(filtered_df)) * 100
            
            # Most common crime category
            top_category = filtered_df['crime_category'].value_counts().index[0]
            top_category_count = filtered_df['crime_category'].value_counts()[0]
            top_category_pct = (top_category_count / len(filtered_df)) * 100
            
            # High security percentage
            high_pct = (len(filtered_df[filtered_df['security_level'] == 'high']) / len(filtered_df)) * 100
            
            # Display insights
            st.markdown(f"• **{top_location}** is the location with the highest number of incidents ({top_location_pct:.1f}%).")
            st.markdown(f"• **{top_category}** is the most common crime category ({top_category_pct:.1f}%).")
            st.markdown(f"• **{high_pct:.1f}%** of incidents are classified as high security level.")
            
            # Location-specific insights
            if selected_location != 'All':
                location_df = df[df['primary_location'] == selected_location]
                top_category_loc = location_df['crime_category'].value_counts().index[0]
                st.markdown(f"• In **{selected_location}**, the most common crime category is **{top_category_loc}**.")
            
            # Security level insights
            if 'day_of_week' in filtered_df.columns:
                high_security_df = filtered_df[filtered_df['security_level'] == 'high']
                if not high_security_df.empty:
                    high_security_day = high_security_df['day_of_week'].value_counts().index[0]
                    st.markdown(f"• **{high_security_day}** has the highest number of high security incidents.")
        else:
            st.write("No data available with the current filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 5: Data Table
    st.markdown('<h2 class="section-header">Detailed Data</h2>', unsafe_allow_html=True)
    
    # Display recent tweets with security classifications
    if 'Tweets' in filtered_df.columns:
        display_cols = ['Tweets', 'primary_location', 'crime_category', 'security_level']
        if 'date' in filtered_df.columns:
            display_cols = ['date'] + display_cols
        
        # Sort by date if available
        if 'date' in filtered_df.columns:
            display_df = filtered_df[display_cols].sort_values('date', ascending=False).head(10)
        else:
            display_df = filtered_df[display_cols].head(10)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for full data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="security_data_filtered.csv",
            mime="text/csv",
        )
else:
    st.error("Please run the main analysis script first to generate the processed data file.")
    
    # Instructions when data is not available
    st.markdown("""
    ## How to Generate the Dashboard Data
    
    1. First, run the main analysis script:
    ```
    python fixed-analysis-script.py --input nyc_crime_tweets.csv
    ```
    
    2. After the analysis completes, run this dashboard:
    ```
    streamlit run security_dashboard.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Created for Twitter Security Monitoring and Mapping Project")