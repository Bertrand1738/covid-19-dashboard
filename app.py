import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from covid_19 import CovidDataAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .insight-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return CovidDataAnalyzer()

analyzer = get_analyzer()

# App title
st.markdown("<h1 class='main-header'>COVID-19 Global Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-time insights from Disease.sh API</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://disease.sh/assets/img/disease.png", width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Global Overview", "Country Analysis", "Trends & Forecasts", "AI Insights"])

# Load data
@st.cache_data(ttl=3600)
def load_world_data():
    return analyzer.fetch_world_data()

@st.cache_data(ttl=3600)
def load_countries_data():
    analyzer.fetch_countries_data()
    return analyzer.clean_countries_data()

@st.cache_data(ttl=3600)
def load_historical_data():
    analyzer.fetch_historical_data(days=90)
    return analyzer.process_historical_data()

@st.cache_data(ttl=3600)
def load_clustered_countries():
    countries_data = load_countries_data()
    return analyzer.cluster_countries()

# Global Overview Page
if page == "Global Overview":
    st.header("Global COVID-19 Situation")
    
    # Load data
    world_data = load_world_data()
    
    # Display global metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{world_data['cases']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Cases</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{world_data['deaths']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Deaths</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{world_data['recovered']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Recovered</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{world_data['active']:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Active Cases</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Today's numbers
    st.subheader("Today's Update")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("New Cases", f"{world_data['todayCases']:,}")
        
    with col2:
        st.metric("New Deaths", f"{world_data['todayDeaths']:,}")
        
    with col3:
        st.metric("New Recovered", f"{world_data['todayRecovered']:,}")
    
    # Historical data visualization
    st.markdown("---")
    st.subheader("Trends Over Time")
    
    historical_df = load_historical_data()
    
    # Create time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['cases'],
        mode='lines',
        name='Total Cases',
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['deaths'],
        mode='lines',
        name='Total Deaths',
        line=dict(color='#E53935', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['recovered'],
        mode='lines',
        name='Total Recovered',
        line=dict(color='#43A047', width=2)
    ))
    
    fig.update_layout(
        title='Cumulative COVID-19 Cases Over Time',
        xaxis_title='Date',
        yaxis_title='Count',
        legend_title='Metric',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily new cases
    st.subheader("Daily New Cases")
    
    fig = px.bar(
        historical_df,
        x='date',
        y='new_cases',
        title='Daily New COVID-19 Cases',
        labels={'new_cases': 'New Cases', 'date': 'Date'},
        color_discrete_sequence=['#1E88E5']
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Country Analysis Page
elif page == "Country Analysis":
    st.header("Country-by-Country Analysis")
    
    # Load data
    countries_data = load_countries_data()
    
    # Country selector
    all_countries = countries_data['country'].tolist()
    selected_countries = st.multiselect(
        "Select countries to compare:",
        all_countries,
        default=["USA", "India", "Brazil", "UK", "Russia"][:3]
    )
    
    # Display country comparison if countries are selected
    if selected_countries:
        # Filter data for selected countries
        filtered_data = countries_data[countries_data['country'].isin(selected_countries)]
        
        # Comparison metrics
        st.subheader("Key Metrics Comparison")
        
        # Create comparison bar charts
        metrics = ['cases', 'deaths', 'recovered', 'active']
        metric_labels = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases']
        
        # Create tabs for different metrics
        tabs = st.tabs(metric_labels)
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            with tabs[i]:
                fig = px.bar(
                    filtered_data,
                    x='country',
                    y=metric,
                    title=f'{label} by Country',
                    color='country',
                    labels={metric: label, 'country': 'Country'},
                    height=400
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        
        # Per million comparison
        st.subheader("Per Million Population Metrics")
        
        per_million_metrics = ['casesPerOneMillion', 'casesPerOneMillion', 'testsPer1M']
        per_million_labels = ['Cases per Million', 'Deaths per Million', 'Tests per Million']
        
        # Create tabs for different per million metrics
        tabs = st.tabs(per_million_labels)
        
        for i, (metric, label) in enumerate(zip(per_million_metrics, per_million_labels)):
            with tabs[i]:
                fig = px.bar(
                    filtered_data,
                    x='country',
                    y='casesPerOneMillion',
                    title=f'{label} by Country',
                    color='country',
                    labels={metric: label, 'country': 'Country'},
                    height=400
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        
        # Rate comparison
        st.subheader("Rate Analysis")
        
        rate_metrics = ['recoveryRate', 'fatalityRate']
        rate_labels = ['Recovery Rate (%)', 'Fatality Rate (%)']
        
        # Create tabs for different rate metrics
        tabs = st.tabs(rate_labels)
        
        for i, (metric, label) in enumerate(zip(rate_metrics, rate_labels)):
            with tabs[i]:
                fig = px.bar(
                    filtered_data,
                    x='country',
                    y=metric,
                    title=f'{label} by Country',
                    color='country',
                    labels={metric: label, 'country': 'Country'},
                    height=400
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
    
    # World map visualization
    st.markdown("---")
    st.subheader("Global COVID-19 Map")
    
    # Select metric for map
    map_metric = st.selectbox(
        "Select metric to visualize:",
        options=['cases', 'deaths', 'recovered', 'active', 'casesPerOneMillion', 'casesPerOneMillion', 'recoveryRate', 'fatalityRate'],
        format_func=lambda x: {
            'cases': 'Total Cases',
            'deaths': 'Total Deaths',
            'recovered': 'Total Recovered',
            'active': 'Active Cases',
            'casesPerOneMillion': 'Cases per Million',
            'casesPerOneMillion': 'Deaths per Million',
            'recoveryRate': 'Recovery Rate (%)',
            'fatalityRate': 'Fatality Rate (%)'
        }[x]
    )
    
    # Create choropleth map
    fig = px.choropleth(
        countries_data,
        locations="iso3",
        color=map_metric,
        hover_name="country",
        hover_data=[map_metric, 'cases', 'deaths', 'recovered'],
        projection="natural earth",
        color_continuous_scale=px.colors.sequential.Blues,
        title=f"Global {map_metric} Distribution"
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Country clustering
    st.markdown("---")
    st.subheader("Country Clustering Analysis")
    st.write("Countries clustered based on COVID-19 metrics using K-means algorithm")
    
    # Load clustered data
    clustered_countries = load_clustered_countries()
    
    # Create scatter plot of clusters
    fig = px.scatter(
        clustered_countries.dropna(subset=['cluster']),
        x='casesPerOneMillion',
        y='casesPerOneMillion',
        color='cluster',
        hover_name='country',
        size='cases',
        size_max=50,
        title='Country Clusters based on Cases and Deaths per Million',
        labels={
            'casesPerOneMillion': 'Cases per Million',
            'casesPerOneMillion': 'Deaths per Million',
            'cluster': 'Cluster'
        }
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster characteristics
    if not clustered_countries[clustered_countries['cluster'].notna()].empty:
        st.subheader("Cluster Characteristics")
        
        # Calculate cluster averages
        cluster_stats = clustered_countries.groupby('cluster').agg({
            'casesPerOneMillion': 'mean',
            'casesPerOneMillion': 'mean',
            'recoveryRate': 'mean',
            'fatalityRate': 'mean',
            'country': 'count'
        }).rename(columns={'country': 'count'}).reset_index()
        
        # Display cluster statistics
        st.dataframe(cluster_stats.style.format({
            'casesPerOneMillion': '{:.2f}',
            'casesPerOneMillion': '{:.2f}',
            'recoveryRate': '{:.2f}',
            'fatalityRate': '{:.2f}',
            'count': '{:.0f}'
        }))
        
        # Show countries in each cluster
        st.subheader("Countries by Cluster")
        
        # Create tabs for each cluster
        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(len(cluster_stats))])
        
        for i, tab in enumerate(cluster_tabs):
            with tab:
                cluster_countries = clustered_countries[clustered_countries['cluster'] == i]
                st.write(f"**Number of countries:** {len(cluster_countries)}")
                st.dataframe(
                    cluster_countries[['country', 'casesPerOneMillion', 'casesPerOneMillion', 'recoveryRate', 'fatalityRate']]
                    .sort_values('casesPerOneMillion', ascending=False)
                    .reset_index(drop=True)
                    .style.format({
                        'casesPerOneMillion': '{:.2f}',
                        'casesPerOneMillion': '{:.2f}',
                        'recoveryRate': '{:.2f}',
                        'fatalityRate': '{:.2f}'
                    })
                )

# Trends & Forecasts Page
elif page == "Trends & Forecasts":
    st.header("COVID-19 Trends and Forecasts")
    
    # Load historical data
    historical_df = load_historical_data()
    
    # Time series decomposition
    st.subheader("Time Series Analysis")
    st.write("Analyzing COVID-19 trends over time to identify patterns")
    
    # Create tabs for different metrics
    trend_tabs = st.tabs(["Cases Trend", "Deaths Trend", "Recovered Trend"])
    
    with trend_tabs[0]:
        # Plot cases trend
        fig = px.line(
            historical_df,
            x='date',
            y='new_cases',
            title='Daily New Cases Trend',
            labels={'new_cases': 'New Cases', 'date': 'Date'},
            template='plotly_white'
        )
        
        # Add 7-day moving average
        historical_df['cases_7day_avg'] = historical_df['new_cases'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=historical_df['date'],
                y=historical_df['cases_7day_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate growth rate
        historical_df['cases_growth_rate'] = historical_df['new_cases'].pct_change() * 100
        
        # Plot growth rate
        fig = px.line(
            historical_df.dropna(),
            x='date',
            y='cases_growth_rate',
            title='Daily Cases Growth Rate (%)',
            labels={'cases_growth_rate': 'Growth Rate (%)', 'date': 'Date'},
            template='plotly_white'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with trend_tabs[1]:
        # Plot deaths trend
        fig = px.line(
            historical_df,
            x='date',
            y='new_deaths',
            title='Daily New Deaths Trend',
            labels={'new_deaths': 'New Deaths', 'date': 'Date'},
            template='plotly_white'
        )
        
        # Add 7-day moving average
        historical_df['deaths_7day_avg'] = historical_df['new_deaths'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=historical_df['date'],
                y=historical_df['deaths_7day_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with trend_tabs[2]:
        # Plot recovered trend
        fig = px.line(
            historical_df,
            x='date',
            y='new_recovered',
            title='Daily New Recoveries Trend',
            labels={'new_recovered': 'New Recoveries', 'date': 'Date'},
            template='plotly_white'
        )
        
        # Add 7-day moving average
        historical_df['recovered_7day_avg'] = historical_df['new_recovered'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=historical_df['date'],
                y=historical_df['recovered_7day_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Simple forecasting
    st.markdown("---")
    st.subheader("Simple Forecast")
    st.write("A simple forecast of future COVID-19 cases based on recent trends")
    
    # Prepare data for forecasting
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Use only the last 30 days for forecasting
    forecast_df = historical_df.tail(30).copy()
    forecast_df['day_index'] = range(len(forecast_df))
    
    # Train a simple linear regression model
    X = forecast_df['day_index'].values.reshape(-1, 1)
    y_cases = forecast_df['cases'].values
    
    model = LinearRegression()
    model.fit(X, y_cases)
    
    # Create forecast for next 14 days
    last_day = forecast_df['day_index'].max()
    future_days = np.array(range(last_day + 1, last_day + 15)).reshape(-1, 1)
    
    future_cases = model.predict(future_days)
    
    # Create forecast dataframe
    future_dates = pd.date_range(
        start=forecast_df['date'].max() + pd.Timedelta(days=1),
        periods=14,
        freq='D'
    )
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'cases': future_cases,
        'type': 'Forecast'
    })
    
    # Combine actual and forecast data
    plot_df = pd.concat([
        forecast_df[['date', 'cases']].assign(type='Actual'),
        future_df
    ])
    
    # Plot forecast
    fig = px.line(
        plot_df,
        x='date',
        y='cases',
        color='type',
        title='14-Day Cases Forecast',
        labels={'cases': 'Total Cases', 'date': 'Date', 'type': 'Data Type'},
        template='plotly_white',
        color_discrete_map={'Actual': 'blue', 'Forecast': 'red'}
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.info("""
    **Disclaimer:** This is a simple linear forecast for illustrative purposes only. 
    It does not account for many factors that influence COVID-19 spread such as 
    vaccination rates, public health measures, variants, etc.
    """)
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("Correlation Analysis")
    
    # Calculate correlations
    corr_df = historical_df[['cases', 'deaths', 'recovered', 'new_cases', 'new_deaths', 'new_recovered']].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Correlation Between COVID-19 Metrics',
        labels={'color': 'Correlation'}
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# AI Insights Page
elif page == "AI Insights":
    st.header("AI-Generated COVID-19 Insights")
    
    # Load data
    world_data = load_world_data()
    countries_data = load_countries_data()
    historical_df = load_historical_data()
    
    # Generate AI summary
    st.subheader("AI Summary of Global Situation")
    
    with st.spinner("Generating AI summary..."):
        try:
            # Get AI-generated summary
            summary = analyzer.get_ai_summary()
            
            # Display the summary in an insight box
            st.markdown(f"<div class='insight-box'>{summary}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating AI summary: {str(e)}")
            st.markdown("""
            <div class='insight-box'>
            Based on the current global COVID-19 data, the pandemic continues to affect countries worldwide with varying impact.
            The countries with the highest case counts tend to be those with larger populations or those experiencing recent outbreaks.
            Recovery rates and fatality rates vary significantly between countries, likely due to differences in healthcare systems,
            population demographics, and public health measures.
            </div>
            """, unsafe_allow_html=True)
    
    # AI-driven country risk assessment
    st.markdown("---")
    st.subheader("Country Risk Assessment")
    st.write("AI-based classification of countries by COVID-19 risk level")
    
    # Calculate risk score based on multiple factors
    countries_data['risk_score'] = (
        countries_data['casesPerOneMillion'] / countries_data['casesPerOneMillion'].max() * 0.4 +
        countries_data['casesPerOneMillion'] / countries_data['casesPerOneMillion'].max() * 0.4 +
        (1 - countries_data['recoveryRate'] / 100) * 0.2
    )
    
    # Classify countries into risk categories
    def risk_category(score):
        if score < 0.3:
            return "Low Risk"
        elif score < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    countries_data['risk_category'] = countries_data['risk_score'].apply(risk_category)
    
    # Create risk category counts
    risk_counts = countries_data['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    
    # Plot risk categories
    fig = px.pie(
        risk_counts,
        values='Count',
        names='Risk Category',
        title='Countries by COVID-19 Risk Level',
        color='Risk Category',
        color_discrete_map={
            'Low Risk': '#4CAF50',
            'Medium Risk': '#FFC107',
            'High Risk': '#F44336'
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top high-risk countries
    high_risk = countries_data[countries_data['risk_category'] == 'High Risk'].sort_values('risk_score', ascending=False)
    
    if not high_risk.empty:
        st.subheader("Top 10 Highest Risk Countries")
        st.dataframe(
            high_risk[['country', 'casesPerOneMillion', 'casesPerOneMillion', 'recoveryRate', 'risk_score']]
            .head(10)
            .reset_index(drop=True)
            .style.format({
                'casesPerOneMillion': '{:.2f}',
                'casesPerOneMillion': '{:.2f}',
                'recoveryRate': '{:.2f}',
                'risk_score': '{:.4f}'
            })
        )
    
    # AI-driven pattern detection
    st.markdown("---")
    st.subheader("Pattern Detection")
    st.write("AI analysis to detect patterns in COVID-19 data")
    
    # Calculate daily case acceleration (2nd derivative)
    historical_df['case_acceleration'] = historical_df['new_cases'].diff()
    
    # Detect significant changes in trend
    threshold = historical_df['case_acceleration'].std() * 2
    historical_df['significant_change'] = historical_df['case_acceleration'].abs() > threshold
    
    # Find dates with significant changes
    significant_dates = historical_df[historical_df['significant_change']].copy()
    
    # Plot with highlighted significant changes
    fig = go.Figure()
    
    # Add line for all data
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['new_cases'],
        mode='lines',
        name='Daily New Cases',
        line=dict(color='blue', width=2)
    ))
    
    # Add markers for significant changes
    if not significant_dates.empty:
        fig.add_trace(go.Scatter(
            x=significant_dates['date'],
            y=significant_dates['new_cases'],
            mode='markers',
            name='Significant Changes',
            marker=dict(color='red', size=10)
        ))
    
    fig.update_layout(
        title='AI-Detected Significant Changes in COVID-19 Trends',
        xaxis_title='Date',
        yaxis_title='New Cases',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the significant changes
    if not significant_dates.empty:
        st.subheader("Detected Pattern Changes")
        st.write("The AI has detected significant changes in COVID-19 trends on these dates:")
        
        for _, row in significant_dates.iterrows():
            direction = "increase" if row['case_acceleration'] > 0 else "decrease"
            st.markdown(f"""
            <div class='insight-box'>
            <strong>{row['date'].strftime('%B %d, %Y')}</strong>: Significant {direction} in the rate of new cases.
            Daily new cases: {int(row['new_cases']):,}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant pattern changes detected in the current data.")