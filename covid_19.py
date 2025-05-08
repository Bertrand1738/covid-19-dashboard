import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

class CovidDataAnalyzer:
    def __init__(self):
        self.base_url = "https://disease.sh/v3/covid-19"
        self.world_data = None
        self.countries_data = None
        self.historical_data = None
        self.summarizer = pipeline("summarization")
        
    def fetch_world_data(self):
        """Fetch global COVID-19 data"""
        response = requests.get(f"{self.base_url}/all")
        self.world_data = response.json()
        return self.world_data
    
    def fetch_countries_data(self):
        """Fetch COVID-19 data for all countries"""
        response = requests.get(f"{self.base_url}/countries")
        data = response.json()
        self.countries_data = pd.DataFrame(data)
        return self.countries_data
    
    def fetch_historical_data(self, days=30):
        """Fetch historical COVID-19 data for the specified number of days"""
        response = requests.get(f"{self.base_url}/historical/all?lastdays={days}")
        self.historical_data = response.json()
        return self.historical_data
    
    def clean_countries_data(self):
        """Clean and process countries data"""
        if self.countries_data is None:
            self.fetch_countries_data()
            
        # Extract country info
        self.countries_data['countryInfo'] = self.countries_data['countryInfo'].apply(lambda x: {} if pd.isna(x) else x)
        self.countries_data['flag'] = self.countries_data['countryInfo'].apply(lambda x: x.get('flag', '') if isinstance(x, dict) else '')
        self.countries_data['iso3'] = self.countries_data['countryInfo'].apply(lambda x: x.get('iso3', '') if isinstance(x, dict) else '')
        
        # Calculate per million metrics
        self.countries_data['activePer1M'] = self.countries_data['active'] / (self.countries_data['population'] / 1000000)
        
        # Calculate recovery and death rates
        self.countries_data['recoveryRate'] = self.countries_data['recovered'] / self.countries_data['cases'] * 100
        self.countries_data['fatalityRate'] = self.countries_data['deaths'] / self.countries_data['cases'] * 100
        
        return self.countries_data
    
    def process_historical_data(self):
        """Process historical data into a DataFrame"""
        if self.historical_data is None:
            self.fetch_historical_data()
            
        cases_df = pd.DataFrame(list(self.historical_data['cases'].items()), columns=['date', 'cases'])
        deaths_df = pd.DataFrame(list(self.historical_data['deaths'].items()), columns=['date', 'deaths'])
        recovered_df = pd.DataFrame(list(self.historical_data['recovered'].items()), columns=['date', 'recovered'])
        
        # Merge the dataframes
        historical_df = cases_df.merge(deaths_df, on='date').merge(recovered_df, on='date')
        
        # Convert date strings to datetime
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        # Calculate daily new cases, deaths, and recoveries
        historical_df['new_cases'] = historical_df['cases'].diff().fillna(0)
        historical_df['new_deaths'] = historical_df['deaths'].diff().fillna(0)
        historical_df['new_recovered'] = historical_df['recovered'].diff().fillna(0)
        
        return historical_df
    
    def cluster_countries(self, n_clusters=5):
        """Cluster countries based on COVID-19 metrics"""
        if self.countries_data is None:
            self.clean_countries_data()
            
        # Select features for clustering
        features = ['casesPer1M', 'deathsPer1M', 'recoveryRate', 'fatalityRate']
        
        # Remove rows with missing values
        cluster_data = self.countries_data[features].dropna()
        
        # Normalize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_data['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Merge cluster assignments back to the original dataframe
        self.countries_data = self.countries_data.join(cluster_data['cluster'], how='left')
        
        return self.countries_data
    
    def generate_summary(self, text):
        """Generate a summary of the provided text using a transformer model"""
        # Ensure text is not too long for the model
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length]
            
        summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    
    def create_summary_text(self):
        """Create a text summary of the COVID-19 data for AI summarization"""
        if self.world_data is None:
            self.fetch_world_data()
        
        if self.countries_data is None:
            self.clean_countries_data()
            
        # Create a text description of the data
        text = f"""
        Global COVID-19 Analysis as of {datetime.now().strftime('%Y-%m-%d')}:
        
        Total confirmed cases worldwide: {self.world_data['cases']:,}, with {self.world_data['todayCases']:,} new cases today.
        Total deaths: {self.world_data['deaths']:,}, with {self.world_data['todayDeaths']:,} new deaths today.
        Total recovered: {self.world_data['recovered']:,}.
        
        The countries with the highest number of cases are:
        {', '.join(self.countries_data.sort_values('cases', ascending=False)['country'].head(5).tolist())}.
        
        The countries with the highest death rates are:
        {', '.join(self.countries_data.sort_values('fatalityRate', ascending=False)['country'].head(5).tolist())}.
        
        The countries with the highest recovery rates are:
        {', '.join(self.countries_data.sort_values('recoveryRate', ascending=False)['country'].head(5).tolist())}.
        """
        
        return text
    
    def get_ai_summary(self):
        """Generate an AI summary of the COVID-19 data"""
        text = self.create_summary_text()
        summary = self.generate_summary(text)
        return summary