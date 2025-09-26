import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast

# Install required packages:
# pip install vaderSentiment textblob

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing vaderSentiment...")
    os.system("pip install vaderSentiment")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from textblob import TextBlob
except ImportError:
    print("Installing textblob...")
    os.system("pip install textblob")
    from textblob import TextBlob


class CommunityProfiler:
    """
    Advanced community profiling with sentiment analysis and temporal tracking.
    """
    
    def __init__(self, posts_file='data/processed/posts.csv', graph_file='data/graphs/ticker_network.gexf'):
        """Initialize profiler with data sources."""
        print("📊 Loading data for community profiling...")
        
        # Check if files exist
        if not os.path.exists(posts_file):
            raise FileNotFoundError(f"Posts file not found: {posts_file}")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        # Load posts
        try:
            self.posts = pd.read_csv(posts_file, dtype={'user_id': str})
            print(f"   Posts loaded: {len(self.posts):,}")
        except Exception as e:
            raise Exception(f"Error loading posts: {e}")
        
        # Load graph with communities
        try:
            self.G = nx.read_gexf(graph_file)
            print(f"   Graph loaded: {self.G.number_of_nodes():,} nodes")
        except Exception as e:
            raise Exception(f"Error loading graph: {e}")
        
        # Check if graph has community attribute
        sample_nodes = list(self.G.nodes())[:5]
        community_attrs = [self.G.nodes[node].get('community') for node in sample_nodes]
        
        if all(attr is None for attr in community_attrs):
            raise ValueError("Graph nodes don't have 'community' attribute. Run community detection first.")
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Prepare data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean data for analysis."""
        print("🔧 Preparing data...")
        
        # Parse dates - handle different date formats
        try:
            self.posts['created_at'] = pd.to_datetime(self.posts['created_at'], errors='coerce')
            self.posts = self.posts.dropna(subset=['created_at'])
        except Exception as e:
            print(f"Warning: Issue with date parsing: {e}")
            # If no valid dates, create dummy dates
            if 'created_at' not in self.posts.columns or self.posts['created_at'].isna().all():
                print("Creating dummy dates for analysis...")
                self.posts['created_at'] = pd.date_range(start='2023-01-01', periods=len(self.posts), freq='H')
        
        # Parse tickers safely
        def safe_parse_tickers(value):
            if pd.isna(value) or value == "" or value == "[]":
                return []
            try:
                if isinstance(value, str):
                    # Handle different formats: ["TSLA"] or "TSLA,AAPL" or [["TSLA"]]
                    if value.startswith('[') and value.endswith(']'):
                        tickers = ast.literal_eval(value)
                        if isinstance(tickers, list):
                            # Flatten nested lists
                            result = []
                            for item in tickers:
                                if isinstance(item, list):
                                    result.extend([str(t).strip().upper() for t in item if t])
                                elif isinstance(item, str):
                                    result.append(str(item).strip().upper())
                            return result
                    else:
                        # Simple comma-separated format
                        return [t.strip().upper() for t in value.split(',') if t.strip()]
                return []
            except Exception as e:
                print(f"Warning: Error parsing tickers '{value}': {e}")
                return []
        
        # Check if tickers column exists
        if 'tickers' not in self.posts.columns:
            print("Warning: 'tickers' column not found. Creating empty ticker lists.")
            self.posts['tickers_list'] = [[] for _ in range(len(self.posts))]
        else:
            self.posts['tickers_list'] = self.posts['tickers'].apply(safe_parse_tickers)
        
        # Get user-community mapping - handle different attribute names
        community_attr_names = ['community', 'louvain_community', 'spectral_community']
        self.user_community = {}
        
        for attr_name in community_attr_names:
            temp_mapping = nx.get_node_attributes(self.G, attr_name)
            if temp_mapping:
                self.user_community = temp_mapping
                print(f"   Using '{attr_name}' attribute for communities")
                break
        
        if not self.user_community:
            raise ValueError("No community attributes found in graph. Available attributes: " + 
                           str(list(next(iter(self.G.nodes(data=True)))[1].keys()) if self.G.nodes() else []))
        
        # Convert community IDs to strings for consistency
        self.user_community = {str(k): str(v) for k, v in self.user_community.items()}
        
        # Filter posts to users in communities
        original_posts = len(self.posts)
        self.posts = self.posts[self.posts['user_id'].astype(str).isin(self.user_community.keys())].copy()
        
        print(f"   Posts after filtering: {len(self.posts):,} (removed {original_posts - len(self.posts):,})")
        
        if len(self.posts) == 0:
            raise ValueError("No posts found for users in communities. Check user_id matching.")
        
        print(f"   Date range: {self.posts['created_at'].min()} to {self.posts['created_at'].max()}")
        print(f"   Users with communities: {len(self.user_community):,}")
        print(f"   Unique communities: {len(set(self.user_community.values()))}")
    
    def compute_sentiment_scores(self, method='vader'):
        """
        Compute sentiment scores for all posts.
        """
        print(f"🎭 Computing sentiment scores using {method}...")
        
        if 'text' not in self.posts.columns:
            print("Warning: 'text' column not found. Creating neutral sentiment scores.")
            self.posts['sentiment'] = 0.0
            return self.posts['sentiment']
        
        if method == 'vader':
            # VADER sentiment (good for social media)
            def get_sentiment(text):
                if pd.isna(text) or str(text).strip() == '':
                    return 0.0
                try:
                    scores = self.vader_analyzer.polarity_scores(str(text))
                    return scores['compound']  # Range: -1 (negative) to +1 (positive)
                except:
                    return 0.0
            
        elif method == 'textblob':
            # TextBlob sentiment
            def get_sentiment(text):
                if pd.isna(text) or str(text).strip() == '':
                    return 0.0
                try:
                    blob = TextBlob(str(text))
                    return blob.sentiment.polarity  # Range: -1 to +1
                except:
                    return 0.0
            
        else:
            raise ValueError("Method must be 'vader' or 'textblob'")
        
        # Compute sentiment with progress bar
        tqdm.pandas(desc="Analyzing sentiment")
        self.posts['sentiment'] = self.posts['text'].progress_apply(get_sentiment)
        
        print(f"✅ Sentiment computed. Average sentiment: {self.posts['sentiment'].mean():.3f}")
        
        return self.posts['sentiment']
    
    def profile_communities(self, top_tickers_count=10, min_posts=5):
        """
        Create comprehensive community profiles.
        """
        print("👥 Creating community profiles...")
        
        # Ensure sentiment is computed
        if 'sentiment' not in self.posts.columns:
            self.compute_sentiment_scores()
        
        community_profiles = {}
        
        # Get unique communities
        communities = set(self.user_community.values())
        print(f"   Analyzing {len(communities)} communities...")
        
        for comm_id in tqdm(communities, desc="Processing communities"):
            # Get users in this community
            community_users = [user for user, cid in self.user_community.items() if cid == comm_id]
            
            # Get posts from community users
            community_posts = self.posts[self.posts['user_id'].astype(str).isin(community_users)].copy()
            
            if len(community_posts) < min_posts:
                continue  # Skip small communities
            
            # Basic stats
            num_users = len(community_users)
            num_posts = len(community_posts)
            
            # Ticker analysis
            all_tickers = []
            for ticker_list in community_posts['tickers_list']:
                if isinstance(ticker_list, list):
                    all_tickers.extend(ticker_list)
            
            ticker_counter = Counter(all_tickers)
            top_tickers = [ticker for ticker, count in ticker_counter.most_common(top_tickers_count)]
            
            # Sentiment analysis
            sentiment_scores = community_posts['sentiment'].dropna()
            avg_sentiment = float(sentiment_scores.mean()) if len(sentiment_scores) > 0 else 0.0
            sentiment_std = float(sentiment_scores.std()) if len(sentiment_scores) > 0 else 0.0
            
            # Classify sentiment
            if avg_sentiment > 0.1:
                sentiment_label = "Bullish"
            elif avg_sentiment < -0.1:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            # Activity patterns - handle potential date issues
            try:
                posts_per_day = community_posts.groupby(community_posts['created_at'].dt.date).size()
                avg_posts_per_day = float(posts_per_day.mean()) if len(posts_per_day) > 0 else 0.0
                peak_day = posts_per_day.idxmax() if len(posts_per_day) > 0 else None
                peak_posts = int(posts_per_day.max()) if len(posts_per_day) > 0 else 0
            except Exception as e:
                print(f"Warning: Date analysis failed for community {comm_id}: {e}")
                avg_posts_per_day = 0.0
                peak_day = None
                peak_posts = 0
            
            # User engagement
            posts_per_user = community_posts['user_id'].value_counts()
            avg_posts_per_user = float(posts_per_user.mean()) if len(posts_per_user) > 0 else 0.0
            top_user_posts = int(posts_per_user.max()) if len(posts_per_user) > 0 else 0
            
            # Community label generation
            community_label = self.generate_community_label(top_tickers, sentiment_label, num_users)
            
            # Store profile
            community_profiles[str(comm_id)] = {
                'label': community_label,
                'num_users': int(num_users),
                'num_posts': int(num_posts),
                'top_tickers': top_tickers,
                'top_ticker_counts': dict(ticker_counter.most_common(top_tickers_count)),
                'avg_sentiment': float(avg_sentiment),
                'sentiment_std': float(sentiment_std),
                'sentiment_label': sentiment_label,
                'avg_posts_per_day': float(avg_posts_per_day),
                'avg_posts_per_user': float(avg_posts_per_user),
                'peak_activity': {
                    'date': str(peak_day) if peak_day else None,
                    'posts': int(peak_posts)
                },
                'activity_score': float(avg_posts_per_day * num_users),  # Composite activity score
                'engagement_score': float(avg_posts_per_user),
                'date_range': {
                    'start': str(community_posts['created_at'].min().date()) if not community_posts['created_at'].isna().all() else None,
                    'end': str(community_posts['created_at'].max().date()) if not community_posts['created_at'].isna().all() else None
                }
            }
        
        self.community_profiles = community_profiles
        print(f"✅ Created profiles for {len(community_profiles)} communities")
        
        return community_profiles
    
    def generate_community_label(self, top_tickers, sentiment_label, num_users):
        """
        Generate descriptive labels for communities.
        """
        if not top_tickers:
            return f"{sentiment_label} Community ({num_users} users)"
        
        # Special case labels for popular meme stocks
        meme_stocks = {'GME', 'AMC', 'TSLA', 'PLTR', 'BB', 'NOK', 'CLOV', 'WISH'}
        crypto_tickers = {'BTC', 'ETH', 'DOGE', 'ADA', 'SOL', 'MATIC'}
        
        primary_ticker = top_tickers[0]
        
        # Check if it's a meme stock community
        if primary_ticker in meme_stocks:
            if sentiment_label == "Bullish":
                return f"${primary_ticker} Bulls ({num_users} users)"
            elif sentiment_label == "Bearish":
                return f"${primary_ticker} Bears ({num_users} users)"
            else:
                return f"${primary_ticker} Traders ({num_users} users)"
        
        # Check if it's crypto
        elif primary_ticker in crypto_tickers:
            return f"Crypto {sentiment_label} - ${primary_ticker} Focus ({num_users} users)"
        
        # Multi-ticker communities
        elif len(top_tickers) > 1:
            return f"Multi-Ticker {sentiment_label} ({', '.join(top_tickers[:3])}) ({num_users} users)"
        
        # Default label
        else:
            return f"${primary_ticker} {sentiment_label} ({num_users} users)"
    
    def analyze_community_evolution(self, time_window='7D'):
        """
        Track how communities evolve over time.
        """
        print(f"📈 Analyzing community evolution (window: {time_window})...")
        
        if 'sentiment' not in self.posts.columns:
            self.compute_sentiment_scores()
        
        # Check if we have valid dates
        if self.posts['created_at'].isna().all():
            print("Warning: No valid dates found. Skipping temporal analysis.")
            self.evolution_data = pd.DataFrame()
            return self.evolution_data
        
        # Create time periods
        try:
            date_range = pd.date_range(
                start=self.posts['created_at'].min(),
                end=self.posts['created_at'].max(),
                freq=time_window
            )
        except Exception as e:
            print(f"Warning: Could not create date range: {e}")
            self.evolution_data = pd.DataFrame()
            return self.evolution_data
        
        evolution_data = []
        
        for community_id in tqdm(set(self.user_community.values()), desc="Tracking evolution"):
            community_users = [user for user, cid in self.user_community.items() if cid == community_id]
            community_posts = self.posts[self.posts['user_id'].astype(str).isin(community_users)].copy()
            
            if len(community_posts) < 10:  # Skip small communities
                continue
            
            # Group posts by time periods
            try:
                community_posts['period'] = pd.cut(
                    community_posts['created_at'], 
                    bins=date_range, 
                    labels=date_range[:-1],
                    include_lowest=True
                )
                
                period_stats = community_posts.groupby('period').agg({
                    'sentiment': ['mean', 'count'],
                    'user_id': 'nunique'
                }).reset_index()
                
                period_stats.columns = ['period', 'avg_sentiment', 'num_posts', 'active_users']
                period_stats = period_stats.dropna()
                
                # Calculate ticker trends for each period
                for _, row in period_stats.iterrows():
                    period = row['period']
                    period_posts = community_posts[community_posts['period'] == period]
                    
                    # Get top tickers for this period
                    period_tickers = []
                    for ticker_list in period_posts['tickers_list']:
                        if isinstance(ticker_list, list):
                            period_tickers.extend(ticker_list)
                    
                    top_period_tickers = [t for t, c in Counter(period_tickers).most_common(5)]
                    
                    evolution_data.append({
                        'community': str(community_id),
                        'period': period,
                        'date': period.strftime('%Y-%m-%d'),
                        'num_posts': int(row['num_posts']),
                        'active_users': int(row['active_users']),
                        'avg_sentiment': float(row['avg_sentiment']),
                        'top_tickers': top_period_tickers,
                        'activity_score': float(row['num_posts'] * row['active_users'])
                    })
            except Exception as e:
                print(f"Warning: Evolution analysis failed for community {community_id}: {e}")
                continue
        
        evolution_df = pd.DataFrame(evolution_data)
        self.evolution_data = evolution_df
        
        print(f"✅ Evolution tracking complete: {len(evolution_df)} time periods analyzed")
        return evolution_df
    
    def identify_trending_communities(self, lookback_days=7, min_growth_rate=2.0):
        """
        Identify communities with trending activity.
        """
        print(f"🔥 Identifying trending communities (last {lookback_days} days, min growth: {min_growth_rate}x)...")
        
        if not hasattr(self, 'evolution_data') or self.evolution_data.empty:
            print("Warning: No evolution data available. Running evolution analysis first...")
            self.analyze_community_evolution()
            
        if self.evolution_data.empty:
            print("Warning: No evolution data available. Skipping trending analysis.")
            return []
        
        try:
            recent_date = self.posts['created_at'].max()
            cutoff_date = recent_date - timedelta(days=lookback_days)
        except:
            print("Warning: Cannot determine recent date. Using relative analysis.")
            # Use relative analysis instead
            sorted_dates = sorted(pd.to_datetime(self.evolution_data['date']))
            if len(sorted_dates) < 2:
                return []
            cutoff_date = sorted_dates[len(sorted_dates)//2]  # Use middle date as cutoff
            recent_date = sorted_dates[-1]
        
        trending_communities = []
        
        for community_id in self.evolution_data['community'].unique():
            comm_data = self.evolution_data[self.evolution_data['community'] == community_id].copy()
            comm_data['date'] = pd.to_datetime(comm_data['date'])
            
            # Get recent and baseline periods
            recent_data = comm_data[comm_data['date'] >= cutoff_date]
            baseline_data = comm_data[comm_data['date'] < cutoff_date]
            
            if len(recent_data) == 0 or len(baseline_data) == 0:
                continue
            
            recent_activity = recent_data['activity_score'].mean()
            baseline_activity = baseline_data['activity_score'].mean()
            
            if baseline_activity > 0:
                growth_rate = recent_activity / baseline_activity
                if growth_rate >= min_growth_rate:
                    # Flatten ticker lists
                    recent_tickers = []
                    for tickers in recent_data['top_tickers']:
                        if isinstance(tickers, list):
                            recent_tickers.extend(tickers)
                    
                    trending_communities.append({
                        'community': str(community_id),
                        'growth_rate': float(growth_rate),
                        'recent_activity': float(recent_activity),
                        'baseline_activity': float(baseline_activity),
                        'recent_sentiment': float(recent_data['avg_sentiment'].mean()),
                        'recent_top_tickers': list(set(recent_tickers[:5]))
                    })
        
        # Sort by growth rate
        trending_communities = sorted(trending_communities, key=lambda x: x['growth_rate'], reverse=True)
        
        print(f"📈 Found {len(trending_communities)} trending communities")
        
        return trending_communities
    
    def create_community_dashboard_data(self):
        """
        Prepare data for dashboard visualization.
        """
        print("📊 Preparing dashboard data...")
        
        if not hasattr(self, 'community_profiles') or not self.community_profiles:
            self.profile_communities()
        
        if not self.community_profiles:
            print("Warning: No community profiles available.")
            return {}
        
        # Summary statistics
        try:
            profiles_values = list(self.community_profiles.values())
            
            most_active = max(self.community_profiles.items(), key=lambda x: x[1]['activity_score'])
            most_bullish = max(self.community_profiles.items(), key=lambda x: x[1]['avg_sentiment'])
            most_bearish = min(self.community_profiles.items(), key=lambda x: x[1]['avg_sentiment'])
            
            summary_stats = {
                'total_communities': len(self.community_profiles),
                'total_users': sum(profile['num_users'] for profile in profiles_values),
                'total_posts': sum(profile['num_posts'] for profile in profiles_values),
                'avg_sentiment': float(np.mean([profile['avg_sentiment'] for profile in profiles_values])),
                'most_active_community': {
                    'id': most_active[0],
                    'label': most_active[1]['label'],
                    'score': most_active[1]['activity_score']
                },
                'most_bullish_community': {
                    'id': most_bullish[0],
                    'label': most_bullish[1]['label'],
                    'sentiment': most_bullish[1]['avg_sentiment']
                },
                'most_bearish_community': {
                    'id': most_bearish[0],
                    'label': most_bearish[1]['label'],
                    'sentiment': most_bearish[1]['avg_sentiment']
                }
            }
            
            # Top tickers across all communities
            all_tickers = Counter()
            for profile in profiles_values:
                for ticker, count in profile['top_ticker_counts'].items():
                    all_tickers[ticker] += count
            
            summary_stats['top_global_tickers'] = dict(all_tickers.most_common(20))
            
            # Community size distribution
            size_distribution = Counter()
            for profile in profiles_values:
                if profile['num_users'] < 10:
                    size_distribution['Small (< 10 users)'] += 1
                elif profile['num_users'] < 50:
                    size_distribution['Medium (10-50 users)'] += 1
                elif profile['num_users'] < 200:
                    size_distribution['Large (50-200 users)'] += 1
                else:
                    size_distribution['Very Large (200+ users)'] += 1
            
            summary_stats['size_distribution'] = dict(size_distribution)
            
        except Exception as e:
            print(f"Warning: Error creating dashboard data: {e}")
            summary_stats = {
                'total_communities': len(self.community_profiles),
                'error': str(e)
            }
        
        return summary_stats
    
    def visualize_community_profiles(self, out_dir='outputs'):
        """
        Create comprehensive visualization of community profiles.
        """
        print("🎨 Creating community profile visualizations...")
        os.makedirs(out_dir, exist_ok=True)
        
        if not hasattr(self, 'community_profiles') or not self.community_profiles:
            self.profile_communities()
        
        if not self.community_profiles:
            print("Warning: No community profiles to visualize.")
            return
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Community Profiles Analysis', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            communities = list(self.community_profiles.keys())
            profiles = list(self.community_profiles.values())
            
            # 1. Community sizes
            sizes = [p['num_users'] for p in profiles]
            if sizes:
                axes[0, 0].hist(sizes, bins=min(20, len(set(sizes))), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Community Size (Users)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Community Size Distribution')
            if max(sizes) > 1:
                axes[0, 0].set_yscale('log')
            
            # 2. Sentiment distribution
            sentiments = [p['avg_sentiment'] for p in profiles]
            if sentiments:
                axes[0, 1].hist(sentiments, bins=min(20, len(set(sentiments))), alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('Average Sentiment')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sentiment Distribution')
            axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Neutral')
            axes[0, 1].legend()
            
            # 3. Activity vs Sentiment
            activities = [p['activity_score'] for p in profiles]
            if activities and sentiments:
                axes[0, 2].scatter(activities, sentiments, alpha=0.6, s=50)
                axes[0, 2].set_xlabel('Activity Score')
                axes[0, 2].set_ylabel('Average Sentiment')
                axes[0, 2].set_title('Activity vs Sentiment')
                if max(activities) > 1:
                    axes[0, 2].set_xscale('log')
            
            # 4. Top communities by activity
            if activities:
                top_active = sorted(zip(communities, activities, [p['label'] for p in profiles]), 
                                   key=lambda x: x[1], reverse=True)[:15]
                
                comm_labels = [label[:30] + '...' if len(label) > 30 else label for _, _, label in top_active]
                comm_activities = [activity for _, activity, _ in top_active]
                
                y_pos = np.arange(len(comm_labels))
                axes[1, 0].barh(y_pos, comm_activities, color='lightgreen', alpha=0.7)
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(comm_labels, fontsize=8)
                axes[1, 0].set_xlabel('Activity Score')
                axes[1, 0].set_title('Top 15 Most Active Communities')
            
            # 5. Sentiment labels distribution
            sentiment_labels = [p['sentiment_label'] for p in profiles]
            if sentiment_labels:
                sentiment_counts = Counter(sentiment_labels)
                axes[1, 1].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('Community Sentiment Distribution')
            
            # 6. Posts per user distribution
            posts_per_user = [p['avg_posts_per_user'] for p in profiles]
            if posts_per_user:
                axes[1, 2].hist(posts_per_user, bins=min(20, len(set(posts_per_user))), alpha=0.7, color='gold', edgecolor='black')
            axes[1, 2].set_xlabel('Average Posts per User')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('User Engagement Distribution')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(out_dir, 'community_profiles_analysis.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f"📊 Community profiles plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            plt.close('all')  # Close any open figures
    
    def save_results(self, out_dir='outputs'):
        """
        Save all community profiling results.
        """
        print("💾 Saving community profiling results...")
        os.makedirs(out_dir, exist_ok=True)
        
        saved_files = {}
        
        # 1. Community profiles
        if hasattr(self, 'community_profiles') and self.community_profiles:
            profiles_file = os.path.join(out_dir, 'community_profiles.json')
            try:
                with open(profiles_file, 'w') as f:
                    json.dump(self.community_profiles, f, indent=2, default=str)
                print(f"   Community profiles saved to {profiles_file}")
                saved_files['profiles_file'] = profiles_file
            except Exception as e:
                print(f"   Error saving community profiles: {e}")
        
        # 2. Evolution data
        if hasattr(self, 'evolution_data') and not self.evolution_data.empty:
            evolution_file = os.path.join(out_dir, 'temporal_community_activity.csv')
            try:
                self.evolution_data.to_csv(evolution_file, index=False)
                print(f"   Temporal activity saved to {evolution_file}")
                saved_files['evolution_file'] = evolution_file
            except Exception as e:
                print(f"   Error saving evolution data: {e}")
        
        # 3. Dashboard summary
        try:
            summary_stats = self.create_community_dashboard_data()
            summary_file = os.path.join(out_dir, 'dashboard_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            print(f"   Dashboard summary saved to {summary_file}")
            saved_files['summary_file'] = summary_file
        except Exception as e:
            print(f"   Error saving dashboard summary: {e}")
        
        # 4. Trending communities
        try:
            if hasattr(self, 'evolution_data') and not self.evolution_data.empty:
                trending = self.identify_trending_communities()
                if trending:
                    trending_file = os.path.join(out_dir, 'trending_communities.json')
                    with open(trending_file, 'w') as f:
                        json.dump(trending, f, indent=2, default=str)
                    print(f"   Trending communities saved to {trending_file}")
                    saved_files['trending_file'] = trending_file
        except Exception as e:
            print(f"   Error saving trending communities: {e}")
        
        return saved_files


def run_community_profiling_pipeline(posts_file='data/processed/posts.csv', 
                                   graph_file='data/graphs/ticker_network.gexf',
                                   out_dir='outputs'):
    """
    Run the complete community profiling pipeline with error handling.
    """
    print("🚀 STARTING COMMUNITY PROFILING PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize profiler
        print("\n🔧 Initializing profiler...")
        profiler = CommunityProfiler(posts_file, graph_file)
        
        # 1. Compute sentiment
        print("\n1️⃣ Computing sentiment scores...")
        try:
            profiler.compute_sentiment_scores(method='vader')
        except Exception as e:
            print(f"Warning: Sentiment analysis failed: {e}")
            print("Proceeding with neutral sentiment scores...")
        
        # 2. Profile communities
        print("\n2️⃣ Creating community profiles...")
        try:
            community_profiles = profiler.profile_communities(top_tickers_count=10, min_posts=5)
        except Exception as e:
            print(f"Error in community profiling: {e}")
            return None, {}
        
        # 3. Analyze evolution (optional)
        print("\n3️⃣ Analyzing temporal evolution...")
        try:
            evolution_data = profiler.analyze_community_evolution(time_window='7D')
        except Exception as e:
            print(f"Warning: Temporal analysis failed: {e}")
            evolution_data = pd.DataFrame()
        
        # 4. Find trending communities (optional)
        print("\n4️⃣ Identifying trending communities...")
        try:
            trending = profiler.identify_trending_communities(lookback_days=14, min_growth_rate=1.5)
        except Exception as e:
            print(f"Warning: Trending analysis failed: {e}")
            trending = []
        
        # 5. Create visualizations
        print("\n5️⃣ Creating visualizations...")
        try:
            profiler.visualize_community_profiles(out_dir)
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
        
        # 6. Save results
        print("\n6️⃣ Saving results...")
        try:
            output_files = profiler.save_results(out_dir)
        except Exception as e:
            print(f"Warning: Some results could not be saved: {e}")
            output_files = {}
        
        # Print summary
        print(f"\n✅ COMMUNITY PROFILING COMPLETE!")
        print(f"📊 Analyzed {len(community_profiles) if community_profiles else 0} communities")
        print(f"📈 Tracked {len(evolution_data) if not evolution_data.empty else 0} time periods")
        print(f"🔥 Found {len(trending)} trending communities")
        print(f"📁 Results saved in '{out_dir}/' directory")
        
        # Print top insights
        if community_profiles:
            print(f"\n🏆 TOP INSIGHTS:")
            
            try:
                # Most active community
                most_active = max(community_profiles.items(), key=lambda x: x[1]['activity_score'])
                print(f"   Most Active: {most_active[1]['label']}")
                
                # Most bullish community
                most_bullish = max(community_profiles.items(), key=lambda x: x[1]['avg_sentiment'])
                print(f"   Most Bullish: {most_bullish[1]['label']} (sentiment: {most_bullish[1]['avg_sentiment']:.3f})")
                
                # Most bearish community
                most_bearish = min(community_profiles.items(), key=lambda x: x[1]['avg_sentiment'])
                print(f"   Most Bearish: {most_bearish[1]['label']} (sentiment: {most_bearish[1]['avg_sentiment']:.3f})")
                
                # Trending communities
                if trending:
                    top_trending = trending[0]
                    tickers_str = ', '.join(top_trending['recent_top_tickers'][:3]) if top_trending['recent_top_tickers'] else 'N/A'
                    print(f"   Top Trending: {tickers_str} (growth: {top_trending['growth_rate']:.1f}x)")
            except Exception as e:
                print(f"   Could not extract top insights: {e}")
        
        return profiler, output_files
        
    except Exception as e:
        print(f"❌ PIPELINE FAILED: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your graph file exists and has community attributes")
        print("2. Verify that your posts.csv has the expected columns")
        print("3. Ensure user_id values match between posts and graph")
        print("4. Check file paths are correct")
        return None, {}


if __name__ == "__main__":
    # Run the pipeline with error handling
    try:
        profiler, output_files = run_community_profiling_pipeline(
            posts_file='data/processed/posts.csv',
            graph_file='data/graphs/ticker_network.gexf',  # Updated to match your graph construction
            out_dir='outputs'
        )
        
        if profiler is not None:
            print("\n📊 Next steps:")
            print("1. Review community_profiles.json for detailed insights")
            print("2. Run dashboard.py for interactive visualization")
            print("3. Use temporal_community_activity.csv for time series analysis")
        else:
            print("\n⚠️ Pipeline failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your file paths and data format.")