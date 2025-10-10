# Investor Communities: Social & Market Data Analysis

## Overview

Investor Communities is a comprehensive data science project for collecting, processing, analyzing, and visualizing social media and market data to understand online investor communities, their sentiment, and their network structure. The project integrates data from Reddit, Twitter, and market sources, builds user-ticker graphs, and profiles communities using advanced analytics and visualization tools.

---

## Features

- **Data Collection**

  - Collects Reddit posts and comments with rich metadata using PRAW.
  - Downloads and processes Twitter data from Kaggle and local CSVs.
  - Handles large-scale data efficiently and stores in JSON format.

- **Preprocessing**

  - Cleans and standardizes text, dates, and user information.
  - Extracts tickers and builds unified datasets for posts and users.
  - Integrates multiple data sources (Reddit, Twitter, market data).

- **Graph Construction**

  - Builds user co-mention graphs based on ticker mentions.
  - Filters out universal tickers and focuses on meaningful user relationships.
  - Outputs graphs in GEXF format for further analysis.

- **Community Profiling**

  - Detects communities in the user-ticker network.
  - Computes sentiment scores using VADER and TextBlob.
  - Profiles communities by top tickers, sentiment, and user activity.
  - Tracks community evolution and trends over time.

- **Visualization & Dashboard**
  - Provides tools for visualizing community profiles and network structure.
  - Supports dashboard creation for interactive exploration.

---

## Directory Structure

```
.
├── data/
│   ├── raw_twitter.json           # Processed Twitter data (JSON)
│   ├── reddit_wallstreetbets.json # Processed Reddit data (JSON)
│   ├── communities/               # Community summary CSVs
│   └── ...
├── src/
│   ├── collect_reddit.py          # Reddit data collection script
│   ├── collect_twitter.py         # Twitter data collection script
│   ├── preprocess.py              # Data preprocessing pipeline
│   ├── graph_construction.py      # User-ticker graph builder
│   ├── community_profiling.py     # Community profiling and sentiment analysis
│   └── streamlit_app.py           # (Optional) Streamlit dashboard app
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Setup & Installation

1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd investor-communities
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables**

   - Create a `.env` file with Reddit API credentials:
     ```env
     REDDIT_CLIENT_ID=your_id
     REDDIT_CLIENT_SECRET=your_secret
     REDDIT_USER_AGENT=your_agent
     ```
   - For Twitter data, ensure Kaggle API is set up if using KaggleHub.

4. **Data Collection**

   - Run Reddit collection:
     ```sh
     python src/collect_reddit.py --subreddit wallstreetbets --limit 100 --category hot
     ```
   - Run Twitter collection:
     ```sh
     python src/collect_twitter.py
     ```

5. **Preprocessing**

   ```sh
   python src/preprocess.py
   ```

6. **Graph Construction**

   ```sh
   python src/graph_construction.py
   ```

7. **Community Profiling**

   ```sh
   python src/community_profiling.py
   ```

8. **Visualization (Optional)**
   - If using Streamlit:
     ```sh
     streamlit run src/streamlit_app.py
     ```

---

## Key Scripts

- `collect_reddit.py`: Scrapes Reddit posts/comments with metadata.
- `collect_twitter.py`: Downloads and processes Twitter data from Kaggle/local CSV.
- `preprocess.py`: Cleans, merges, and standardizes all data sources.
- `graph_construction.py`: Builds user-ticker co-mention graphs.
- `community_profiling.py`: Profiles communities, computes sentiment, and analyzes trends.
- `streamlit_app.py`: (Optional) Interactive dashboard for data exploration.

---

## Data Files

- `data/raw_twitter.json`: All processed Twitter data.
- `data/reddit_wallstreetbets.json`: Reddit data from r/wallstreetbets.
- `data/communities/community_summary.csv`: Community-level network stats.
- `data/processed/posts.csv`: Unified posts dataset.
- `data/processed/users.csv`: Unified users dataset.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (pandas, praw, kagglehub, networkx, tqdm, vaderSentiment, textblob, streamlit, etc.)

---

## Notes

- Large raw data files and processed outputs are gitignored to avoid repository bloat.
- Ensure you have proper API credentials for Reddit and Kaggle.
- The project is modular—each script can be run independently for flexible workflows.

---

## License

MIT License

---

## Authors

- Spencer2502 (and contributors)

---

## Acknowledgments

- Reddit, Twitter, and Kaggle for data sources
- Open-source Python community
