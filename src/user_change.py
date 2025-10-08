import pandas as pd
import random

# Load data
posts = pd.read_csv("data/processed/posts.csv", low_memory=False)
users = pd.read_csv("data/processed/users.csv")

# Filter valid users (exclude 'None' if desired)
valid_users = users[users['user_id'].notna() & (users['user_id'] != 'None')]['user_id'].tolist()

# If you want to keep the giant 'None' user, include them as well
# valid_users = users['user_id'].tolist()

# Randomly assign user_id to each post
posts['user_id'] = [random.choice(valid_users) for _ in range(len(posts))]

# Also assign tickers from users.csv to posts
user_tickers_map = dict(zip(users['user_id'], users['tickers']))
posts['tickers'] = posts['user_id'].map(user_tickers_map)

# Save new posts CSV
posts.to_csv("data/processed/posts_with_users.csv", index=False)
print(posts.head())

if show_tickers and not posts_df.empty and "text" in posts_df.columns and "ts" in posts_df.columns:
    st.header("📈 Ticker Mentions Over Time")
    
    # Extract tickers from post text
    posts_with_tickers = []
    for _, row in posts_df.iterrows():
        if pd.notna(row.get("text")) and pd.notna(row.get("ts")):
            tickers = extract_tickers(row["text"])
            for ticker in tickers:
                posts_with_tickers.append({
                    "date": row["ts"].date() if pd.notna(row["ts"]) else None,
                    "ticker": ticker
                })
    
    if posts_with_tickers:
        ticker_time_df = pd.DataFrame(posts_with_tickers)
        ticker_time_df = ticker_time_df[ticker_time_df["date"].notna()]
        
        # Get top tickers
        top_tickers_time = ticker_time_df["ticker"].value_counts().head(10).index.tolist()
        ticker_time_filtered = ticker_time_df[ticker_time_df["ticker"].isin(top_tickers_time)]
        
        if not ticker_time_filtered.empty:
            ticker_daily = ticker_time_filtered.groupby(["date", "ticker"]).size().reset_index(name="mentions")
            
            # Line chart
            fig_ticker_time = px.line(
                ticker_daily,
                x="date", y="mentions",
                color="ticker",
                markers=True,
                title="Top 10 Ticker Mentions Over Time",
                template="plotly_dark"
            )
            fig_ticker_time.update_layout(height=500)
            st.plotly_chart(fig_ticker_time, use_container_width=True)
            
            # Animated bar race if enabled
            if animate_growth and len(ticker_daily["date"].unique()) > 1:
                st.subheader("🎬 Ticker Popularity Race")
                ticker_daily_sorted = ticker_daily.sort_values("date")
                
                fig_anim = px.bar(
                    ticker_daily_sorted,
                    x="mentions", y="ticker",
                    color="ticker",
                    orientation="h",
                    animation_frame=ticker_daily_sorted["date"].astype(str),
                    range_x=[0, ticker_daily_sorted["mentions"].max() * 1.1],
                    title="📊 Daily Ticker Mention Race",
                    template="plotly_dark"
                )
                fig_anim.update_layout(height=600, showlegend=False)
                fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
                st.plotly_chart(fig_anim, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------
# COMMUNITY DEEP DIVE
# --------------------------------------------------------------------
if sel_comm != "All" and not node_df.empty:
    st.header(f"🔍 Deep Dive: Community {sel_comm}")
    
    sc = int(sel_comm)
    members = node_df[node_df['community'] == sc]
    
    if not members.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Members", len(members))
        with col2:
            avg_deg = members["weighted_degree"].mean()
            st.metric("🔗 Avg Degree", f"{avg_deg:.2f}")
        with col3:
            avg_betw = members["betweenness"].mean()
            st.metric("🌉 Avg Betweenness", f"{avg_betw:.6f}")
        
        st.subheader("🌟 Top Influencers")
        top_users = members.sort_values("betweenness", ascending=False).head(10)
        
        # Build display columns based on what's available
        display_cols = []
        col_rename = {}
        
        if user_id_col and user_id_col in top_users.columns:
            display_cols.append(user_id_col)
            col_rename[user_id_col] = "User"
        
        if "weighted_degree" in top_users.columns:
            display_cols.append("weighted_degree")
            col_rename["weighted_degree"] = "Degree"
        
        if "betweenness" in top_users.columns:
            display_cols.append("betweenness")
            col_rename["betweenness"] = "Betweenness"
        
        if "num_posts" in top_users.columns:
            display_cols.append("num_posts")
            col_rename["num_posts"] = "Posts"
        
        if display_cols:
            st.dataframe(
                top_users[display_cols].rename(columns=col_rename),
                use_container_width=True
            )
        else:
            st.dataframe(top_users, use_container_width=True)
        
        # Show community-specific tickers if available
        if not users_df.empty and 'tickers' in users_df.columns and user_id_col:
            comm_users = users_df[users_df['community'] == sc] if 'community' in users_df.columns else pd.DataFrame()
            if not comm_users.empty:
                st.subheader("🎯 Popular Tickers in This Community")
                comm_ticker_rows = []
                for _, row in comm_users.iterrows():
                    for ticker in parse_ticker_list(row['tickers']):
                        if ticker:
                            comm_ticker_rows.append({"ticker": ticker})
                
                if comm_ticker_rows:
                    comm_ticker_df = pd.DataFrame(comm_ticker_rows)
                    comm_ticker_counts = comm_ticker_df["ticker"].value_counts().head(15)
                    
                    fig_comm_ticker = px.bar(
                        x=comm_ticker_counts.index,
                        y=comm_ticker_counts.values,
                        title=f"Top 15 Tickers in Community {sel_comm}",
                        labels={"x": "Ticker", "y": "User Count"},
                        color=comm_ticker_counts.values,
                        color_continuous_scale="Blues"
                    )
                    fig_comm_ticker.update_layout(template="plotly_dark", showlegend=False, height=400)
                    st.plotly_chart(fig_comm_ticker, use_container_width=True)
    else:
        st.info("No member data available for this community.")

st.markdown("---")

# --------------------------------------------------------------------
# PLATFORM ANALYSIS
# --------------------------------------------------------------------
if not posts_df.empty and "platform" in posts_df.columns:
    st.header("🌐 Platform Distribution")
    
    platform_counts = posts_df["platform"].value_counts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_platform = px.pie(
            values=platform_counts.values,
            names=platform_counts.index,
            title="Posts by Platform",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_platform.update_traces(textinfo='label+percent+value')
        fig_platform.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_platform, use_container_width=True)
    
    with col2:
        if "community" in posts_df.columns and "ts" in posts_df.columns:
            # Platform activity over time
            platform_time = posts_df.groupby([posts_df["ts"].dt.floor('D'), "platform"]).size().reset_index(name="posts")
            platform_time.columns = ["date", "platform", "posts"]
            
            fig_platform_time = px.area(
                platform_time,
                x="date", y="posts",
                color="platform",
                title="Platform Activity Over Time",
                template="plotly_dark"
            )
            fig_platform_time.update_layout(height=400)
            st.plotly_chart(fig_platform_time, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------
# USER STATISTICS
# --------------------------------------------------------------------
if not users_df.empty:
    st.header("👥 User Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users_df))
    
    with col2:
        if "num_posts" in users_df.columns:
            avg_posts = users_df["num_posts"].mean()
            st.metric("Avg Posts/User", f"{avg_posts:.1f}")
    
    with col3:
        if "community" in users_df.columns:
            users_with_comm = users_df["community"].notna().sum()
            st.metric("Users in Communities", users_with_comm)
    
    with col4:
        if "tickers" in users_df.columns:
            users_with_tickers = users_df["tickers"].notna().sum()
            st.metric("Users with Tickers", users_with_tickers)
    
    # Top users by posts
    if "num_posts" in users_df.columns:
        st.subheader("🏆 Most Active Users")
        top_active = users_df.nlargest(20, "num_posts")
        
        display_cols = ["user_id", "num_posts"]
        if "community" in top_active.columns:
            display_cols.append("community")
        if "tickers" in top_active.columns:
            display_cols.append("tickers")
        
        fig_top_users = px.bar(
            top_active,
            x="user_id", y="num_posts",
            color="community" if "community" in top_active.columns else None,
            title="Top 20 Most Active Users",
            labels={"user_id": "User ID", "num_posts": "Number of Posts"},
            template="plotly_dark"
        )
        fig_top_users.update_layout(height=450, xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_top_users, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------
# NETWORK STATISTICS
# --------------------------------------------------------------------
st.header("📊 Network Statistics Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔵 Node Metrics")
    if not node_df.empty:
        st.metric("Total Nodes", len(node_df))
        st.metric("Avg Weighted Degree", f"{node_df['weighted_degree'].mean():.2f}")
        st.metric("Max Weighted Degree", f"{node_df['weighted_degree'].max():.2f}")
        st.metric("Avg Betweenness", f"{node_df['betweenness'].mean():.6f}")

with col2:
    st.markdown("### 🏘️ Community Metrics")
    if not comm_df.empty:
        st.metric("Communities Found", len(comm_df))
        st.metric("Avg Community Size", f"{comm_df['size'].mean():.1f}")
        st.metric("Largest Community", comm_df['size'].max())
        st.metric("Avg Density", f"{comm_df['density'].mean():.4f}")

with col3:
    st.markdown("### 🌐 Graph Metrics")
    st.metric("Total Nodes", G.number_of_nodes())
    st.metric("Total Edges", G.number_of_edges())
    if G.number_of_nodes() > 0:
        density = (2 * G.number_of_edges()) / (G.number_of_nodes() * (G.number_of_nodes() - 1))
        st.metric("Graph Density", f"{density:.6f}")
        avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        st.metric("Avg Degree", f"{avg_degree:.2f}")

st.markdown("---")

# --------------------------------------------------------------------
# DATA EXPORT
# --------------------------------------------------------------------
st.header("📂 Data Export")
st.markdown("Download the processed data for further analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if not node_df.empty:
        st.download_button(
            label="📥 Download Node Data",
            data=to_csv_bytes(node_df),
            file_name=f"node_communities_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if not comm_df.empty:
        st.download_button(
            label="📥 Download Community Summary",
            data=to_csv_bytes(comm_df),
            file_name=f"community_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if not filtered_node_df.empty:
        st.download_button(
            label="📥 Download Filtered Nodes",
            data=to_csv_bytes(filtered_node_df),
            file_name=f"filtered_nodes_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col4:
    if not users_df.empty:
        st.download_button(
            label="📥 Download Users Data",
            data=to_csv_bytes(users_df),
            file_name=f"users_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# --------------------------------------------------------------------
# INSIGHTS & RECOMMENDATIONS
# --------------------------------------------------------------------
st.markdown("---")
st.header("💡 Key Insights & Recommendations")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("### 🎯 Community Insights")
    if not comm_df.empty:
        largest_comm = comm_df.loc[comm_df['size'].idxmax()]
        st.success(f"**Largest Community:** Community {largest_comm['community']} with {largest_comm['size']} members")
        
        densest_comm = comm_df.loc[comm_df['density'].idxmax()]
        st.info(f"**Most Dense Community:** Community {densest_comm['community']} (density: {densest_comm['density']:.4f})")
        
        most_connected = comm_df.loc[comm_df['avg_weighted_degree'].idxmax()]
        st.warning(f"**Most Connected Community:** Community {most_connected['community']} (avg degree: {most_connected['avg_weighted_degree']:.2f})")

with insights_col2:
    st.markdown("### 🌟 Network Insights")
    if not node_df.empty:
        top_influencer = node_df.loc[node_df['betweenness'].idxmax()]
        influencer_name = top_influencer[user_id_col] if user_id_col else "N/A"
        st.success(f"**Top Influencer:** {influencer_name} (betweenness: {top_influencer['betweenness']:.6f})")
        
        most_connected_user = node_df.loc[node_df['weighted_degree'].idxmax()]
        connected_name = most_connected_user[user_id_col] if user_id_col else "N/A"
        st.info(f"**Most Connected User:** {connected_name} (degree: {most_connected_user['weighted_degree']:.2f})")
        
        if not posts_df.empty and "ts" in posts_df.columns:
            latest_date = posts_df["ts"].max()
            if pd.notna(latest_date):
                st.warning(f"**Latest Activity:** {latest_date.strftime('%Y-%m-%d')}")

# --------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
    <h3>💡 Built with ❤️ for Investor Community Analysis</h3>
    <p style="opacity: 0.8;">
        Powered by Streamlit • Plotly • NetworkX • PyVis • WordCloud
    </p>
    <p style="opacity: 0.6; font-size: 0.9rem;">
        🌟 Use the sidebar controls to filter and customize your analysis<br>
        📊 Analyze investor communities, track trending tickers, and identify key influencers<br>
        🔍 Export data for further analysis in your preferred tools
    </p>
</div>
""", unsafe_allow_html=True)