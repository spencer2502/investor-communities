"""
Ultimate Investor Community Analyzer – Streamlit Dashboard 🌐✨
Combining network visualization, sentiment analysis, temporal evolution, and ticker tracking
Run:
    streamlit run src/streamlit_app.py
"""

import os
from pathlib import Path
import io
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import re

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Investor Community Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(".")
NODE_CSV = BASE / "data" / "communities" / "node_communities.csv"
COMM_CSV = BASE / "data" / "communities" / "community_summary.csv"
GEXF_P = BASE / "data" / "communities" / "ticker_network_with_communities.gexf"
USERS_CSV = BASE / "data" / "users.csv"
POSTS_CSV = BASE / "data" / "posts.csv"

# --------------------------------------------------------------------
# DATA LOADERS
# --------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_resource
def load_graph():
    return nx.read_gexf(GEXF_P) if GEXF_P.exists() else nx.Graph()

@st.cache_data(ttl=3600)
def load_posts():
    if POSTS_CSV.exists():
        df = pd.read_csv(POSTS_CSV)
        # Normalize timestamp fields
        for col in ["created_at", "created_utc", "timestamp", "date"]:
            if col in df.columns:
                try:
                    df["ts"] = pd.to_datetime(df[col], utc=True, errors="coerce")
                    break
                except Exception:
                    continue
        if "ts" not in df.columns:
            numeric_cols = df.select_dtypes("number").columns.tolist()
            if numeric_cols:
                try:
                    df["ts"] = pd.to_datetime(df[numeric_cols[0]], unit="s", errors="coerce")
                except Exception:
                    df["ts"] = pd.NaT
            else:
                df["ts"] = pd.NaT
        return df
    return pd.DataFrame()

def to_csv_bytes(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def extract_tickers(text):
    """Extract ticker symbols from text"""
    if pd.isna(text): 
        return []
    # Look for uppercase words 2-5 chars
    return list(set(re.findall(r"\b[A-Z]{2,5}\b", str(text))))

def parse_ticker_list(ticker_str):
    """Parse ticker strings from various formats"""
    if pd.isna(ticker_str):
        return []
    ticker_str = str(ticker_str)
    # Handle pipe-separated
    if '|' in ticker_str:
        return [t.strip() for t in ticker_str.split('|') if t.strip()]
    # Handle JSON-like arrays
    if '[' in ticker_str and ']' in ticker_str:
        try:
            import json
            return json.loads(ticker_str.replace("'", '"'))
        except:
            pass
    # Handle comma-separated
    if ',' in ticker_str:
        return [t.strip() for t in ticker_str.split(',') if t.strip()]
    return [ticker_str.strip()] if ticker_str.strip() else []

# Load data
node_df = load_csv(NODE_CSV)
comm_df = load_csv(COMM_CSV)
users_df = load_csv(USERS_CSV)
posts_df = load_posts()
G = load_graph()

# Identify user ID column in node_df
user_id_col = None
for col in ['user_id', 'node', 'user', 'username']:
    if col in node_df.columns:
        user_id_col = col
        break

# Merge community info with users if available
if not users_df.empty and not node_df.empty and user_id_col:
    users_df = users_df.merge(
        node_df[[user_id_col, 'community']], 
        left_on='user_id', 
        right_on=user_id_col, 
        how='left'
    )

# Merge community info with posts if available
if not posts_df.empty and not node_df.empty and user_id_col:
    posts_df = posts_df.merge(
        node_df[[user_id_col, 'community']], 
        left_on='user_id', 
        right_on=user_id_col, 
        how='left'
    )

# --------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

# Community filter
available_comms = sorted(comm_df["community"].unique()) if not comm_df.empty else []
comm_options = ["All"] + [str(c) for c in available_comms]
sel_comm = st.sidebar.selectbox("🎯 Select Community", options=comm_options, index=0)

# Size and betweenness filters
min_size = st.sidebar.slider("📊 Min Community Size", 1, int(comm_df["size"].max()) if not comm_df.empty else 100, 5)
min_betw = st.sidebar.slider("🔗 Min Betweenness", 0.0, float(node_df["betweenness"].max()) if not node_df.empty else 1.0, 0.0, format="%.4f")

# Visualization options
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visualization Options")
show_network = st.sidebar.checkbox("Show Network Graph", value=True)
show_wordcloud = st.sidebar.checkbox("Show WordClouds", value=True)
animate_growth = st.sidebar.checkbox("Enable Growth Animations", value=True)
show_tickers = st.sidebar.checkbox("Show Ticker Analysis", value=True)

# Advanced filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Advanced Filters")
date_range_filter = st.sidebar.checkbox("Enable Date Range Filter", value=False)
if date_range_filter and not posts_df.empty and "ts" in posts_df.columns:
    min_date = posts_df["ts"].min()
    max_date = posts_df["ts"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )

# --------------------------------------------------------------------
# HEADER SECTION
# --------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>💹 Ultimate Investor Community Analyzer</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">
        Detecting, Profiling, and Visualizing Investor Communities from Reddit & Twitter
    </p>
</div>
""", unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🔵 Nodes", f"{G.number_of_nodes():,}")
with col2:
    st.metric("🔗 Edges", f"{G.number_of_edges():,}")
with col3:
    st.metric("🏘️ Communities", len(comm_df))
with col4:
    mod = comm_df['overall_modularity'].iloc[0] if 'overall_modularity' in comm_df.columns and not comm_df.empty else comm_df['density'].mean() if not comm_df.empty else None
    st.metric("📐 Modularity", f"{mod:.4f}" if mod is not None else "n/a")
with col5:
    total_posts = len(posts_df) if not posts_df.empty else 0
    st.metric("📝 Total Posts", f"{total_posts:,}")

st.markdown("---")

# --------------------------------------------------------------------
# NETWORK VISUALIZATION
# --------------------------------------------------------------------
if show_network:
    st.header("🌐 Interactive Community Network")
    st.markdown("Explore the interconnected web of investors — hover to see detailed metrics")

    def build_pyvis(G, selected_comm=None, min_betw=0.0):
        net = Network(height="800px", width="100%", bgcolor="#0e1117", font_color="white", notebook=False)
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=100, spring_strength=0.001)
        
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        comms = sorted({d.get("community") for _, d in G.nodes(data=True) if d.get("community") is not None})
        comm_to_color = {str(c): f"rgb{tuple(int(255*x) for x in cmap(i % 20)[:3])}" for i, c in enumerate(comms)}
        
        nodes_added = 0
        max_nodes = 1000  # Limit for performance
        
        for n, d in G.nodes(data=True):
            if nodes_added >= max_nodes:
                break
                
            comm = d.get("community")
            # Get betweenness, default to 0 if not present
            bw = float(d.get("betweenness", 0.0))
            
            # Apply filters
            if selected_comm and selected_comm != "All" and str(comm) != str(selected_comm):
                continue
            if bw < min_betw:
                continue
            
            color = comm_to_color.get(str(comm), "gray")
            deg = d.get('weighted_degree', d.get('degree', 0))
            title = f"<b>{n}</b><br>Community: {comm}<br>Degree: {deg:.2f}<br>Betweenness: {bw:.6f}"
            size = max(8, 8 + np.sqrt(float(deg) if deg else 1) * 2)
            net.add_node(str(n), label=str(n)[:15], title=title, color=color, size=size)
            nodes_added += 1
        
        # Add edges for nodes that exist in the network
        existing_ids = {n['id'] for n in net.get_nodes()} if hasattr(net, 'get_nodes') else set()
        if not existing_ids:
            existing_ids = set(net.node_ids) if hasattr(net, 'node_ids') else set()
        
        for u, v, ed in G.edges(data=True):
            if str(u) in existing_ids and str(v) in existing_ids:
                w = float(ed.get("weight", 1))
                net.add_edge(str(u), str(v), value=w, title=f"Weight: {w:.2f}")
        
        net.set_options("""
        var options = {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.5)",
              "size": 10
            }
          },
          "edges": {
            "color": {
              "inherit": true,
              "opacity": 0.4
            },
            "smooth": {
              "type": "continuous"
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -80000,
              "centralGravity": 0.3,
              "springLength": 100,
              "springConstant": 0.001
            },
            "maxVelocity": 50,
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        return net

    try:
        net = build_pyvis(G, None if sel_comm == "All" else sel_comm, min_betw)
        
        # Check if network has nodes
        if len(net.nodes) == 0:
            st.warning("⚠️ No nodes match the current filters. Try reducing the betweenness threshold.")
        else:
            html_path = "temp_vis.html"
            net.save_graph(html_path)
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=820, scrolling=True)
            st.success(f"✅ Displaying {len(net.nodes)} nodes and {len(net.edges)} edges")
    except Exception as e:
        st.error(f"⚠️ PyVis visualization failed: {e}")
        st.info("Switching to Plotly static visualization...")
        
        # Fallback to Plotly visualization
        try:
            # Filter graph based on selection
            if sel_comm != "All":
                nodes_to_show = [n for n, d in G.nodes(data=True) 
                               if str(d.get('community')) == str(sel_comm) and d.get('betweenness', 0) >= min_betw]
            else:
                nodes_to_show = [n for n, d in G.nodes(data=True) 
                               if d.get('betweenness', 0) >= min_betw]
            
            # Limit to reasonable number for performance
            if len(nodes_to_show) > 500:
                st.warning(f"⚠️ Too many nodes ({len(nodes_to_show)}). Sampling 500 nodes for visualization.")
                import random
                nodes_to_show = random.sample(nodes_to_show, 500)
            
            if len(nodes_to_show) == 0:
                st.warning("⚠️ No nodes match the current filters. Try reducing the betweenness threshold.")
            else:
                subG = G.subgraph(nodes_to_show)
                pos = nx.spring_layout(subG, k=0.5, iterations=50, seed=42)
                
                # Create edge trace
                edge_x = []
                edge_y = []
                for edge in subG.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create node trace
                node_x = []
                node_y = []
                node_colors = []
                node_sizes = []
                node_text = []
                
                for node in subG.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    data = subG.nodes[node]
                    comm = data.get('community', 'N/A')
                    deg = data.get('weighted_degree', data.get('degree', 0))
                    betw = data.get('betweenness', 0)
                    
                    node_colors.append(comm if comm != 'N/A' else 0)
                    node_sizes.append(max(8, 5 + np.sqrt(float(deg)) * 3))
                    node_text.append(f"<b>{node}</b><br>Community: {comm}<br>Degree: {deg:.2f}<br>Betweenness: {betw:.6f}")
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Community",
                            thickness=15,
                            xanchor='left'
                        ),
                        line=dict(width=1, color='white')
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    title=f"Network Graph ({len(nodes_to_show)} nodes, {subG.number_of_edges()} edges)",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    height=800,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ Displaying {len(nodes_to_show)} nodes (Plotly visualization)")
        
        except Exception as e2:
            st.error(f"⚠️ Both visualizations failed. Error: {e2}")
            st.info("Please check that your GEXF file contains valid node and edge data.")

    st.markdown("---")

# --------------------------------------------------------------------
# COMMUNITY ANALYTICS
# --------------------------------------------------------------------
st.header("📊 Community Analytics Dashboard")

# Filter data based on selections
filtered_comm_df = comm_df[comm_df["size"] >= min_size] if not comm_df.empty else pd.DataFrame()
filtered_node_df = node_df[node_df["betweenness"] >= min_betw] if not node_df.empty else pd.DataFrame()
if sel_comm != "All" and not filtered_node_df.empty:
    filtered_node_df = filtered_node_df[filtered_node_df["community"] == int(sel_comm)]

# Two column layout
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("🏘️ Community Size Distribution")
    if not filtered_comm_df.empty:
        fig_bar = px.bar(
            filtered_comm_df.sort_values("size", ascending=False).head(20),
            x="community", y="size",
            color="avg_weighted_degree",
            hover_data=["avg_weighted_degree", "avg_betweenness", "density"],
            title="Top 20 Communities by Size",
            color_continuous_scale="viridis",
            labels={"avg_weighted_degree": "Avg Degree"}
        )
        fig_bar.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No community data available.")

with col_right:
    st.subheader("📈 Network Metrics")
    if not filtered_comm_df.empty:
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(
            x=filtered_comm_df["avg_weighted_degree"],
            y=filtered_comm_df["density"],
            mode="markers",
            marker=dict(
                size=filtered_comm_df["size"] / 2,
                color=filtered_comm_df["avg_betweenness"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Avg Betweenness")
            ),
            text=filtered_comm_df["community"],
            hovertemplate="<b>Community %{text}</b><br>Avg Degree: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>"
        ))
        fig_metrics.update_layout(
            title="Community Connectivity vs Density",
            xaxis_title="Average Weighted Degree",
            yaxis_title="Density",
            height=450,
            template="plotly_dark"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

# Scatter plot: betweenness vs degree
if not filtered_node_df.empty:
    st.subheader("🔗 Node Influence Analysis")
    sample_size = min(len(filtered_node_df), 3000)
    sample_df = filtered_node_df.sample(sample_size) if len(filtered_node_df) > sample_size else filtered_node_df
    
    hover_col = user_id_col if user_id_col and user_id_col in sample_df.columns else None
    
    fig_scatter = px.scatter(
        sample_df,
        x="weighted_degree", y="betweenness",
        color="community",
        hover_name=hover_col,
        title=f"Betweenness Centrality vs Weighted Degree (n={sample_size})",
        template="plotly_dark",
        color_continuous_scale="Turbo"
    )
    fig_scatter.update_traces(marker=dict(line=dict(width=0.5, color='white')))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------
# TICKER ANALYSIS FROM USERS
# --------------------------------------------------------------------
if show_tickers:
    st.header("📈 Ticker Analysis by Community")
    
    if not users_df.empty and 'tickers' in users_df.columns and 'community' in users_df.columns:
        st.subheader("🔥 Most Popular Tickers per Community")
        
        # Parse tickers from users
        ticker_rows = []
        for _, row in users_df.iterrows():
            tickers = parse_ticker_list(row['tickers'])
            comm = row.get('community')
            if pd.notna(comm):
                for t in tickers:
                    if t:
                        ticker_rows.append({"community": int(comm), "ticker": t})
        
        if ticker_rows:
            ticker_df = pd.DataFrame(ticker_rows)
            ticker_summary = ticker_df.groupby(["community", "ticker"]).size().reset_index(name="count")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Top tickers overall
                top_tickers = ticker_df["ticker"].value_counts().head(20)
                fig_ticker_bar = px.bar(
                    x=top_tickers.index,
                    y=top_tickers.values,
                    title="Top 20 Most Popular Tickers",
                    labels={"x": "Ticker", "y": "User Count"},
                    color=top_tickers.values,
                    color_continuous_scale="Plasma"
                )
                fig_ticker_bar.update_layout(height=450, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_ticker_bar, use_container_width=True)
            
            with col2:
                # Sunburst: community -> ticker
                top_10_tickers = top_tickers.head(10).index.tolist()
                ticker_summary_filtered = ticker_summary[ticker_summary["ticker"].isin(top_10_tickers)]
                
                if not ticker_summary_filtered.empty:
                    fig_sun = px.sunburst(
                        ticker_summary_filtered,
                        path=["ticker", "community"],
                        values="count",
                        title="Top 10 Tickers Across Communities",
                        color="count",
                        color_continuous_scale="Viridis"
                    )
                    fig_sun.update_layout(height=450, template="plotly_dark")
                    st.plotly_chart(fig_sun, use_container_width=True)
            
            # Community-specific ticker breakdown
            if sel_comm != "All":
                st.subheader(f"🎯 Top Tickers in Community {sel_comm}")
                comm_tickers = ticker_summary[ticker_summary["community"] == int(sel_comm)]
                if not comm_tickers.empty:
                    comm_tickers = comm_tickers.sort_values("count", ascending=False).head(15)
                    fig_comm = px.bar(
                        comm_tickers,
                        x="ticker", y="count",
                        title=f"Top 15 Tickers in Community {sel_comm}",
                        color="count",
                        color_continuous_scale="Blues"
                    )
                    fig_comm.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_comm, use_container_width=True)
        else:
            st.info("No ticker data found in users.")
    else:
        st.info("Users data doesn't contain ticker information.")

st.markdown("---")

# --------------------------------------------------------------------
# WORDCLOUD ANALYSIS
# --------------------------------------------------------------------
if show_wordcloud and not posts_df.empty and "text" in posts_df.columns:
    st.header("☁️ Word Cloud Analysis")
    
    # Overall word cloud
    st.subheader("💬 Overall Discourse")
    all_text = " ".join(str(t) for t in posts_df["text"].dropna())
    
    if all_text.strip():
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                wc = WordCloud(
                    width=1000, height=500,
                    background_color="#0e1117",
                    colormap="viridis",
                    max_words=150,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(all_text)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("### 📊 Text Statistics")
                st.metric("Total Posts", len(posts_df))
                st.metric("Total Words", len(all_text.split()))
                st.metric("Avg Words/Post", f"{len(all_text.split()) / len(posts_df):.1f}")
                
                # Most common words
                from collections import Counter
                words = [w.lower() for w in all_text.split() if len(w) > 4 and w.isalpha()]
                common_words = Counter(words).most_common(10)
                st.markdown("**Top 10 Words:**")
                for word, count in common_words:
                    st.text(f"{word}: {count}")
        
        except Exception as e:
            st.error(f"WordCloud generation failed: {e}")
    
    # Community-specific word cloud
    if "community" in posts_df.columns and sel_comm != "All":
        st.subheader(f"💬 Community {sel_comm} Discourse")
        comm_posts = posts_df[posts_df["community"] == int(sel_comm)]
        
        if not comm_posts.empty:
            comm_text = " ".join(str(t) for t in comm_posts["text"].dropna())
            if comm_text.strip():
                try:
                    wc = WordCloud(
                        width=800, height=400,
                        background_color="#0e1117",
                        colormap="plasma",
                        max_words=100,
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate(comm_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    plt.tight_layout(pad=0)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Community WordCloud failed: {e}")

st.markdown("---")

# --------------------------------------------------------------------
# TEMPORAL EVOLUTION
# --------------------------------------------------------------------
st.header("📅 Temporal Evolution & Growth Patterns")

if not posts_df.empty and "ts" in posts_df.columns and not posts_df["ts"].isna().all():
    posts_df["date"] = posts_df["ts"].dt.tz_convert(None).dt.floor('D')
    
    # Overall activity timeline
    if "community" in posts_df.columns:
        timeline = posts_df.groupby(["date", "community"]).size().reset_index(name="posts")
        timeline = timeline[timeline["community"].notna()]
        
        if not filtered_comm_df.empty:
            timeline = timeline[timeline["community"].isin(filtered_comm_df["community"])]
        
        if not timeline.empty:
            st.subheader("📈 Community Activity Over Time")
            fig_timeline = px.line(
                timeline,
                x="date", y="posts",
                color="community",
                markers=True,
                title="Daily Post Volume by Community",
                template="plotly_dark"
            )
            fig_timeline.update_traces(line=dict(width=2))
            fig_timeline.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Cumulative growth
            if animate_growth:
                st.subheader("📊 Cumulative Growth")
                pivot = timeline.pivot(index='date', columns='community', values='posts').fillna(0).cumsum()
                pivot = pivot.sort_index()
                
                fig_cum = go.Figure()
                for comm in pivot.columns:
                    fig_cum.add_trace(go.Scatter(
                        x=pivot.index, y=pivot[comm],
                        mode='lines+markers',
                        name=f"Community {comm}",
                        line=dict(width=3)
                    ))
                fig_cum.update_layout(
                    title="Cumulative Post Growth",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Posts",
                    height=500,
                    template="plotly_dark",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.info("No temporal data available for selected communities.")
    else:
        # Show overall timeline without communities
        timeline = posts_df.groupby("date").size().reset_index(name="posts")
        if not timeline.empty:
            fig_timeline = px.line(
                timeline,
                x="date", y="posts",
                markers=True,
                title="Daily Post Volume",
                template="plotly_dark"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("No timestamp data available for temporal analysis.")

st.markdown("---")

# --------------------------------------------------------------------
# TICKER MENTIONS OVER TIME
# --------------------------------------------------------------------
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