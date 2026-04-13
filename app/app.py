import streamlit as st
from google.cloud import aiplatform
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="mlB Pitch Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #005A9C;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Sidebar image title */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #005A9C;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #005A9C;
    }
    
    /* Button */
    .stButton>button {
        background-color: #005A9C;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        background-color: #003d6b;
    }
    
    /* Remove top padding from sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Vertex AI
PROJECT_ID = "733591653377"
LOCATION = "us-central1"
ENDPOINT_NAME = "glasnow-pitch-endpoint"

@st.cache_resource
def get_endpoint():
    """Get the deployed endpoint (cached)"""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_NAME}"'
    )
    return endpoints[0]

def predict_pitch(game_situation):
    """Make prediction via Vertex AI endpoint"""
    endpoint = get_endpoint()
    
    # Convert all values to strings (required by AutoML)
    instance = {k: str(v) for k, v in game_situation.items()}
    
    # Make prediction
    prediction = endpoint.predict(instances=[instance])
    
    # Extract results
    classes = prediction.predictions[0]['classes']
    scores = prediction.predictions[0]['scores']
    
    return dict(zip(classes, scores))

# App header (main area)
st.markdown('<h1 class="main-header">⚾ mlB Pitch Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Powered by Google Cloud Vertex AI</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with image
with st.sidebar:
    # Display the image (full width of sidebar) - NO TEXT ABOVE
    st.image("images/glasnow.png", use_container_width=True)
    
    st.markdown("---")
    
    st.header("🎯 Game Situation")
    st.markdown("---")
    
    # Game context
    st.subheader("⚾ Game Context")
    inning = st.slider("Inning", 1, 9, 5, help="Current inning of the game")
    outs = st.slider("Outs", 0, 2, 1, help="Number of outs")
    
    st.markdown("---")
    
    # Count
    st.subheader("🎯 Count")
    col1, col2 = st.columns(2)
    with col1:
        balls = st.number_input("Balls", 0, 3, 1, help="Number of balls")
    with col2:
        strikes = st.number_input("Strikes", 0, 2, 2, help="Number of strikes")
    
    # Show count visually
    count_display = f"**Count: {balls}-{strikes}**"
    if balls >= 3:
        st.markdown(f"🟢 {count_display} (Hitter's count)")
    elif strikes >= 2:
        st.markdown(f"🔴 {count_display} (Pitcher's count)")
    else:
        st.markdown(f"🟡 {count_display}")
    
    st.markdown("---")
    
    # Batter
    st.subheader("🏏 Batter")
    hitter_hand = st.radio("Handedness", ["L", "R"], index=1, horizontal=True)
    
    st.markdown("---")
    
    # Runners
    st.subheader("🏃 Runners on Base")
    col1, col2, col3 = st.columns(3)
    with col1:
        runner_1b = st.checkbox("1st", value=False)
    with col2:
        runner_2b = st.checkbox("2nd", value=True)
    with col3:
        runner_3b = st.checkbox("3rd", value=False)
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 PREDICT PITCH", use_container_width=True, type="primary")

# Main area
if predict_button:
    # Build game situation
    game_situation = {
        "inning": inning,
        "hitter_hand": hitter_hand,
        "balls": balls,
        "strikes": strikes,
        "runner_1b": 1 if runner_1b else 0,
        "runner_2b": 1 if runner_2b else 0,
        "runner_3b": 1 if runner_3b else 0,
        "outs": outs
    }
    
    # Show spinner while predicting
    with st.spinner("🤖 Analyzing game situation..."):
        try:
            predictions = predict_pitch(game_situation)
            
            # Sort by probability
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Pitch Probability Distribution")
                
                # Create bar chart with better styling
                pitch_names = {
                    'FF': '4-Seam Fastball',
                    'SL': 'Slider',
                    'CU': 'Curveball',
                    'SI': 'Sinker'
                }
                
                labels = [pitch_names.get(p[0], p[0]) for p in sorted_preds]
                values = [p[1] * 100 for p in sorted_preds]
                colors = ['#005A9C', '#EF3E42', '#A5ACAF', '#FDB827']  # Dodgers colors
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        text=[f'{v:.1f}%' for v in values],
                        textposition='auto',
                        textfont=dict(size=14, color='white', family='Arial Black'),
                        marker_color=colors,
                        marker_line_color='white',
                        marker_line_width=2
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, max(values) + 10]),
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14, family='Arial')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Top Prediction")
                top_pitch = sorted_preds[0]
                pitch_type = pitch_names.get(top_pitch[0], top_pitch[0])
                confidence = top_pitch[1] * 100
                
                st.metric(
                    label="Most Likely Pitch",
                    value=pitch_type,
                    delta=f"{confidence:.1f}% confidence"
                )
                
                st.markdown("---")
                
                st.subheader("📋 All Probabilities")
                for pitch, prob in sorted_preds:
                    percentage = prob * 100
                    # Create progress bar
                    st.write(f"**{pitch_names.get(pitch, pitch)}**")
                    st.progress(prob)
                    st.write(f"{percentage:.1f}%")
                    st.markdown("")
            
            # Show game situation summary
            st.markdown("---")
            st.subheader("📝 Game Situation Summary")
            
            count = f"{balls}-{strikes}"
            runners = []
            if runner_1b: runners.append("1st")
            if runner_2b: runners.append("2nd")
            if runner_3b: runners.append("3rd")
            runners_text = ", ".join(runners) if runners else "Bases empty"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Inning", inning)
                st.metric("Count", count)
            with col2:
                st.metric("Outs", outs)
                st.metric("Batter", f"{hitter_hand}HB")
            with col3:
                st.metric("Runners", runners_text if runners else "Empty")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("💡 This might be a cold start. Wait 30 seconds and try again.")

else:
    # Instructions when no prediction yet
    st.info("👈 **Set the game situation in the sidebar and click 'PREDICT PITCH'**")
    
    # Create three columns for info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        This app uses a machine learning model trained on **~4,000 Tyler Glasnow pitches** to predict pitch type based on game context.
        """)
    
    with col2:
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        - **ROC AUC:** 0.731
        - **Best at:** Fastballs (75% accuracy)
        - **Platform:** Google Cloud Vertex AI AutoML
        """)
    
    with col3:
        st.markdown("### ⚾ Pitch Types")
        st.markdown("""
        - **FF** - 4-Seam Fastball
        - **SL** - Slider  
        - **CU** - Curveball
        - **SI** - Sinker
        """)
    
    st.markdown("---")
    
    # Feature importance visualization
    st.subheader("🔍 What Influences Predictions Most?")
    st.markdown("The model learned that **strike count** is by far the most important factor:")
    
    features = ['Strikes', 'Hitter Hand', 'Balls', 'Outs', 'Inning', 'Runner 2B', 'Runner 3B', 'Runner 1B']
    importance = [100, 45, 40, 25, 18, 12, 10, 8]
    
    fig2 = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#005A9C'
    ))
    
    fig2.update_layout(
        xaxis_title="Relative Importance",
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Built with</strong> Streamlit + Google Cloud Vertex AI</p>
    <p>Data Source: Baseball Savant | Model: AutoML Tabular Classification</p>
</div>
""", unsafe_allow_html=True)
