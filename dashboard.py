import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

@st.cache_resource
def load_model():
    """Loads the saved model and encoders."""
    try:
        model = joblib.load("conversion_predictor_v1.pkl")
        encoders = joblib.load("conversion_predictor_v1_encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        return None, None

# Load the model once at the start of the script
model, encoders = load_model()

# Set the page configuration to be wide
st.set_page_config(layout="wide")

# --- Data Loading ---
# Load the CSV files into pandas DataFrames
try:
    roi_df = pd.read_csv("persistent_roi_analysis.csv")
    leads_df = pd.read_csv("persistent_lead_scoring.csv")
    importance_df = pd.read_csv("persistent_feature_importance.csv")
except FileNotFoundError as e:
    st.error(f"❌ Error loading file: {e}. Make sure the required CSV files are in the correct directory.")
    st.stop() # Stop the app if files are missing

# --- Dashboard Title ---
st.title("🚀 Lead Conversion & ROI Dashboard")
st.markdown("---")

# --- Section 1: Lead Funnel Overview ---
st.header("📊 Lead Funnel Overview")

# KPI Cards using st.metric
col1, col2, col3 = st.columns(3)
total_leads = len(leads_df)
total_conversions = leads_df['converted'].sum()
conversion_rate = total_conversions / total_leads

col1.metric("Total Leads", f"{total_leads:,}")
col2.metric("Total Conversions", f"{total_conversions:,}")
col3.metric("Overall Conversion Rate", f"{conversion_rate:.1%}")

# Visualizations for Section 1
st.subheader("Lead Insights")
col1, col2 = st.columns(2)

# Chart 1: Leads by Quality
with col1:
    quality_dist = leads_df['primary_lead_quality'].value_counts().reset_index()
    fig_quality = px.pie(quality_dist, names='primary_lead_quality', values='count', title='Leads by Quality')
    st.plotly_chart(fig_quality, use_container_width=True)

# Chart 2: Distribution of Conversion Probability (Corrected)
with col2:
    fig_hist = px.histogram(
        leads_df,
        x="conversion_probability",
        color="converted",
        marginal="box",
        title="Distribution of Conversion Probability",
        labels={'converted': 'Converted Status'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")


# --- Section 2: Model ROI & Performance ---
st.header("📈 Model ROI & Performance")

# Extract data for ROI KPIs from the DataFrame
current_approach = roi_df[roi_df['Strategy'] == 'Current Approach'].iloc[0]
top_10_model = roi_df[roi_df['Strategy'] == 'Top 10%'].iloc[0]

# ROI KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Current Approach ROI", f"{current_approach['ROI']:.1f}%")
col2.metric("Model-Driven ROI (Top 10%)", f"{top_10_model['ROI']:.1f}%", delta=f"{top_10_model['ROI_Improvement']:.1f} pts")
col3.metric("Cost Savings (Top 10%)", f"${int(top_10_model['Cost_Savings']):,}")

# Bar chart for ROI comparison
st.subheader("ROI Strategy Comparison")
roi_chart_df = roi_df.set_index('Strategy')[['ROI']]
st.bar_chart(roi_chart_df)

# Bar chart for feature importance
st.subheader("Key Success Factors")
importance_chart_df = importance_df.head(10).set_index('feature')
st.bar_chart(importance_chart_df)

st.markdown("---")

# --- Section 3: Actionable Lead List ---
st.header("🎯 Actionable Lead List")
st.write("Use this table to prioritize leads. Sort by 'conversion_probability' to see the highest potential leads first.")

# Display the filterable and sortable lead scoring table (Corrected)
# Removed 'primary_location' from this list as it's not in the source file
display_cols = ['user_id', 'lead_score', 'conversion_probability', 'primary_lead_quality', 'engagement_score']
# Add the new column only if it exists in the DataFrame
if 'primary_interaction_day' in leads_df.columns:
    display_cols.append('primary_interaction_day')

st.dataframe(leads_df[display_cols])

# --- Add this new section to dashboard.py ---

st.markdown("---")
st.header("💡 Lead Profiling & Campaign Strategy")

high_value_leads = leads_df[leads_df['lead_score'] == 'High']

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profile of a High-Value Lead")
    avg_engagement = high_value_leads['engagement_score'].mean()
    avg_interactions = high_value_leads['total_interactions'].mean()
    st.info(f"**Average Engagement Score:** {avg_engagement:.2f}")
    st.info(f"**Average Interactions:** {avg_interactions:.2f}")

with col2:
    st.subheader("Top Channels for High-Value Leads")
    # This chart now works if you regenerated the CSV with the 'primary_channel' column
    if 'primary_channel' in high_value_leads.columns:
        channel_dist = high_value_leads['primary_channel'].value_counts().reset_index()
        fig_channel = px.pie(channel_dist, names='primary_channel', values='count', title='Primary Channel')
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.warning("Re-generate your lead scoring CSV with the 'primary_channel' column to see this chart.")

# --- Add this new section to the bottom of your dashboard.py ---

st.markdown("---")
st.header("🔮 'What-If' Scenario Planner")
st.write("Adjust the sliders and dropdowns to see how these key factors impact the probability of conversion in real-time.")

# Load the model and print a status message
# model, encoders = load_model()

if model is None or encoders is None:
    st.error("Could not find saved model files (`.pkl`). Please run the training script first.")
else:
    col1, col2, col3 = st.columns(3)
    
    # Create interactive input widgets
    with col1:
        engagement_score = st.slider("Engagement Score", 1.0, 10.0, 5.0, 0.1)
        primary_channel = st.selectbox("Primary Channel", encoders['primary_channel'].classes_)

    with col2:
        total_duration = st.slider("Total Interaction Duration (mins)", 5, 100, 30, 5)
        lead_quality = st.selectbox("Lead Quality", encoders['primary_lead_quality'].classes_)

    with col3:
        interaction_variety = st.slider("Interaction Variety (e.g., number of unique touchpoints)", 1, 10, 3)
        day_options = {"Weekday": 0, "Weekend": 5}
        selected_day = st.selectbox("Primary Interaction Day", options=list(day_options.keys()))
        primary_location = st.selectbox("Primary Location", encoders['primary_location'].classes_)

    # When the button is clicked, assemble the data and make a prediction
    if st.button("Predict Conversion Probability"):
        
        interaction_day_value = day_options[selected_day]

        # The model needs all 20 features to make a prediction.
        # We'll use your inputs for the key features and fill the rest with typical values.
        feature_dict = {
            'total_interactions': [8], 'interaction_variety': [interaction_variety],
            'primary_channel': [primary_channel], 'primary_lead_quality': [lead_quality],
            'gave_consent': [True], 'avg_interest': [3.5], 'max_interest': [5],
            'total_duration': [total_duration], 'avg_duration': [15], 'max_duration': [30],
            'primary_location': [primary_location], 'engagement_score': [engagement_score],
            'interaction_span_days': [2], 'email_consent': [True], 'sms_consent': [False],
            'data_tracking_consent': [True], 'still_consented': [True],
            'total_communications': [2], 'total_responses': [1], 'response_rate': [0.5],
            'primary_interaction_day': [interaction_day_value]
        }
        
        # Create a DataFrame and encode the categorical features
        input_df = pd.DataFrame(feature_dict)
        input_df['primary_channel'] = encoders['primary_channel'].transform(input_df['primary_channel'])
        input_df['primary_lead_quality'] = encoders['primary_lead_quality'].transform(input_df['primary_lead_quality'])
        input_df['primary_location'] = encoders['primary_location'].transform(input_df['primary_location'])

        # Make the prediction
        predicted_prob = model.predict_proba(input_df)[:, 1][0]
        
        # Display the dynamic result
        st.metric("Predicted Conversion Probability", f"{predicted_prob:.1%}")
