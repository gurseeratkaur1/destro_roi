import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# --- Page and Model Setup ---
st.set_page_config(layout="wide", page_title="Lead Conversion Dashboard")

@st.cache_resource
def load_model():
    """Loads the saved model and encoders."""
    try:
        model = joblib.load("conversion_predictor_v1.pkl")
        encoders = joblib.load("conversion_predictor_v1_encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_model()

# --- Sidebar Navigation ---
# st.sidebar.image("path/to/your/logo.png", width=150) # Uncomment and add your logo path
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Strategic Summary", "Funnel Overview", "ROI & Performance", "Lead Insights", "Key Driver Analysis", "What-If Planner")
)

# --- Data Loading ---
try:
    roi_df = pd.read_csv("persistent_roi_analysis.csv")
    leads_df = pd.read_csv("persistent_lead_scoring.csv")
    importance_df = pd.read_csv("persistent_feature_importance.csv")
except FileNotFoundError as e:
    st.error(f"❌ Error loading file: {e}. Make sure the required CSV files are in the correct directory.")
    st.stop()

# --- Main Dashboard ---
st.title("🚀 Lead Conversion & ROI Dashboard")
st.markdown("A strategic tool for **Destro Marketing**")
st.markdown("---")

# --- Conditional Page Display (Controlled by Sidebar) ---

if page == "Strategic Summary":
    st.header("Key Findings & Actionable Recommendations")
    
    st.subheader("Key Findings")
    st.info("""
    - **Engagement Score is the #1 predictor** of whether a lead will convert.
    - Targeting only the **Top 10% of leads** identified by the model can increase ROI from **696% to over 3800%**.
    - Leads from **in-person booth interactions** are consistently among the highest-value segments.
    """)

    st.subheader("Recommendations for Destro Marketing")
    st.success("""
    1.  **Prioritize Follow-Up:** Use the "Actionable Lead List" in the 'Lead Insights' tab to focus sales efforts on 'High' priority leads immediately.
    2.  **Launch Nurturing Campaigns:** Target 'Medium' priority leads with automated email campaigns to increase their engagement score.
    3.  **Optimize Campaigns:** Use the "What-If" planner to simulate how changes in marketing strategy could improve conversion rates.
    """)
    # Calculate overall conversion rate
    overall_cr = leads_df['converted'].mean()
    # Calculate conversion rate for high-priority leads
    high_priority_cr = leads_df[leads_df['lead_score'] == 'High']['converted'].mean()
    # Calculate the "lift"
    conversion_lift = high_priority_cr / overall_cr if overall_cr > 0 else 1

    # Add it as a KPI
    st.metric("💥 Conversion Lift", f"{conversion_lift:.1f}x", help="How many times more likely a 'High' scored lead is to convert compared to the average lead.")

elif page == "Funnel Overview":
    st.header("📊 Lead Funnel Overview")
    kpi_cols = st.columns(4) # Increased to 4 columns
    total_leads = len(leads_df)
    total_conversions = leads_df['converted'].sum()
    conversion_rate = total_conversions / total_leads if total_leads > 0 else 0

    kpi_cols[0].metric("Total Leads", f"{total_leads:,}")
    kpi_cols[1].metric("Total Conversions", f"{total_conversions:,}")
    kpi_cols[2].metric("Overall Conversion Rate", f"{conversion_rate:.1%}")

    # --- ADDED: Time-to-Conversion Insights ---
    if 'days_to_conversion' in leads_df.columns:
        converted_leads = leads_df.dropna(subset=['days_to_conversion'])
        avg_time_to_convert = converted_leads['days_to_conversion'].mean()
        kpi_cols[3].metric("Average Time to Convert", f"{avg_time_to_convert:.1f} days")
    else:
        kpi_cols[3].metric("Average Time to Convert", "N/A")

    st.subheader("Lead Insights")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        quality_dist = leads_df['primary_lead_quality'].value_counts().reset_index()
        fig_quality = px.pie(quality_dist, names='primary_lead_quality', values='count', title='Leads by Quality')
        st.plotly_chart(fig_quality, use_container_width=True)
    with chart_cols[1]:
        fig_hist = px.histogram(leads_df, x="conversion_probability", color="converted", marginal="box", title="Distribution of Conversion Probability", labels={'converted': 'Converted Status'})
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if 'days_to_conversion' not in leads_df.columns:
        st.warning("To see 'Time-to-Conversion' metrics, re-generate your CSV with the `days_to_conversion` column.")

elif page == "ROI & Performance":
    st.header("📈 Model ROI & Performance")
    current_approach = roi_df[roi_df['Strategy'] == 'Current Approach'].iloc[0]
    top_10_model = roi_df[roi_df['Strategy'] == 'Top 10%'].iloc[0]
    
    roi_kpi_cols = st.columns(3)
    roi_kpi_cols[0].metric("Current Approach ROI", f"{current_approach['ROI']:.1f}%")
    roi_kpi_cols[1].metric("Model-Driven ROI (Top 10%)", f"{top_10_model['ROI']:.1f}%", delta=f"{top_10_model['ROI_Improvement']:.1f} pts")
    roi_kpi_cols[2].metric("Cost Savings (Top 10%)", f"${int(top_10_model['Cost_Savings']):,}")
    
    st.subheader("ROI Strategy Comparison")
    roi_chart_df = roi_df.set_index('Strategy')[['ROI']]
    st.bar_chart(roi_chart_df)
    st.caption("This chart compares the ROI of targeting all leads vs. using the model's predictions.")
    
    st.subheader("Key Success Factors")
    importance_chart_df = importance_df.head(10).set_index('feature')
    st.bar_chart(importance_chart_df)
    st.caption("This chart shows the top factors the model uses to predict conversion.")
    
    # --- ADDED: Lead Score Performance Chart ---
    st.subheader("Lead Score Tier Performance")
    score_performance = leads_df.groupby('lead_score')['converted'].mean().reset_index().sort_values(by='converted', ascending=False)
    fig_score = px.bar(score_performance, x='lead_score', y='converted', title='Actual Conversion Rate by Lead Score Tier', labels={'converted': 'Actual Conversion Rate', 'lead_score': 'Lead Score Tier'}, text_auto='.1%')
    st.plotly_chart(fig_score, use_container_width=True)
    st.caption("This chart validates the model's scoring by showing the actual performance of each tier.")

elif page == "Lead Insights":
    st.header("💡 Lead Insights & Segments")
    st.subheader("Actionable Lead List")
    display_cols = ['user_id', 'lead_score', 'conversion_probability', 'primary_lead_quality', 'engagement_score']
    if 'primary_interaction_day' in leads_df.columns:
        display_cols.append('primary_interaction_day')
    st.dataframe(leads_df[display_cols])
    
    st.subheader("Deep-Dive into Lead Segments")
    required_cols = ['primary_channel', 'total_duration', 'engagement_score']
    if all(col in leads_df.columns for col in required_cols):
        segment = st.selectbox("Choose a Lead Segment to Analyze:", ("High-Value Converts", "High-Potential (Non-Converts)", "Low-Priority Leads"))
        if segment == "High-Value Converts": segment_df = leads_df[(leads_df['lead_score'] == 'High') & (leads_df['converted'] == 1)]
        elif segment == "High-Potential (Non-Converts)": segment_df = leads_df[(leads_df['lead_score'] == 'High') & (leads_df['converted'] == 0)]
        else: segment_df = leads_df[leads_df['lead_score'] == 'Low']
        if not segment_df.empty:
            st.write(f"Profile of **{segment}**:")
            seg_cols = st.columns(3)
            seg_cols[0].metric("Average Engagement Score", f"{segment_df['engagement_score'].mean():.2f}")
            seg_cols[1].metric("Average Interaction Duration", f"{segment_df['total_duration'].mean():.1f} mins")
            seg_cols[2].metric("Most Common Channel", segment_df['primary_channel'].mode()[0])
        else: st.warning(f"No leads found in the '{segment}' segment.")
    else:
        st.warning("Re-generate your CSV with additional columns (e.g., primary_channel) to enable this feature.")

# --- NEW: Key Driver Analysis Page ---
elif page == "Key Driver Analysis":
    st.header("🔎 Key Driver Analysis")
    st.write("Understand how the most important features impact conversion probability.")

    required_cols = ['engagement_score', 'total_duration', 'primary_channel']
    if all(col in leads_df.columns for col in required_cols):
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(leads_df, x="engagement_score", y="conversion_probability", color="converted", title="Engagement Score vs. Conversion Probability", color_discrete_map={1: '#10b981', 0: '#ef4444'})
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            channel_performance = leads_df.groupby('primary_channel')['converted'].mean().sort_values(ascending=False).reset_index()
            fig2 = px.bar(channel_performance, x="primary_channel", y="converted", title="Average Conversion Rate by Channel", labels={'converted': 'Average Conversion Rate', 'primary_channel': 'Channel'})
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Re-generate your CSV with additional columns to enable this feature.")

    st.markdown("---")
    st.subheader("Channel Performance vs. Lead Quality")

    # This check ensures the required columns are in your regenerated CSV
    if 'primary_channel' in leads_df.columns:
        # Create a pivot table to count conversions for each combination
        heatmap_data = pd.crosstab(
            leads_df['primary_lead_quality'],
            leads_df['primary_channel'],
            values=leads_df['converted'],
            aggfunc='sum'
        ).fillna(0)

        # Create the heatmap
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            labels=dict(x="Marketing Channel", y="Initial Lead Quality", color="Total Conversions"),
            title="Heatmap of Converted Leads by Channel & Quality"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("This heatmap shows which channels (e.g., in_person_booth) are most effective at converting high-quality leads (e.g., hot leads).")

elif page == "What-If Planner":
    st.header("🔮 'What-If' Scenario Planner")
    st.write("Adjust the inputs to see how key factors impact conversion probability.")
    if model is None or encoders is None:
        st.error("Model files not found. Please run the training script to use this feature.")
    else:
        planner_cols = st.columns(3)
        with planner_cols[0]:
            engagement_score = st.slider("Engagement Score", 1.0, 10.0, 5.0, 0.1, key="planner_engagement")
            primary_channel = st.selectbox("Primary Channel", encoders['primary_channel'].classes_, key="planner_channel")
        with planner_cols[1]:
            total_duration = st.slider("Total Interaction Duration (mins)", 5, 100, 30, 5, key="planner_duration")
            lead_quality = st.selectbox("Lead Quality", encoders['primary_lead_quality'].classes_, key="planner_quality")
        with planner_cols[2]:
            interaction_variety = st.slider("Interaction Variety", 1, 10, 3, key="planner_variety")
            day_options = {"Weekday": 0, "Weekend": 5}
            selected_day = st.selectbox("Primary Interaction Day", options=list(day_options.keys()), key="planner_day")

        if st.button("Predict Conversion Probability"):
            interaction_day_value = day_options[selected_day]
            feature_dict = { 'total_interactions': [8], 'interaction_variety': [interaction_variety], 'primary_channel': [primary_channel], 'primary_lead_quality': [lead_quality], 'gave_consent': [True], 'avg_interest': [3.5], 'max_interest': [5], 'total_duration': [total_duration], 'avg_duration': [15], 'max_duration': [30], 'primary_location': ['Auckland'], 'engagement_score': [engagement_score], 'interaction_span_days': [2], 'email_consent': [True], 'sms_consent': [False], 'data_tracking_consent': [True], 'still_consented': [True], 'total_communications': [2], 'total_responses': [1], 'response_rate': [0.5], 'primary_interaction_day': [interaction_day_value] }
            
            input_df = pd.DataFrame(feature_dict)
            for col in ['primary_channel', 'primary_lead_quality', 'primary_location']:
                if col not in encoders:
                    st.error(f"Encoder for '{col}' not found. Please ensure it was saved during training.")
                    st.stop()
                input_df[col] = encoders[col].transform(input_df[col])
            
            predicted_prob = model.predict_proba(input_df)[:, 1][0]
            st.metric("Predicted Conversion Probability", f"{predicted_prob:.1%}")