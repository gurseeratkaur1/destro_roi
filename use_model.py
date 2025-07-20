import pandas as pd
from persistent_predictor import PersistentConversionPredictor

def load_and_use_model():
    """Load the saved model and use it for predictions"""
    
    print("=" * 60)
    print("USING PERSISTENT MODEL FOR PREDICTIONS")
    print("=" * 60)
    
    # Step 1: Initialize and load the saved model
    print("\n1. Loading saved model...")
    predictor = PersistentConversionPredictor(model_name="conversion_predictor_v1")
    
    if not predictor.load_model():
        print("❌ Could not load model. Run train_model.py first!")
        return
    
    # Step 2: Create new prospect data for scoring
    print("\n2. Scoring new prospects...")
    
    new_prospects = pd.DataFrame({
        'user_id': ['prospect_001', 'prospect_002', 'prospect_003', 'prospect_004'],
        'total_interactions': [5, 2, 1, 8],
        'interaction_variety': [3, 1, 1, 4],
        'primary_channel': ['in_person_booth', 'qr_code_scan', 'staff_referral', 'in_person_booth'],
        'primary_lead_quality': ['hot', 'warm', 'cold', 'hot'],
        'gave_consent': [True, True, False, True],
        'avg_interest': [4.5, 3.0, 2.0, 4.8],
        'max_interest': [5, 4, 2, 5],
        'total_duration': [45, 15, 5, 60],
        'avg_duration': [15, 7, 5, 20],
        'max_duration': [20, 10, 5, 25],
        'primary_location': ['Auckland', 'Wellington', 'Christchurch', 'Auckland'],
        'engagement_score': [3.2, 2.1, 1.5, 3.8],
        'interaction_span_days': [2, 0, 0, 3],
        'email_consent': [True, True, False, True],
        'sms_consent': [False, False, False, True],
        'data_tracking_consent': [True, True, False, True],
        'still_consented': [True, True, False, True],
        'total_communications': [0, 0, 0, 0],
        'total_responses': [0, 0, 0, 0],
        'response_rate': [0, 0, 0, 0]
    })
    
    # Step 3: Make predictions
    print("\n3. Making predictions...")
    predictions, probabilities = predictor.predict_conversion_probability(new_prospects)
    
    # Step 4: Create scoring results
    scoring_results = new_prospects[['user_id', 'primary_lead_quality', 'engagement_score']].copy()
    scoring_results['conversion_probability'] = probabilities
    scoring_results['predicted_conversion'] = predictions
    scoring_results['priority'] = 'Low'
    scoring_results.loc[scoring_results['conversion_probability'] > 0.3, 'priority'] = 'Medium'
    scoring_results.loc[scoring_results['conversion_probability'] > 0.7, 'priority'] = 'High'
    
    print(f"\nNEW PROSPECT SCORING RESULTS:")
    print(scoring_results.to_string(index=False))
    
    # Step 5: Generate recommendations
    print(f"\n4. Generating recommendations...")
    
    high_priority = scoring_results[scoring_results['priority'] == 'High']
    medium_priority = scoring_results[scoring_results['priority'] == 'Medium']
    low_priority = scoring_results[scoring_results['priority'] == 'Low']
    
    if not high_priority.empty:
        print(f"\n🔥 HIGH PRIORITY - IMMEDIATE FOLLOW-UP:")
        for _, prospect in high_priority.iterrows():
            print(f"   • {prospect['user_id']}: {prospect['conversion_probability']:.1%} probability")
    
    if not medium_priority.empty:
        print(f"\n📈 MEDIUM PRIORITY - NURTURE CAMPAIGN:")
        for _, prospect in medium_priority.iterrows():
            print(f"   • {prospect['user_id']}: {prospect['conversion_probability']:.1%} probability")
    
    if not low_priority.empty:
        print(f"\n📉 LOW PRIORITY - MINIMAL EFFORT:")
        for _, prospect in low_priority.iterrows():
            print(f"   • {prospect['user_id']}: {prospect['conversion_probability']:.1%} probability")
    
    # Step 6: Show feature importance
    print(f"\n5. Key success factors:")
    importance_df = predictor.get_feature_importance()
    top_factors = importance_df.head(5)
    
    for i, (_, row) in enumerate(top_factors.iterrows(), 1):
        clean_name = row['feature'].replace('_', ' ').title()
        print(f"   {i}. {clean_name}: {row['importance']:.3f}")
    
    # Step 7: Save results
    print(f"\n6. Saving results...")
    scoring_results.to_csv('new_prospect_scores.csv', index=False)
    print("   • Results saved to: new_prospect_scores.csv")
    
    print(f"\n✅ Prediction complete!")

def batch_score_prospects(csv_file_path):
    """Score prospects from a CSV file"""
    
    print(f"\n" + "=" * 60)
    print("BATCH SCORING FROM CSV FILE")
    print("=" * 60)
    
    # Load the model
    predictor = PersistentConversionPredictor(model_name="conversion_predictor_v1")
    
    if not predictor.load_model():
        print("❌ Could not load model. Run train_model.py first!")
        return
    
    try:
        # Load prospects from CSV
        prospects = pd.read_csv(csv_file_path)
        print(f"✅ Loaded {len(prospects)} prospects from {csv_file_path}")
        
        # Make predictions
        predictions, probabilities = predictor.predict_conversion_probability(prospects)
        
        # Add results to dataframe
        prospects['conversion_probability'] = probabilities
        prospects['predicted_conversion'] = predictions
        prospects['priority'] = 'Low'
        prospects.loc[prospects['conversion_probability'] > 0.3, 'priority'] = 'Medium'
        prospects.loc[prospects['conversion_probability'] > 0.7, 'priority'] = 'High'
        
        # Sort by probability
        prospects = prospects.sort_values('conversion_probability', ascending=False)
        
        # Save results
        output_file = csv_file_path.replace('.csv', '_scored.csv')
        prospects.to_csv(output_file, index=False)
        
        # Show summary
        priority_counts = prospects['priority'].value_counts()
        print(f"\nScoring Summary:")
        for priority, count in priority_counts.items():
            print(f"  • {priority} Priority: {count} prospects")
        
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Run the model usage demo
    load_and_use_model()
    
    # Uncomment to score prospects from a CSV file:
    # batch_score_prospects('your_prospects.csv')
