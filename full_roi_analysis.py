import pandas as pd
from enhanced_dataset import EnhancedConversionDataGenerator
from persistent_predictor import PersistentConversionPredictor

def full_roi_analysis_with_persistence():
    """Complete ROI analysis using persistent model"""
    
    print("=" * 70)
    print("COMPLETE ROI ANALYSIS WITH PERSISTENT MODEL")
    print("=" * 70)
    
    # Initialize predictor
    predictor = PersistentConversionPredictor(model_name="conversion_predictor_v1")
    
    # Check if model exists, if not train one
    if not predictor.is_model_saved():
        print("\n📝 No saved model found. Training new model...")
        
        # Generate training data
        generator = EnhancedConversionDataGenerator(num_users=1000)
        datasets = generator.generate_complete_dataset()
        
        # Prepare data and train
        user_features = predictor.load_and_prepare_data(
            datasets['interactions'],
            datasets['consent'],
            datasets['communications'],
            datasets['conversions']
        )
        
        X, y = predictor.prepare_features(user_features)
        results = predictor.train_model(X, y)
        predictor.save_model()
        
    else:
        print("\n✅ Loading existing model...")
        predictor.load_model()
        
        # Generate new data for analysis (in production, this would be your real data)
        print("\n📊 Generating analysis data...")
        generator = EnhancedConversionDataGenerator(num_users=1000)
        datasets = generator.generate_complete_dataset()
        
        user_features = predictor.load_and_prepare_data(
            datasets['interactions'],
            datasets['consent'],
            datasets['communications'],
            datasets['conversions']
        )
    
    # ROI Analysis
    print("\n💰 Calculating ROI impact...")
    roi_analysis = predictor.calculate_roi_impact(user_features)
    
    print(f"\nCURRENT APPROACH (Follow up with all leads):")
    current = roi_analysis['current_approach']
    print(f"  • Leads targeted: {current['leads_targeted']:,}")
    print(f"  • Conversions: {current['conversions']}")
    print(f"  • Conversion rate: {current['conversion_rate']:.1%}")
    print(f"  • Total cost: ${current['cost']:,}")
    print(f"  • Total revenue: ${current['revenue']:,}")
    print(f"  • ROI: {current['roi']:.1f}%")
    
    # Find best strategy
    best_strategy = max(roi_analysis['model_scenarios'].items(), 
                       key=lambda x: x[1]['roi_improvement'])
    best_name, best_data = best_strategy
    
    print(f"\n🎯 RECOMMENDED STRATEGY: {best_name}")
    print(f"  • Leads targeted: {best_data['leads_targeted']:,}")
    print(f"  • Conversions: {best_data['conversions']}")
    print(f"  • Conversion rate: {best_data['conversion_rate']:.1%}")
    print(f"  • Total cost: ${best_data['cost']:,}")
    print(f"  • Total revenue: ${best_data['revenue']:,}")
    print(f"  • ROI: {best_data['roi']:.1f}%")
    print(f"  • Cost savings: ${best_data['cost_savings']:,}")
    print(f"  • ROI improvement: +{best_data['roi_improvement']:.1f} points")
    
    # Lead scoring report
    print(f"\n📋 Lead scoring analysis...")
    scoring_report = predictor.generate_lead_scoring_report(user_features)
    
    print(f"\nLEAD SCORING SUMMARY:")
    print(scoring_report['summary'])
    
    # Feature importance
    print(f"\n⚡ Key success factors:")
    importance_df = predictor.get_feature_importance()
    top_factors = importance_df.head(5)
    
    for i, (_, row) in enumerate(top_factors.iterrows(), 1):
        clean_name = row['feature'].replace('_', ' ').title()
        print(f"  {i}. {clean_name}: {row['importance']:.3f}")
    
    # Save all results
    print(f"\n💾 Saving analysis results...")
    
    # Lead scoring
    scoring_report['detailed_data'].to_csv('persistent_lead_scoring.csv', index=False)
    
    # ROI analysis
    roi_summary = []
    roi_summary.append(['Current Approach'] + list(current.values()))
    for name, data in roi_analysis['model_scenarios'].items():
        roi_summary.append([name] + list(data.values()))
    
    roi_df = pd.DataFrame(roi_summary, columns=['Strategy', 'Leads_Targeted', 'Conversions', 
                                               'Conversion_Rate', 'Cost', 'Revenue', 'ROI', 
                                               'Cost_Savings', 'ROI_Improvement'])
    roi_df.to_csv('persistent_roi_analysis.csv', index=False)
    
    # Feature importance
    importance_df.to_csv('persistent_feature_importance.csv', index=False)
    
    print(f"  • Lead scoring: persistent_lead_scoring.csv")
    print(f"  • ROI analysis: persistent_roi_analysis.csv") 
    print(f"  • Feature importance: persistent_feature_importance.csv")
    
    print(f"\n✅ Complete analysis finished!")
    
    return predictor, user_features, roi_analysis

if __name__ == "__main__":
    full_roi_analysis_with_persistence()
