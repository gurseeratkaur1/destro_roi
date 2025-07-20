import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

class PersistentConversionPredictor:
    def __init__(self, model_name="conversion_predictor"):
        self.model_name = model_name
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_metadata = {}
        
        # File paths for persistence
        self.model_file = f"{model_name}.pkl"
        self.encoders_file = f"{model_name}_encoders.pkl"
        self.metadata_file = f"{model_name}_metadata.json"
        
    def load_and_prepare_data(self, interactions_df, consent_df, communications_df, conversions_df):
        """Load and prepare the data for modeling"""
        print("Loading and preparing data...")
        
        # Create target variable
        converted_users = set(conversions_df['user_id'].unique())
        interactions_df['converted'] = interactions_df['user_id'].apply(lambda x: 1 if x in converted_users else 0)
        
        # Aggregate interactions by user
        user_features = self._aggregate_user_interactions(interactions_df)
        
        # Add consent information
        user_features = self._add_consent_features(user_features, consent_df)
        
        # Add communication engagement features
        user_features = self._add_communication_features(user_features, communications_df)
        
        # Add conversion timeline features
        user_features = self._add_conversion_timeline_features(user_features, conversions_df, interactions_df)
        
        print(f"Prepared dataset with {len(user_features)} users and {len(user_features.columns)-1} features")
        print(f"Conversion rate: {user_features['converted'].mean():.2%}")
        
        return user_features

    def _aggregate_user_interactions(self, interactions_df):
        interactions_df['day_of_week'] = interactions_df['timestamp'].dt.dayofweek
        """Aggregate interaction data by user"""
        user_agg = interactions_df.groupby('user_id').agg({
            'interaction_type': ['count', 'nunique'],
            'timestamp': ['min', 'max'],
            'interaction_channel': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'lead_quality': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'day_of_week': lambda x: x.mode()[0] if not x.empty else -1 ,
            'contact_consent_given': 'any',
            'interest_level': ['mean', 'max'],
            'duration_minutes': ['sum', 'mean', 'max'],
            'location': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'converted': 'first'
        }).reset_index()
        
        # Flatten column names
        user_agg.columns = ['user_id', 'total_interactions', 'interaction_variety',
                           'first_interaction', 'last_interaction', 'primary_channel',
                           'primary_lead_quality', 'primary_interaction_day', 'gave_consent', 'avg_interest',
                           'max_interest', 'total_duration', 'avg_duration',
                           'max_duration', 'primary_location', 'converted']
        
        # Create engagement score
        user_agg['engagement_score'] = (
            user_agg['avg_interest'] * 0.3 +
            user_agg['total_interactions'] * 0.2 +
            user_agg['avg_duration'] * 0.1 +
            user_agg['interaction_variety'] * 0.4
        )
        
        # Calculate interaction span
        user_agg['interaction_span_days'] = (
            user_agg['last_interaction'] - user_agg['first_interaction']
        ).dt.days
        
        return user_agg

    def _add_consent_features(self, user_features, consent_df):
        """Add consent-related features"""
        if not consent_df.empty:
            consent_features = consent_df.groupby('user_id').agg({
                'email_consent': 'first',
                'sms_consent': 'first',
                'data_tracking_consent': 'first',
                'opt_out_date': lambda x: x.isna().all()
            }).reset_index()
            
            consent_features.columns = ['user_id', 'email_consent', 'sms_consent',
                                      'data_tracking_consent', 'still_consented']
            
            user_features = user_features.merge(consent_features, on='user_id', how='left')
        else:
            user_features['email_consent'] = False
            user_features['sms_consent'] = False
            user_features['data_tracking_consent'] = False
            user_features['still_consented'] = False
        
        return user_features

    def _add_communication_features(self, user_features, communications_df):
        """Add communication engagement features"""
        if not communications_df.empty:
            comm_features = communications_df.groupby('user_id').agg({
                'communication_type': 'count',
                'response_status': lambda x: (x != 'no_response').sum(),
                'communication_date': 'count'
            }).reset_index()
            
            comm_features.columns = ['user_id', 'total_communications', 'total_responses', 'comm_count']
            comm_features['response_rate'] = comm_features['total_responses'] / comm_features['total_communications']
            comm_features = comm_features.drop('comm_count', axis=1)
            
            user_features = user_features.merge(comm_features, on='user_id', how='left')
        else:
            user_features['total_communications'] = 0
            user_features['total_responses'] = 0
            user_features['response_rate'] = 0
        
        # Fill NaN values
        user_features[['total_communications', 'total_responses', 'response_rate']] = \
            user_features[['total_communications', 'total_responses', 'response_rate']].fillna(0)
        
        return user_features

    def _add_conversion_timeline_features(self, user_features, conversions_df, interactions_df):
        """Add conversion timeline features"""
        if not conversions_df.empty:
            conversion_timeline = conversions_df.merge(
                interactions_df.groupby('user_id')['timestamp'].min().reset_index(),
                on='user_id'
            )
            
            conversion_timeline['days_to_conversion'] = (
                conversion_timeline['conversion_date'] - conversion_timeline['timestamp']
            ).dt.days
            
            timeline_features = conversion_timeline[['user_id', 'days_to_conversion']].copy()
            user_features = user_features.merge(timeline_features, on='user_id', how='left')
        
        return user_features

    def prepare_features(self, df):
        """Prepare features for modeling"""
        feature_columns = [
            'total_interactions', 'interaction_variety', 'primary_channel',
            'primary_lead_quality', 'gave_consent', 'avg_interest', 'max_interest',
            'total_duration', 'avg_duration', 'max_duration', 'primary_location',
            'engagement_score', 'interaction_span_days', 'email_consent',
            'sms_consent', 'data_tracking_consent', 'still_consented',
            'total_communications', 'total_responses', 'response_rate', 'primary_interaction_day'
        ]
        
        X = df[feature_columns].copy()
        y = df['converted'] if 'converted' in df.columns else None
        
        # Handle missing values
        X = X.fillna(0)
        
        # Encode categorical variables
        categorical_columns = ['primary_channel', 'primary_lead_quality', 'primary_location']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle new categories in prediction data
                X[col] = X[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else 'unknown')
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Convert boolean columns to int
        boolean_columns = ['gave_consent', 'email_consent', 'sms_consent',
                          'data_tracking_consent', 'still_consented']
        for col in boolean_columns:
            X[col] = X[col].astype(int)
        
        self.feature_names = X.columns.tolist()
        return X, y

    def train_model(self, X, y):
        """Train the Random Forest model and save it"""
        print("Training model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Define the grid of parameters to test
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        }

        # Set up GridSearchCV
        # This will test 18 different combinations of parameters (2 * 3 * 3)
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            param_grid=param_grid,
            cv=3, n_jobs=-1, verbose=2, scoring='roc_auc'
        )

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # The best model is now stored in grid_search.best_estimator_
        self.model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store test data and results
        self.X_test = X_test
        self.y_test = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Store metadata
        self.model_metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'XGBClassifier',
            'accuracy': float(accuracy),
            'auc_score': float(auc_score),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def save_model(self):
        """Save the trained model, encoders, and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            # Save the main model
            joblib.dump(self.model, self.model_file)
            
            # Save the label encoders
            joblib.dump(self.label_encoders, self.encoders_file)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            print(f"✅ Model saved successfully!")
            print(f"   • Model: {self.model_file}")
            print(f"   • Encoders: {self.encoders_file}")
            print(f"   • Metadata: {self.metadata_file}")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise

    def load_model(self):
        """Load a previously trained model"""
        try:
            # Check if model files exist
            if not all(os.path.exists(f) for f in [self.model_file, self.encoders_file, self.metadata_file]):
                raise FileNotFoundError("Model files not found. Train and save the model first.")
            
            # Load the model
            self.model = joblib.load(self.model_file)
            
            # Load the label encoders
            self.label_encoders = joblib.load(self.encoders_file)
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Restore feature names
            self.feature_names = self.model_metadata['feature_names']
            
            print(f"✅ Model loaded successfully!")
            print(f"   • Trained on: {self.model_metadata['training_date']}")
            print(f"   • Accuracy: {self.model_metadata['accuracy']:.3f}")
            print(f"   • AUC Score: {self.model_metadata['auc_score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def is_model_saved(self):
        """Check if a saved model exists"""
        return all(os.path.exists(f) for f in [self.model_file, self.encoders_file, self.metadata_file])

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_metadata:
            return "No model loaded"
        
        return self.model_metadata

    def predict_conversion_probability(self, user_data):
        """Predict conversion probability for new users"""
        if self.model is None:
            raise ValueError("No model loaded. Load a trained model first.")
        
        X_new, _ = self.prepare_features(user_data)
        probabilities = self.model.predict_proba(X_new)[:, 1]
        predictions = self.model.predict(X_new)
        
        return predictions, probabilities

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("No model loaded. Load a trained model first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def calculate_roi_impact(self, user_features, avg_deal_value=2000, cost_per_lead=50):
        """Calculate ROI impact of using the model vs current approach"""
        if self.model is None:
            raise ValueError("No model loaded. Load a trained model first.")
        
        # Get predictions for all users
        X_all, y_all = self.prepare_features(user_features)
        all_probabilities = self.model.predict_proba(X_all)[:, 1]
        
        total_leads = len(user_features)
        
        # Current approach: Follow up with everyone
        current_cost = total_leads * cost_per_lead
        current_conversions = user_features['converted'].sum()
        current_revenue = current_conversions * avg_deal_value
        current_roi = (current_revenue - current_cost) / current_cost * 100 if current_cost > 0 else 0
        
        # Model approach scenarios
        scenarios = {}
        
        for top_percent in [0.1, 0.2, 0.3, 0.4, 0.5]:
            top_leads_count = int(len(user_features) * top_percent)
            
            # Get top leads by probability
            user_features_with_prob = user_features.copy()
            user_features_with_prob['conversion_probability'] = all_probabilities
            top_leads = user_features_with_prob.nlargest(top_leads_count, 'conversion_probability')
            
            model_cost = top_leads_count * cost_per_lead
            model_conversions = top_leads['converted'].sum()
            model_revenue = model_conversions * avg_deal_value
            model_roi = (model_revenue - model_cost) / model_cost * 100 if model_cost > 0 else 0
            
            efficiency = model_conversions / top_leads_count if top_leads_count > 0 else 0
            cost_savings = current_cost - model_cost
            
            scenarios[f'Top {int(top_percent*100)}%'] = {
                'leads_targeted': top_leads_count,
                'conversions': model_conversions,
                'conversion_rate': efficiency,
                'cost': model_cost,
                'revenue': model_revenue,
                'roi': model_roi,
                'cost_savings': cost_savings,
                'roi_improvement': model_roi - current_roi
            }
        
        return {
            'current_approach': {
                'leads_targeted': total_leads,
                'conversions': current_conversions,
                'conversion_rate': current_conversions / total_leads,
                'cost': current_cost,
                'revenue': current_revenue,
                'roi': current_roi
            },
            'model_scenarios': scenarios
        }

    def generate_lead_scoring_report(self, user_features):
        """Generate lead scoring report with business insights"""
        if self.model is None:
            raise ValueError("No model loaded. Load a trained model first.")
        
        # Get predictions for all users
        X_all, y_all = self.prepare_features(user_features)
        all_probabilities = self.model.predict_proba(X_all)[:, 1]
        
        # Create scoring tiers
        user_features_scored = user_features.copy()
        user_features_scored['conversion_probability'] = all_probabilities
        user_features_scored['lead_score'] = 'Low'
        user_features_scored.loc[user_features_scored['conversion_probability'] > 0.3, 'lead_score'] = 'Medium'
        user_features_scored.loc[user_features_scored['conversion_probability'] > 0.7, 'lead_score'] = 'High'
        
        # Generate scoring report
        scoring_summary = user_features_scored.groupby('lead_score').agg({
            'user_id': 'count',
            'converted': ['sum', 'mean'],
            'conversion_probability': ['mean', 'min', 'max']
        }).round(3)
        
        scoring_summary.columns = ['count', 'actual_conversions', 'actual_rate', 
                                 'avg_predicted_prob', 'min_prob', 'max_prob']
        
        # Sort by lead score priority
        score_order = ['High', 'Medium', 'Low']
        scoring_summary = scoring_summary.reindex(score_order)

        detailed_cols = [
            'user_id', 'primary_lead_quality', 'engagement_score',
            'conversion_probability', 'lead_score', 'converted',
            'primary_channel', 'total_duration', 'total_interactions', 'interaction_variety', 'primary_interaction_day' 
        ]
        return {
            'summary': scoring_summary,
            'detailed_data': user_features_scored[detailed_cols].sort_values('conversion_probability', ascending=False)
        }
