import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

np.random.seed(42)
random.seed(42)

class EnhancedConversionDataGenerator:
    def __init__(self, num_users=1000, num_events=5):
        self.num_users = num_users
        self.num_events = num_events
        self.base_date = datetime(2025, 1, 1)
        
        self.interaction_types = [
            "registration_booth_visit", "demo_request_form", "contact_info_exchange",
            "brochure_download", "newsletter_signup", "product_trial_request",
            "pricing_inquiry", "consultation_booking"
        ]
        
        self.conversion_types = [
            "software_purchase", "service_signup", "trial_to_paid", "consultation_booked",
            "demo_scheduled", "newsletter_confirmed", "whitepaper_download", "webinar_registration"
        ]
        
        self.communication_types = [
            "welcome_email", "product_demo_invite", "pricing_follow_up", "case_study_share",
            "newsletter_send", "consultation_reminder"
        ]
        
        self.nz_locations = [
            "Auckland", "Wellington", "Christchurch", "Hamilton", "Tauranga",
            "Dunedin", "Napier", "Palmerston North", "Nelson", "Rotorua"
        ]

    def generate_user_ids(self):
        return [hashlib.md5(f"user_{i}@company.com".encode()).hexdigest()[:12] for i in range(self.num_users)]

    def generate_event_interactions(self):
        user_ids = self.generate_user_ids()
        interactions = []
        
        for _ in range(int(self.num_users * 1.3)):
            user_id = np.random.choice(user_ids)
            event_date = self.base_date + timedelta(days=random.randint(0, 30))
            
            interaction = {
                'user_id': user_id,
                'event_id': f"event_{random.randint(1, self.num_events)}",
                'interaction_type': np.random.choice(self.interaction_types),
                'timestamp': event_date + timedelta(hours=random.randint(9, 17), minutes=random.randint(0, 59)),
                'interaction_channel': np.random.choice(['in_person_booth', 'qr_code_scan', 'staff_referral', 'self_service_kiosk']),
                'lead_quality': np.random.choice(['hot', 'warm', 'cold']),
                'contact_consent_given': np.random.choice([True, False], p=[0.7, 0.3]),
                'interest_level': random.randint(1, 5),
                'staff_notes': np.random.choice(['highly_engaged', 'specific_use_case', 'budget_confirmed', 'decision_maker', 'evaluating_options']),
                'location': np.random.choice(self.nz_locations),
                'duration_minutes': random.randint(1, 30)
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)

    def generate_consent_preferences(self, interactions_df):
        consent_data = []
        
        for user_id in interactions_df['user_id'].unique():
            first_interaction = interactions_df[interactions_df['user_id'] == user_id].iloc[0]
            
            if first_interaction['contact_consent_given']:
                consent = {
                    'user_id': user_id,
                    'email_consent': np.random.choice([True, False], p=[0.85, 0.15]),
                    'sms_consent': np.random.choice([True, False], p=[0.3, 0.7]),
                    'data_tracking_consent': np.random.choice([True, False], p=[0.6, 0.4]),
                    'consent_date': first_interaction['timestamp'],
                    'consent_method': np.random.choice(['event_form', 'booth_tablet', 'staff_collection']),
                    'privacy_policy_version': 'v2.1_2025',
                    'opt_out_date': None if np.random.random() > 0.05 else first_interaction['timestamp'] + timedelta(days=np.random.randint(1, 60))
                }
                consent_data.append(consent)
        
        return pd.DataFrame(consent_data)

    def generate_communication_history(self, consent_df):
        communications = []
        
        for _, row in consent_df.iterrows():
            if row['email_consent'] and pd.isna(row['opt_out_date']):
                for i in range(random.randint(1, 5)):
                    comm_date = row['consent_date'] + timedelta(days=i*3 + random.randint(0, 3))
                    
                    communication = {
                        'user_id': row['user_id'],
                        'communication_type': np.random.choice(self.communication_types),
                        'communication_date': comm_date,
                        'response_status': np.random.choice(['opened', 'clicked', 'replied', 'no_response'], p=[0.3, 0.15, 0.05, 0.5]),
                        'campaign_id': f"campaign_{np.random.randint(1, 10)}",
                        'content_theme': np.random.choice(['product_demo', 'pricing_offer', 'case_study', 'industry_insights', 'feature_update'])
                    }
                    communications.append(communication)
        
        return pd.DataFrame(communications)

    def generate_conversions(self, interactions_df, communications_df):
        conversions = []
        interaction_users = interactions_df['user_id'].unique()
        converting_users = random.sample(list(interaction_users), int(len(interaction_users) * 0.2))
        
        for user_id in converting_users:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            first_interaction = user_interactions['timestamp'].min()
            user_comms = communications_df[communications_df['user_id'] == user_id]
            location = user_interactions['location'].iloc[0]
            
            channel = 'email_followup' if not user_comms.empty and random.random() < 0.7 else 'event_direct'
            
            conversion = {
                'user_id': user_id,
                'conversion_type': random.choice(self.conversion_types),
                'conversion_value': round(random.uniform(50, 5000), 2),
                'conversion_date': first_interaction + timedelta(days=random.randint(1, 30)),
                'attribution_channel': channel,
                'customer_segment': random.choice(['new_customer', 'existing_customer', 'prospect']),
                'product_category': random.choice(['enterprise_software', 'consulting_services', 'training_program', 'subscription_service']),
                'days_to_conversion': (first_interaction + timedelta(days=random.randint(1, 30)) - first_interaction).days,
                'location': location,
                'conversion_channel': channel
            }
            conversions.append(conversion)
        
        return pd.DataFrame(conversions)

    def generate_complete_dataset(self):
        print("\nGenerating enhanced conversion tracking dataset...")
        
        interactions_df = self.generate_event_interactions()
        consent_df = self.generate_consent_preferences(interactions_df)
        communications_df = self.generate_communication_history(consent_df)
        conversions_df = self.generate_conversions(interactions_df, communications_df)
        
        print(f"Dataset Stats:")
        print(f"Total Interactions: {len(interactions_df)}")
        print(f"Users with Consent: {len(consent_df)}")
        print(f"Total Communications: {len(communications_df)}")
        print(f"Total Conversions: {len(conversions_df)}")
        
        return {
            'interactions': interactions_df,
            'consent': consent_df,
            'communications': communications_df,
            'conversions': conversions_df
        }
