import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import random


class LeadScoringDataGenerator:
    """
    Generate realistic synthetic data for WhatsApp lead scoring.
    
    Enhanced to create CLEAR SEPARATION between hot/warm/cold leads:
    - Hot leads: Ready to buy, urgent, detailed questions
    - Warm leads: Interested, researching, asking questions (DISTINCT from cold)
    - Cold leads: Minimal engagement, vague, browsing
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Hot leads: Buying intent, urgency, specific questions
        self.hot_lead_templates = [
            "Hi, I'm interested in your product. Can we schedule a demo? I'm ready to buy.",
            "What are your pricing plans? I need this for my business ASAP.",
            "I saw your ad and I'm ready to buy. What's the next step?",
            "Can you send me more details? I have a meeting tomorrow about this.",
            "My company needs this solution. Do you offer enterprise pricing?",
            "I've been looking for something like this! When can we start?",
            "What's your onboarding process? We want to implement this month.",
            "Do you have case studies? I need to present this to my team.",
            "Can we schedule a meeting this week? I want to implement ASAP.",
            "I need pricing for 50 users. Can we discuss enterprise plans?",
            "This is exactly what we need. What's the implementation timeline?",
            "I'm ready to sign up. What payment options do you accept?",
        ]
        
        # Warm leads: Interest + specific questions (NOT vague)
        self.warm_lead_templates = [
            "Can I see a demo of the product?",
            "Tell me more about your product features and pricing.",
            "I'm interested but need to check my budget first. What's the cost?",
            "How does this compare to [competitor]? What makes you different?",
            "What features do you offer? I need X, Y, and Z capabilities.",
            "Is this suitable for small businesses? What plans do you have?",
            "I'm researching options for my team. Can you send detailed information?",
            "What's included in your basic plan? Do you have a trial?",
            "I'm evaluating solutions. Can you explain your key benefits?",
            "What kind of support do you provide? Is training included?",
            "How long does implementation take? What's the onboarding process?",
            "Do you integrate with other tools? I use X and Y currently.",
        ]
        
        # Cold leads: Vague, minimal effort, not specific
        self.cold_lead_templates = [
            "Just browsing",
            "Maybe later",
            "Not sure yet",
            "Ok",
            "Thanks",
            "I'll think about it",
            "Send me info",
            "What is this?",
            "Not interested",
            "Hello",
            "Hmm",
            "Later",
        ]
        
    def _generate_message(self, lead_score: str) -> str:
        """Generate realistic message based on lead quality."""
        if lead_score == "hot":
            base = random.choice(self.hot_lead_templates)
            # Hot leads often ask multiple questions
            if random.random() > 0.4:
                followup = random.choice([
                    ' Also, what are the payment terms?',
                    ' Do you have customer support?',
                    ' Can I get a volume discount?',
                    ' What integrations do you offer?'
                ])
                base += followup
                
        elif lead_score == "warm":
            base = random.choice(self.warm_lead_templates)
            # Warm leads sometimes ask follow-up questions
            if random.random() > 0.6:
                followup = random.choice([
                    ' Also, do you have any customer reviews?',
                    ' How long is the contract?',
                    ' Can I speak with a sales rep?'
                ])
                base += followup
                
        else:  # cold
            base = random.choice(self.cold_lead_templates)
            
        return base
    
    def _generate_conversation_metrics(self, lead_score: str) -> Dict:
        """
        Generate conversation metrics with CLEARER separation between categories.
        
        Key changes:
        - Warm leads have DISTINCTLY better metrics than cold
        - More separation in message counts
        - User-initiated is a stronger signal
        """
        
        if lead_score == "hot":
            # Hot leads: Long, engaged, quick responses
            duration = np.random.gamma(shape=6, scale=3)  # Mean ~18 min (increased)
            response_time = np.random.exponential(scale=25)  # Mean 25 sec (faster)
            msg_count = np.random.poisson(lam=10) + 5  # 5-20 messages (increased)
            user_initiated = np.random.choice([True, False], p=[0.85, 0.15])  # Usually initiated
            returning = np.random.choice([True, False], p=[0.3, 0.7])
            
        elif lead_score == "warm":
            # Warm leads: MEDIUM engagement (clear middle ground)
            duration = np.random.gamma(shape=4, scale=1.8)  # Mean ~7 min (increased)
            response_time = np.random.exponential(scale=70)  # Mean 70 sec (faster than before)
            msg_count = np.random.poisson(lam=5) + 3  # 3-10 messages (increased)
            user_initiated = np.random.choice([True, False], p=[0.75, 0.25])  # Usually initiated
            returning = np.random.choice([True, False], p=[0.15, 0.85])
            
        else:  # cold
            # Cold leads: Brief, slow, often bot-initiated
            duration = np.random.gamma(shape=1.2, scale=1)  # Mean ~1.2 min (decreased)
            response_time = np.random.exponential(scale=200)  # Mean 200 sec (slower)
            msg_count = np.random.poisson(lam=1.5) + 1  # 1-4 messages (decreased)
            user_initiated = np.random.choice([True, False], p=[0.15, 0.85])  # Rarely initiated
            returning = np.random.choice([True, False], p=[0.02, 0.98])
        
        return {
            "messages_in_session": max(1, msg_count),
            "conversation_duration_minutes": max(0.5, duration),
            "user_response_time_avg_seconds": max(5, response_time),
            "user_initiated_conversation": user_initiated,
            "is_returning_customer": returning,
        }
    
    def _assign_time_of_day(self, hour: int) -> str:
        """
        Categorize time of day.
        NOTE: Your API only accepts 'business_hours' currently.
        """
        # For now, return business_hours for all since API validates strictly
        return "business_hours"
        
        # Future: when API supports multiple values
        # if 9 <= hour < 18:
        #     return "business_hours"
        # elif 6 <= hour < 9:
        #     return "morning"
        # elif 18 <= hour < 22:
        #     return "evening"
        # else:
        #     return "night"
    
    def generate_dataset(
        self,
        n_samples: int = 1000,
        hot_ratio: float = 0.20,    # Increased from 0.15
        warm_ratio: float = 0.40,   # Increased from 0.35
        base_date: datetime = None
    ) -> pd.DataFrame:
        """
        Generate synthetic lead scoring dataset with better class separation.
        
        Args:
            n_samples: Number of leads to generate
            hot_ratio: Proportion of hot leads (default 20%)
            warm_ratio: Proportion of warm leads (default 40%)
            base_date: Starting date for timestamps
            
        Returns:
            DataFrame with synthetic lead data
        """
        
        if base_date is None:
            base_date = datetime.now() - timedelta(days=90)
        
        cold_ratio = 1 - hot_ratio - warm_ratio
        
        print(f"Generating dataset with distribution:")
        print(f"  Hot:  {hot_ratio*100:.0f}% ({int(n_samples*hot_ratio)} samples)")
        print(f"  Warm: {warm_ratio*100:.0f}% ({int(n_samples*warm_ratio)} samples)")
        print(f"  Cold: {cold_ratio*100:.0f}% ({int(n_samples*cold_ratio)} samples)")
        
        # Generate lead scores with specified distribution
        lead_scores = np.random.choice(
            ["hot", "warm", "cold"],
            size=n_samples,
            p=[hot_ratio, warm_ratio, cold_ratio]
        )
        
        data = []
        
        for i, score in enumerate(lead_scores):
            # Generate timestamp
            days_offset = np.random.randint(0, 90)
            timestamp = base_date + timedelta(
                days=days_offset,
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Generate conversation metrics
            metrics = self._generate_conversation_metrics(score)
            
            # Generate message
            message = self._generate_message(score)
            
            # Compile record
            record = {
                "lead_id": f"lead_{i:05d}",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "user_msg": message,
                "score": score,
                **metrics,
                "time_of_day": self._assign_time_of_day(timestamp.hour),
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add controlled noise (reduced from 10% to 5%)
        noise_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        for idx in noise_indices:
            if df.loc[idx, "score"] == "hot":
                # Few hot leads might have slightly slower response times
                df.loc[idx, "user_response_time_avg_seconds"] *= np.random.uniform(1.5, 2.5)
            elif df.loc[idx, "score"] == "warm":
                # Few warm leads might have odd metrics
                if random.random() > 0.5:
                    df.loc[idx, "user_response_time_avg_seconds"] *= np.random.uniform(1.3, 2.0)
                else:
                    df.loc[idx, "conversation_duration_minutes"] *= np.random.uniform(0.7, 1.3)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str = "data/whatsapp_leads.csv"):
        """Save dataset to CSV with summary statistics."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ Saved {len(df)} leads to {filepath}")
        print(f"{'='*80}")
        
        # Print distribution
        print(f"\n📊 Lead Distribution:")
        dist = df["score"].value_counts()
        for category in ['hot', 'warm', 'cold']:
            if category in dist.index:
                count = dist[category]
                pct = (count / len(df)) * 100
                print(f"  {category.capitalize():6s}: {count:4d} ({pct:5.1f}%)")
        
        # Print average metrics by category
        print(f"\n📈 Average Metrics by Category:")
        summary = df.groupby("score").agg({
            "messages_in_session": "mean",
            "conversation_duration_minutes": "mean",
            "user_response_time_avg_seconds": "mean",
            "user_initiated_conversation": lambda x: (x.sum() / len(x)) * 100,
            "is_returning_customer": lambda x: (x.sum() / len(x)) * 100,
        }).round(2)
        
        summary.columns = ['Msgs', 'Duration(min)', 'Response(sec)', 'User Init(%)', 'Returning(%)']
        print(summary)
        
        # Check for clear separation
        print(f"\n🎯 Separation Quality Check:")
        for metric in ['messages_in_session', 'conversation_duration_minutes', 'user_response_time_avg_seconds']:
            cold_avg = df[df['score']=='cold'][metric].mean()
            warm_avg = df[df['score']=='warm'][metric].mean()
            hot_avg = df[df['score']=='hot'][metric].mean()
            
            if metric == 'user_response_time_avg_seconds':
                # Lower is better, so check descending
                sep_score = "✓ Good" if cold_avg > warm_avg > hot_avg else "✗ Needs work"
            else:
                # Higher is better
                sep_score = "✓ Good" if hot_avg > warm_avg > cold_avg else "✗ Needs work"
            
            print(f"  {metric:40s}: {sep_score}")


# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED LEAD SCORING DATA GENERATOR".center(80))
    print("="*80 + "\n")
    
    # Initialize generator
    generator = LeadScoringDataGenerator(seed=42)
    
    # Generate training dataset with better distribution
    print("Generating TRAINING dataset...")
    df = generator.generate_dataset(
        n_samples=2000,
        hot_ratio=0.20,   # 20% hot leads (increased)
        warm_ratio=0.40,  # 40% warm leads (increased for better learning)
        # 40% cold leads (decreased)
    )
    
    # Preview sample data
    print(f"\n📋 Sample Data Preview:")
    print(df.head(3).to_string())
    
    # Save training data
    generator.save_dataset(df, "data/whatsapp_leads.csv")
    
    # Generate test set (same distribution)
    print(f"\n{'='*80}")
    print("Generating TEST dataset...")
    test_df = generator.generate_dataset(
        n_samples=500,
        hot_ratio=0.20,
        warm_ratio=0.40,
    )
    generator.save_dataset(test_df, "data/whatsapp_leads_test.csv")
    
    print(f"\n{'='*80}")
    print("✅ DATA GENERATION COMPLETE!".center(80))
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the separation quality metrics above")
    print("  2. Train your model: train_and_save_lead_score('data/whatsapp_leads.csv')")
    print("  3. Test on the test set: 'data/whatsapp_leads_test.csv'")
    print("="*80)