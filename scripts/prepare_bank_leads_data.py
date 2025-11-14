"""
Bank Leads Conversion Dataset Adapter
======================================
Converts bank loan leads dataset to WhatsApp lead score format.

Features from dataset:
- Approved (Target): 1/0 whether loan approved (qualified lead)
- Contacted: Y/N if contact verified
- Source: Lead source (search, display, email, etc)
- Monthly_Income, Loan_Amount, EMI: Financial metrics
- Lead_Creation_Date: Timing information

Usage:
    python scripts/prepare_bank_leads_data.py \
        --input data/kaggle/train.csv \
        --output data/whatsapp_leads.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def prepare_bank_leads_data(input_path: str, output_path: str):
    """Convert bank leads dataset to WhatsApp lead score format."""
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    if 'Approved' not in df.columns:
        print("\n⚠️  'Approved' column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nLooking for alternative target columns...")
        
        # Try to find alternative target column
        target_candidates = ['approved', 'qualified', 'converted', 'outcome', 'status']
        target_col = None
        for col in df.columns:
            if any(candidate in col.lower() for candidate in target_candidates):
                target_col = col
                print(f"Found potential target column: {target_col}")
                break
        
        if not target_col:
            print("❌ Cannot find target column. Please specify manually.")
            return
    else:
        target_col = 'Approved'
    
    print(f"\n🎯 Using '{target_col}' as target column")
    print(f"Target distribution:")
    print(df[target_col].value_counts())
    
    print("\n🔄 Converting to WhatsApp lead format...")
    
    adapted_data = []
    
    for idx, row in df.iterrows():
        # Skip rows with missing target
        if pd.isna(row.get(target_col)):
            continue
        
        # ===== TARGET: score =====
        target_value = row[target_col]
        if isinstance(target_value, str):
            score = 1 if target_value.lower() in ['1', 'yes', 'y', 'approved', 'qualified'] else 0
        else:
            score = int(target_value)
        
        # ===== FEATURE 1: messages_in_session =====
        # Infer from contact status and source
        contacted = str(row.get('Contacted', 'N')).upper()
        
        if score == 1:  # Qualified leads have more engagement
            if contacted == 'Y':
                messages_in_session = np.random.randint(8, 16)
            else:
                messages_in_session = np.random.randint(5, 10)
        else:  # Unqualified leads have less engagement
            if contacted == 'Y':
                messages_in_session = np.random.randint(3, 8)
            else:
                messages_in_session = np.random.randint(1, 5)
        
        # ===== FEATURE 2: user_msg =====
        # Create message based on loan details
        source = str(row.get('Source', 'Direct'))
        loan_amount = row.get('Loan_Amount', 0)
        monthly_income = row.get('Monthly_Income', 0)
        
        # Create realistic message
        if score == 1:
            messages = [
                f"I need a loan of ${loan_amount:.0f}. Can you help? (from {source})",
                f"What's the interest rate for ${loan_amount:.0f}? I'm interested.",
                f"I want to apply for a loan. My income is ${monthly_income:.0f}/month.",
                f"Can you process my loan application? Amount: ${loan_amount:.0f}",
                f"I'm ready to apply. What documents do you need? (via {source})"
            ]
        else:
            messages = [
                f"Just checking loan options. (from {source})",
                f"What are your rates?",
                f"Tell me about loans.",
                f"Not sure if I qualify.",
                f"Maybe later. Thanks."
            ]
        
        user_msg = np.random.choice(messages)[:500]
        
        # ===== FEATURE 3: conversation_duration_minutes =====
        # Based on qualification and contact status
        if score == 1:
            if contacted == 'Y':
                conversation_duration_minutes = float(np.random.lognormal(2.8, 0.4))  # ~15-25 min
            else:
                conversation_duration_minutes = float(np.random.lognormal(2.3, 0.5))  # ~8-15 min
        else:
            if contacted == 'Y':
                conversation_duration_minutes = float(np.random.lognormal(1.8, 0.6))  # ~5-10 min
            else:
                conversation_duration_minutes = float(np.random.lognormal(0.9, 0.7))  # ~1-5 min
        
        conversation_duration_minutes = round(np.clip(conversation_duration_minutes, 0.5, 45), 2)
        
        # ===== FEATURE 4: user_response_time_avg_seconds =====
        # Qualified leads respond faster
        if score == 1:
            user_response_time_avg_seconds = float(np.random.lognormal(3.7, 0.5))  # ~30-80 sec
        else:
            user_response_time_avg_seconds = float(np.random.lognormal(4.8, 0.6))  # ~80-300 sec
        
        user_response_time_avg_seconds = round(np.clip(user_response_time_avg_seconds, 15, 600), 1)
        
        # ===== FEATURE 5: user_initiated_conversation =====
        # Check source - Direct/Referral = user initiated, Display/Email = company initiated
        source_lower = source.lower()
        if any(keyword in source_lower for keyword in ['direct', 'referral', 'search', 'organic']):
            user_initiated_conversation = True
        elif any(keyword in source_lower for keyword in ['display', 'email', 'campaign']):
            user_initiated_conversation = False
        else:
            # Default based on score
            user_initiated_conversation = np.random.choice([True, False], p=[0.6, 0.4] if score == 1 else [0.4, 0.6])
        
        # ===== FEATURE 6: is_returning_customer =====
        # Check if has existing EMI (indicates existing customer)
        existing_emi = row.get('Existing_EMI', 0)
        if pd.isna(existing_emi):
            existing_emi = 0
        
        is_returning_customer = (existing_emi > 0)
        
        # ===== FEATURE 7: time_of_day =====
        # Parse Lead_Creation_Date if available
        lead_date = row.get('Lead_Creation_Date', None)
        
        if pd.notna(lead_date):
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d', '%d-%m-%Y']:
                    try:
                        dt = pd.to_datetime(lead_date, format=fmt)
                        hour = dt.hour
                        break
                    except:
                        continue
                else:
                    hour = 12  # Default
            except:
                hour = 12
        else:
            # Infer from score - qualified leads tend to be created during business hours
            if score == 1:
                hour = np.random.choice(range(9, 18), p=[0.08, 0.1, 0.12, 0.15, 0.13, 0.12, 0.11, 0.1, 0.09])
            else:
                hour = np.random.randint(8, 21)
        
        if 9 <= hour <= 17:
            time_of_day = 'business_hours'
        elif (8 <= hour < 9) or (17 < hour <= 20):
            time_of_day = 'extended_hours'
        else:
            time_of_day = 'off_hours'
        
        # Create output row
        adapted_data.append({
            'messages_in_session': messages_in_session,
            'user_msg': user_msg,
            'conversation_duration_minutes': conversation_duration_minutes,
            'user_response_time_avg_seconds': user_response_time_avg_seconds,
            'user_initiated_conversation': user_initiated_conversation,
            'is_returning_customer': is_returning_customer,
            'time_of_day': time_of_day,
            'score': score
        })
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Create DataFrame
    output_df = pd.DataFrame(adapted_data)
    
    # Shuffle
    output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print summary
    print(f"\n✅ Conversion complete!")
    print(f"📊 Output: {len(output_df)} samples")
    print(f"📊 Qualified: {output_df['score'].sum()} ({output_df['score'].mean()*100:.1f}%)")
    print(f"📊 Unqualified: {len(output_df) - output_df['score'].sum()}")
    
    # Show sample
    print("\n📋 Sample data:")
    print(output_df[['user_msg', 'messages_in_session', 'score']].head(3))
    
    # Check data quality
    print("\n🔍 Data Quality Check:")
    print(f"Messages range: {output_df['messages_in_session'].min()} - {output_df['messages_in_session'].max()}")
    print(f"Duration range: {output_df['conversation_duration_minutes'].min():.1f} - {output_df['conversation_duration_minutes'].max():.1f} min")
    print(f"User initiated: {output_df['user_initiated_conversation'].sum()} ({output_df['user_initiated_conversation'].mean()*100:.1f}%)")
    print(f"Returning customers: {output_df['is_returning_customer'].sum()} ({output_df['is_returning_customer'].mean()*100:.1f}%)")
    print(f"\nTime of day distribution:")
    print(output_df['time_of_day'].value_counts())
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"\n🚀 Next step: Train model!")
    print(f"curl -X POST http://localhost:8082/ml/v1/train_score \\")
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"data_path": "{output_path}", "grid_search": false}}\'')


def main():
    parser = argparse.ArgumentParser(description='Prepare bank leads data for lead scoring')
    parser.add_argument('--input', type=str, required=True, help='Path to bank leads CSV (train.csv)')
    parser.add_argument('--output', type=str, default='data/whatsapp_leads.csv', help='Output path')
    
    args = parser.parse_args()
    
    prepare_bank_leads_data(args.input, args.output)


if __name__ == "__main__":
    main()