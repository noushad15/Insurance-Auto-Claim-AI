#!/usr/bin/env python3
"""
Script to populate the database with a large amount of sample claim data for testing
"""

import sys
import os
import random
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from database import DatabaseManager

def random_name():
    first_names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George', 'Helen', 'Sam', 'Olivia', 'Liam', 'Emma', 'Noah', 'Ava', 'Mason', 'Sophia', 'Logan', 'Isabella']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson']
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_diagnosis():
    diagnoses = [
        'Fractured arm', 'Broken leg', 'Sprained ankle', 'Concussion', 'Chest pain',
        'Appendicitis', 'Heart attack', 'Stroke', 'Diabetes management', 'Hypertension',
        'Routine checkup', 'Flu shot', 'Dental cleaning', 'Eye exam', 'Physical therapy',
        'Migraine', 'Asthma attack', 'Allergic reaction', 'Back pain', 'Ear infection'
    ]
    return random.choice(diagnoses)

def random_provider():
    providers = [
        'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis',
        'City Hospital', 'Medical Center', 'Urgent Care', 'Specialty Clinic', 'Emergency Hospital'
    ]
    return random.choice(providers)

def random_status():
    # Weighted: 60% auto_approved, 30% manual_review, 10% rejected
    r = random.random()
    if r < 0.6:
        return 'auto_approved', 'Auto-Approve', 'Auto-approve claim'
    elif r < 0.9:
        return 'manual_review', 'Manual Review', 'Manual review required'
    else:
        return 'rejected', 'Rejected', 'Claim rejected'

def random_amount(diagnosis):
    if 'routine' in diagnosis.lower() or 'checkup' in diagnosis.lower():
        return round(random.uniform(50, 300), 2)
    elif 'fracture' in diagnosis.lower() or 'broken' in diagnosis.lower():
        return round(random.uniform(2000, 8000), 2)
    elif 'surgery' in diagnosis.lower() or 'appendicitis' in diagnosis.lower():
        return round(random.uniform(5000, 15000), 2)
    elif 'heart attack' in diagnosis.lower() or 'stroke' in diagnosis.lower():
        return round(random.uniform(10000, 25000), 2)
    else:
        return round(random.uniform(500, 3000), 2)

def populate_sample_data():
    db_manager = DatabaseManager()
    print("Populating database with sample claims for the last 30 days...")
    total_claims = 0
    for days_ago in range(0, 30):
        date = datetime.now() - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        num_claims = random.randint(10, 30)
        for _ in range(num_claims):
            name = random_name()
            diagnosis = random_diagnosis()
            provider = random_provider()
            status, prediction, recommended_action = random_status()
            amount = random_amount(diagnosis)
            confidence = round(random.uniform(60, 99), 1)
            risk_score = round(random.uniform(10, 90), 1)
            processing_time = round(random.uniform(1.0, 5.0), 2)
            policy_number = f"POL-{random.randint(10000, 99999)}"
            claim = {
                'patient_name': name,
                'claim_date': date_str,
                'diagnosis': diagnosis,
                'amount': amount,
                'provider': provider,
                'policy_number': policy_number,
                'service_date': date_str,
                'status': status,
                'prediction': prediction,
                'confidence': confidence,
                'risk_score': risk_score,
                'recommended_action': recommended_action,
                'extracted_fields': {'name': name, 'diagnosis': diagnosis, 'amount': str(amount)},
                'raw_text': f'Sample claim text for {name}',
                'processing_time': processing_time
            }
            try:
                db_manager.add_claim(claim)
                total_claims += 1
            except Exception as e:
                print(f"Error adding claim for {name} on {date_str}: {e}")
    print(f"✅ Added {total_claims} sample claims over the last 30 days.")
    # Show summary
    claims = db_manager.get_claims()
    analytics = db_manager.get_analytics()
    print(f"\n📊 Database Summary:")
    print(f"  Total claims: {len(claims)}")
    print(f"  Analytics records: {len(analytics.get('daily_analytics', []))}")
    if claims:
        status_counts = {}
        for claim in claims:
            status = claim['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        print(f"  Claims by status:")
        for status, count in status_counts.items():
            print(f"    {status}: {count}")
if __name__ == "__main__":
    populate_sample_data() 