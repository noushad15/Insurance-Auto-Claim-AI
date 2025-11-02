import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "auto_claim_ai.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Claims table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS claims (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim_id TEXT UNIQUE NOT NULL,
                        patient_name TEXT NOT NULL,
                        claim_date TEXT NOT NULL,
                        diagnosis TEXT,
                        amount REAL NOT NULL,
                        provider TEXT,
                        policy_number TEXT,
                        service_date TEXT,
                        status TEXT DEFAULT 'pending',
                        prediction TEXT,
                        confidence REAL,
                        risk_score REAL,
                        recommended_action TEXT,
                        extracted_fields TEXT,
                        raw_text TEXT,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Analytics table (remove avg_* columns)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        claims_processed INTEGER DEFAULT 0,
                        auto_approved INTEGER DEFAULT 0,
                        manual_review INTEGER DEFAULT 0,
                        rejected INTEGER DEFAULT 0,
                        total_amount REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Model training history
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_training (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT NOT NULL,
                        training_date TEXT NOT NULL,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        training_samples INTEGER,
                        test_samples INTEGER,
                        model_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert default settings
                default_settings = [
                    ('confidence_threshold', '75', 'Minimum confidence for auto-approval'),
                    ('risk_threshold', '60', 'Risk score above which manual review is required'),
                    ('max_auto_approval_amount', '5000', 'Maximum amount for auto-approval'),
                    ('processing_timeout', '120', 'Maximum processing time in seconds'),
                    ('auto_approval_enabled', 'true', 'Enable auto-approval feature'),
                    ('notification_enabled', 'true', 'Enable notifications for manual reviews')
                ]
                
                for key, value, description in default_settings:
                    cursor.execute('''
                        INSERT OR IGNORE INTO settings (key, value, description)
                        VALUES (?, ?, ?)
                    ''', (key, value, description))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_claim(self, claim_data: Dict[str, Any]) -> int:
        """Add a new claim to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Generate unique claim ID
                claim_id = f"CLM-{uuid.uuid4().hex[:12].upper()}"
                
                cursor.execute('''
                    INSERT INTO claims (
                        claim_id, patient_name, claim_date, diagnosis, amount,
                        provider, policy_number, service_date, status, prediction,
                        confidence, risk_score, recommended_action, extracted_fields,
                        raw_text, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    claim_id,
                    claim_data.get('patient_name', ''),
                    claim_data.get('claim_date', ''),
                    claim_data.get('diagnosis', ''),
                    claim_data.get('amount', 0.0),
                    claim_data.get('provider', ''),
                    claim_data.get('policy_number', ''),
                    claim_data.get('service_date', ''),
                    claim_data.get('status', 'pending'),
                    claim_data.get('prediction', ''),
                    claim_data.get('confidence', 0.0),
                    claim_data.get('risk_score', 0.0),
                    claim_data.get('recommended_action', ''),
                    json.dumps(claim_data.get('extracted_fields', {})),
                    claim_data.get('raw_text', ''),
                    claim_data.get('processing_time', 0.0)
                ))
                
                claim_id_db = cursor.lastrowid
                conn.commit()
                
                # Update analytics
                self._update_analytics(claim_data)
                
                logger.info(f"Claim added successfully with ID: {claim_id}")
                return claim_id_db
                
        except Exception as e:
            logger.error(f"Error adding claim: {e}")
            raise
    
    def get_claims(self, limit: int = 100, status: Optional[str] = None) -> List[Dict]:
        """Get claims from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM claims"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                claims = []
                for row in rows:
                    claim = dict(row)
                    # Parse JSON fields
                    if claim['extracted_fields']:
                        claim['extracted_fields'] = json.loads(claim['extracted_fields'])
                    claims.append(claim)
                
                return claims
                
        except Exception as e:
            logger.error(f"Error getting claims: {e}")
            return []
    
    def update_claim_status(self, claim_id: int, status: str, prediction: str = None, confidence: float = None):
        """Update claim status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]
                
                if prediction:
                    update_fields.append("prediction = ?")
                    params.append(prediction)
                
                if confidence is not None:
                    update_fields.append("confidence = ?")
                    params.append(confidence)
                
                params.append(claim_id)
                
                query = f"UPDATE claims SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()
                
                logger.info(f"Claim {claim_id} status updated to {status}")
                
        except Exception as e:
            logger.error(f"Error updating claim status: {e}")
            raise
    
    def get_analytics(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get analytics data (calculate averages live from claims)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get daily analytics
                query = "SELECT * FROM analytics"
                params = []
                
                if start_date and end_date:
                    query += " WHERE date BETWEEN ? AND ?"
                    params.extend([start_date, end_date])
                
                query += " ORDER BY date DESC"
                cursor.execute(query, params)
                analytics_rows = cursor.fetchall()
                
                # Get summary statistics and averages live from claims
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_claims,
                        SUM(CASE WHEN status = 'auto_approved' THEN 1 ELSE 0 END) as auto_approved,
                        SUM(CASE WHEN status = 'manual_review' THEN 1 ELSE 0 END) as manual_review,
                        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                        AVG(confidence) as avg_confidence,
                        AVG(risk_score) as avg_risk_score,
                        AVG(processing_time) as avg_processing_time,
                        SUM(amount) as total_amount
                    FROM claims
                ''')
                summary_row = cursor.fetchone()
                summary = {
                    'total_claims': summary_row[0] or 0,
                    'auto_approved': summary_row[1] or 0,
                    'manual_review': summary_row[2] or 0,
                    'rejected': summary_row[3] or 0,
                    'avg_confidence': summary_row[4] or 0.0,
                    'avg_risk_score': summary_row[5] or 0.0,
                    'avg_processing_time': summary_row[6] or 0.0,
                    'total_amount': summary_row[7] or 0.0
                }
                
                # Get claims by status
                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM claims
                    GROUP BY status
                ''')
                
                status_counts = {}
                for row in cursor.fetchall():
                    status, count = row
                    status_counts[status] = count
                
                # Get claims by amount range
                cursor.execute('''
                    SELECT 
                        CASE 
                            WHEN amount < 1000 THEN '0-1K'
                            WHEN amount < 5000 THEN '1K-5K'
                            WHEN amount < 10000 THEN '5K-10K'
                            ELSE '10K+'
                        END as range,
                        COUNT(*) as count
                    FROM claims
                    GROUP BY range
                ''')
                
                amount_ranges = {}
                for row in cursor.fetchall():
                    range_name, count = row
                    amount_ranges[range_name] = count
                
                return {
                    'daily_analytics': analytics_rows,
                    'summary': summary,
                    'status_counts': status_counts,
                    'amount_ranges': amount_ranges
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}
    
    def get_settings(self) -> Dict[str, str]:
        """Get all settings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM settings")
                rows = cursor.fetchall()
                return dict(rows)
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return {}
    
    def update_setting(self, key: str, value: str):
        """Update a setting"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE settings 
                    SET value = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE key = ?
                ''', (value, key))
                conn.commit()
                logger.info(f"Setting {key} updated to {value}")
        except Exception as e:
            logger.error(f"Error updating setting: {e}")
            raise
    
    def _update_analytics(self, claim_data: Dict):
        """Update analytics for a new claim (no avg columns)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Check if analytics entry exists for today
                cursor.execute("SELECT * FROM analytics WHERE date = ?", (today,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry
                    cursor.execute('''
                        UPDATE analytics SET
                            claims_processed = claims_processed + 1,
                            auto_approved = auto_approved + ?,
                            manual_review = manual_review + ?,
                            rejected = rejected + ?,
                            total_amount = total_amount + ?
                        WHERE date = ?
                    ''', (
                        1 if claim_data.get('status') == 'auto_approved' else 0,
                        1 if claim_data.get('status') == 'manual_review' else 0,
                        1 if claim_data.get('status') == 'rejected' else 0,
                        claim_data.get('amount', 0),
                        today
                    ))
                else:
                    # Create new entry
                    cursor.execute('''
                        INSERT INTO analytics (
                            date, claims_processed, auto_approved, manual_review,
                            rejected, total_amount
                        ) VALUES (?, 1, ?, ?, ?, ?)
                    ''', (
                        today,
                        1 if claim_data.get('status') == 'auto_approved' else 0,
                        1 if claim_data.get('status') == 'manual_review' else 0,
                        1 if claim_data.get('status') == 'rejected' else 0,
                        claim_data.get('amount', 0)
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
    
    def add_training_record(self, training_data: Dict):
        """Add model training record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO model_training (
                        model_version, training_date, accuracy, precision,
                        recall, f1_score, training_samples, test_samples, model_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    training_data.get('model_version', '1.0'),
                    training_data.get('training_date', datetime.now().strftime('%Y-%m-%d')),
                    training_data.get('accuracy', 0.0),
                    training_data.get('precision', 0.0),
                    training_data.get('recall', 0.0),
                    training_data.get('f1_score', 0.0),
                    training_data.get('training_samples', 0),
                    training_data.get('test_samples', 0),
                    training_data.get('model_path', '')
                ))
                
                conn.commit()
                logger.info("Training record added successfully")
                
        except Exception as e:
            logger.error(f"Error adding training record: {e}")
            raise
    
    def get_training_history(self) -> List[Dict]:
        """Get model training history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM model_training 
                    ORDER BY training_date DESC 
                    LIMIT 10
                ''')
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting training history: {e}")
            return [] 