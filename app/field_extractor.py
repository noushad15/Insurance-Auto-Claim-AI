import re
import spacy
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldExtractor:
    def __init__(self):
        """Initialize field extractor with NLP model"""
        try:
            # Load English language model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define field patterns
        self.field_patterns = {
            'name': [
                r'(?i)(?:patient|insured|member|name):\s*([A-Za-z\s]+)',
                r'(?i)(?:name\s+of\s+patient):\s*([A-Za-z\s]+)',
                r'(?i)(?:insured\s+name):\s*([A-Za-z\s]+)'
            ],
            'date': [
                r'(?i)(?:date|filed|submitted|service|treatment):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)(?:date\s+of\s+service):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)(?:claim\s+date):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'diagnosis': [
                r'(?i)(?:diagnosis|condition|injury|illness):\s*([A-Za-z\s]+)',
                r'(?i)(?:primary\s+diagnosis):\s*([A-Za-z\s]+)',
                r'(?i)(?:medical\s+condition):\s*([A-Za-z\s]+)'
            ],
            'amount': [
                r'(?i)(?:amount|claim|total|billed):\s*[\$€£]?([\d,]+\.?\d*)',
                r'(?i)(?:claim\s+amount):\s*[\$€£]?([\d,]+\.?\d*)',
                r'(?i)(?:total\s+amount):\s*[\$€£]?([\d,]+\.?\d*)'
            ],
            'policy_number': [
                r'(?i)(?:policy|claim|reference|member|id):\s*([A-Za-z0-9-]+)',
                r'(?i)(?:policy\s+number):\s*([A-Za-z0-9-]+)',
                r'(?i)(?:member\s+id):\s*([A-Za-z0-9-]+)'
            ],
            'provider': [
                r'(?i)(?:provider|doctor|physician|hospital):\s*([A-Za-z\s]+)',
                r'(?i)(?:attending\s+physician):\s*([A-Za-z\s]+)',
                r'(?i)(?:service\s+provider):\s*([A-Za-z\s]+)'
            ],
            'service_date': [
                r'(?i)(?:service|treatment|visit)\s+date:\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)(?:date\s+of\s+service):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'procedure': [
                r'(?i)(?:procedure|treatment|service):\s*([A-Za-z\s]+)',
                r'(?i)(?:medical\s+procedure):\s*([A-Za-z\s]+)',
                r'(?i)(?:surgical\s+procedure):\s*([A-Za-z\s]+)'
            ]
        }
    
    def extract_fields(self, text: str) -> Dict[str, str]:
        """
        Extract structured fields from text using NLP and regex
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Dictionary of extracted fields
        """
        extracted_fields = {}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract fields using patterns
        for field_name, patterns in self.field_patterns.items():
            value = self._extract_field_with_patterns(text, patterns)
            if value:
                extracted_fields[field_name] = value
        
        # Use NLP for additional extraction
        nlp_extracted = self._extract_with_nlp(doc)
        for field, value in nlp_extracted.items():
            if value and field not in extracted_fields:
                extracted_fields[field] = value
        
        # Clean and validate extracted fields
        cleaned_fields = self._clean_extracted_fields(extracted_fields)
        
        return cleaned_fields
    
    def _extract_field_with_patterns(self, text: str, patterns: List[str]) -> Optional[str]:
        """
        Extract field using regex patterns
        
        Args:
            text: Text to search
            patterns: List of regex patterns
            
        Returns:
            Extracted value or None
        """
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the first match and clean it
                value = matches[0].strip()
                if value and len(value) > 1:  # Ensure it's not just a single character
                    return value
        return None
    
    def _extract_with_nlp(self, doc) -> Dict[str, str]:
        """
        Extract fields using NLP techniques
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary of extracted fields
        """
        extracted = {}
        
        # Extract names (PERSON entities)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if names:
            extracted['name'] = names[0]
        
        # Extract dates (DATE entities)
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        if dates:
            extracted['date'] = dates[0]
        
        # Extract money amounts (MONEY entities)
        money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
        if money:
            # Clean money amount
            amount = re.sub(r'[^\d.,]', '', money[0])
            if amount:
                extracted['amount'] = amount
        
        # Extract organizations (ORG entities) - could be providers
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if orgs:
            extracted['provider'] = orgs[0]
        
        return extracted
    
    def _clean_extracted_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """
        Clean and validate extracted fields
        
        Args:
            fields: Raw extracted fields
            
        Returns:
            Cleaned fields
        """
        cleaned = {}
        
        for field, value in fields.items():
            if not value:
                continue
            
            # Clean the value
            cleaned_value = value.strip()
            
            # Field-specific cleaning
            if field == 'name':
                # Remove extra spaces and capitalize properly
                cleaned_value = ' '.join(cleaned_value.split())
                cleaned_value = cleaned_value.title()
            
            elif field == 'amount':
                # Clean amount - remove currency symbols and extra characters
                cleaned_value = re.sub(r'[^\d.,]', '', cleaned_value)
                # Convert to float if possible
                try:
                    float_val = float(cleaned_value.replace(',', ''))
                    cleaned_value = f"{float_val:.2f}"
                except ValueError:
                    pass
            
            elif field == 'date':
                # Standardize date format
                try:
                    # Try to parse and standardize date
                    parsed_date = self._parse_date(cleaned_value)
                    if parsed_date:
                        cleaned_value = parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
            
            elif field == 'diagnosis':
                # Clean diagnosis text
                cleaned_value = cleaned_value.strip()
                # Remove common prefixes
                cleaned_value = re.sub(r'^(?:diagnosis|condition|injury):\s*', '', cleaned_value, flags=re.IGNORECASE)
            
            if cleaned_value and len(cleaned_value) > 1:
                cleaned[field] = cleaned_value
        
        return cleaned
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string to datetime object
        
        Args:
            date_str: Date string
            
        Returns:
            datetime object or None
        """
        # Common date formats
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%y', '%m-%d-%y', '%d/%m/%y', '%d-%m-%y',
            '%Y-%m-%d', '%Y/%m/%d'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def validate_extraction(self, fields: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate extracted fields
        
        Args:
            fields: Extracted fields
            
        Returns:
            Dictionary of validation results
        """
        validation = {}
        
        # Validate name
        if 'name' in fields:
            validation['name'] = len(fields['name'].split()) >= 2  # At least first and last name
        
        # Validate date
        if 'date' in fields:
            validation['date'] = bool(self._parse_date(fields['date']))
        
        # Validate amount
        if 'amount' in fields:
            try:
                float(fields['amount'].replace(',', ''))
                validation['amount'] = True
            except ValueError:
                validation['amount'] = False
        
        # Validate diagnosis
        if 'diagnosis' in fields:
            validation['diagnosis'] = len(fields['diagnosis']) > 3
        
        return validation
    
    def get_extraction_confidence(self, fields: Dict[str, str], validation: Dict[str, bool]) -> float:
        """
        Calculate confidence score for extraction
        
        Args:
            fields: Extracted fields
            validation: Validation results
            
        Returns:
            Confidence score (0-1)
        """
        if not fields:
            return 0.0
        
        # Count valid fields
        valid_count = sum(validation.values())
        total_count = len(fields)
        
        # Base confidence on validation
        confidence = valid_count / total_count if total_count > 0 else 0.0
        
        # Bonus for having key fields
        key_fields = ['name', 'date', 'amount', 'diagnosis']
        key_field_bonus = sum(1 for field in key_fields if field in fields and validation.get(field, False))
        confidence += (key_field_bonus / len(key_fields)) * 0.2
        
        return min(confidence, 1.0) 