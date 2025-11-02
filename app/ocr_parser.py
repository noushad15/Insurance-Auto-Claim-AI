import pytesseract
import pdfplumber
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRParser:
    def __init__(self):
        """Initialize OCR parser with Tesseract configuration"""
        # Configure Tesseract for better accuracy
        self.custom_config = r'--oem 3 --psm 6'
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PDF using pdfplumber and OCR for images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            extracted_data = {
                'text': '',
                'tables': [],
                'images_text': '',
                'metadata': {}
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                extracted_data['metadata'] = {
                    'pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', '')
                }
                
                # Process each page
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Processing page {page_num + 1}")
                    
                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        extracted_data['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        if table:
                            table_text = self._table_to_text(table)
                            extracted_data['tables'].append({
                                'page': page_num + 1,
                                'table_num': table_num + 1,
                                'data': table_text
                            })
                    
                    # Extract text from images using OCR
                    if page.images:
                        for img in page.images:
                            try:
                                # Convert image to PIL Image
                                img_data = img['stream'].get_data()
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # OCR the image
                                img_text = pytesseract.image_to_string(pil_image, config=self.custom_config)
                                if img_text.strip():
                                    extracted_data['images_text'] += f"\n--- Image on page {page_num + 1} ---\n{img_text}"
                            except Exception as e:
                                logger.warning(f"Failed to OCR image on page {page_num + 1}: {e}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=self.custom_config)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """
        Convert table data to readable text
        
        Args:
            table: Table data as list of lists
            
        Returns:
            Formatted table text
        """
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            # Filter out None values and join with tabs
            row_text = "\t".join([str(cell) if cell else "" for cell in row])
            text_lines.append(row_text)
        
        return "\n".join(text_lines)
    
    def extract_structured_data(self, text: str) -> Dict[str, str]:
        """
        Extract structured data from text using regex patterns
        
        Args:
            text: Raw extracted text
            
        Returns:
            Dictionary of extracted fields
        """
        structured_data = {}
        
        # Common patterns for claim documents
        patterns = {
            'name': r'(?i)(?:name|patient|insured):\s*([A-Za-z\s]+)',
            'date': r'(?i)(?:date|filed|submitted):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'diagnosis': r'(?i)(?:diagnosis|condition|injury):\s*([A-Za-z\s]+)',
            'amount': r'(?i)(?:amount|claim|total):\s*[\$€£]?([\d,]+\.?\d*)',
            'policy_number': r'(?i)(?:policy|claim|reference):\s*([A-Za-z0-9-]+)',
            'provider': r'(?i)(?:provider|doctor|physician):\s*([A-Za-z\s]+)',
            'service_date': r'(?i)(?:service|treatment|visit)\s+date:\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        }
        
        for field, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Take the first match and clean it
                value = matches[0].strip()
                if value:
                    structured_data[field] = value
        
        return structured_data 