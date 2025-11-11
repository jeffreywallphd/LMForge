#!/usr/bin/env python3
"""
Simple Ordered International Finance PDF Extractor

Expected Structure (in exact order):
- Preface
- Chapter 1: Introductory Finance Issues: Current Patterns, Past History, and International Institutions
  - 1.1 The International Economy and International Economics
  - 1.2 GDP Unemployment, Inflation, and Government Budget Balances
  - 1.3 Exchange Rate Regimes, Trade Balances, and Investment Positions
  - 1.4 Business Cycles: Economic Ups and Downs
  - 1.5 International Macroeconomic Institutions: The IMF and the World Bank
- Chapter 2: National Income and the Balance of Payments Accounts
  - 2.1 National Income and Product Accounts
  - 2.2 National Income or Product Identity
  - 2.3 U.S. National Income Statistics (2007–2008)
  - 2.4 Balance of Payments Accounts: Definitions
  - 2.5 Recording Transactions on the Balance of Payments
  - 2.6 U.S. Balance of Payments Statistics (2008)
  - 2.7 The Twin-Deficit Identity
  - 2.8 International Investment Position
- Chapter 3: The Whole Truth about Trade Imbalances
  - 3.1 Overview of Trade Imbalances
  - 3.2 Trade Imbalances and Jobs
  - 3.3 The National Welfare Effects of Trade Imbalances
  - 3.4 Some Further Complications
  - 3.5 How to Evaluate Trade Imbalances
- Chapter 4: Foreign Exchange Markets and Rates of Return
  - 4.1 The Forex: Participants and Objectives
  - 4.2 Exchange Rate: Definitions
  - 4.3 Calculating Rate of Returns on International Investments
  - 4.4 Interpretation of the Rate of Return Formula
  - 4.5 Applying the Rate of Return Formulas
- Chapter 5: Interest Rate Parity
  - 5.1 Overview of Interest Rate Parity
  - 5.2 Comparative Statics in the IRP Theory
  - 5.3 Forex Equilibrium with the Rate of Return Diagram
  - 5.4 Exchange Rate Equilibrium Stories with the RoR Diagram
  - 5.5 Exchange Rate Effects of Changes in U.S. Interest Rates Using the RoR Diagram
  - 5.6 Exchange Rate Effects of Changes in Foreign Interest Rates Using the RoR Diagram
  - 5.7 Exchange Rate Effects of Changes in the Expected Exchange Rate Using the RoR Diagram
- Chapter 6: Purchasing Power Parity
  - 6.1 Overview of Purchasing Power Parity (PPP)
  - 6.2 The Consumer Price Index (CPI) and PPP
  - 6.3 PPP as a Theory of Exchange Rate Determination
  - 6.4 Problems and Extensions of PPP
  - 6.5 PPP in the Long Run
  - 6.6 Overvaluation and Undervaluation
  - 6.7 PPP and Cross-Country Comparisons
- Chapter 7: Interest Rate Determination
  - 7.1 Overview of Interest Rate Determination
  - 7.2 Some Preliminaries
  - 7.3 What Is Money?
  - 7.4 Money Supply Measures
  - 7.5 Controlling the Money Supply
  - 7.6 Money Demand
  - 7.7 Money Functions and Equilibrium
  - 7.8 Money Market Equilibrium Stories
  - 7.9 Effects of a Money Supply Increase
  - 7.10 Effect of a Price Level Increase (Inflation) on Interest Rates
  - 7.11 Effect of a Real GDP Increase (Economic Growth) on Interest Rates
  - 7.12 Integrating the Money Market and the Foreign Exchange Markets
  - 7.13 Comparative Statics in the Combined Money-Forex Model
  - 7.14 Money Supply and Long-Run Prices
- Chapter 8: National Output Determination
  - 8.1 Overview of National Output Determination
  - 8.2 Aggregate Demand for Goods and Services
  - 8.3 Consumption Demand
  - 8.4 Investment Demand
  - 8.5 Government Demand
  - 8.6 Export and Import Demand
  - 8.7 The Aggregate Demand Function
  - 8.8 The Keynesian Cross Diagram
  - 8.9 Goods and Services Market Equilibrium Stories
  - 8.10 Effect of an Increase in Government Demand on Real GNP
  - 8.11 Effect of an Increase in the U.S. Dollar Value on Real GNP
  - 8.12 The J-Curve Effect
- Chapter 9: The AA-DD Model
  - 9.1 Overview of the AA-DD Model
  - 9.2 Derivation of the DD Curve
  - 9.3 Shifting the DD Curve
  - 9.4 Derivation of the AA Curve
  - 9.5 Shifting the AA Curve
  - 9.6 Superequilibrium: Combining DD and AA
  - 9.7 Adjustment to the Superequilibrium
  - 9.8 AA-DD and the Current Account Balance
- Chapter 10: Policy Effects with Floating Exchange Rates
  - 10.1 Overview of Policy with Floating Exchange Rates
  - 10.2 Monetary Policy with Floating Exchange Rates
  - 10.3 Fiscal Policy with Floating Exchange Rates
  - 10.4 Expansionary Monetary Policy with Floating Exchange Rates in the Long Run
  - 10.5 Foreign Exchange Interventions with Floating Exchange Rates
- Chapter 11: Fixed Exchange Rates
  - 11.1 Overview of Fixed Exchange Rates
  - 11.2 Fixed Exchange Rate Systems
  - 11.3 Interest Rate Parity with Fixed Exchange Rates
  - 11.4 Central Bank Intervention with Fixed Exchange Rates
  - 11.5 Balance of Payments Deficits and Surpluses
  - 11.6 Black Markets
- Chapter 12: Policy Effects with Fixed Exchange Rates
  - 12.1 Overview of Policy with Fixed Exchange Rates
  - 12.2 Monetary Policy with Fixed Exchange Rates
  - 12.3 Fiscal Policy with Fixed Exchange Rates
  - 12.4 Exchange Rate Policy with Fixed Exchange Rates
  - 12.5 Reserve Country Monetary Policy under Fixed Exchange Rates
  - 12.6 Currency Crises and Capital Flight
  - 12.7 Case Study: The Breakup of the Bretton Woods System, 1973
- Chapter 13: Fixed versus Floating Exchange Rates
  - 13.1 Overview of Fixed versus Floating Exchange Rates
  - 13.2 Exchange Rate Volatility and Risk
  - 13.3 Inflationary Consequences of Exchange Rate Systems
  - 13.4 Monetary Autonomy and Exchange Rate Systems
  - 13.5 Which Is Better: Fixed or Floating Exchange Rates?

Total: 75 sections (Preface + 10 chapters with detailed subsections that actually exist in PDF)
Output: Ordered metadata with complete content for each section
"""

import json
from pathlib import Path
import re

try:
    import pdfplumber
except ImportError:
    print("Run: pip install pdfplumber")
    import PyPDF2

class SimpleOrderedExtractor:
    def __init__(self):
        # Complete outline in exact order with all subsections (only chapters 1-10 exist in PDF)
        self.outline_order = [
            "Preface",
            "Chapter 1: Introductory Finance Issues: Current Patterns, Past History, and International Institutions",
            "1.1 The International Economy and International Economics",
            "1.2 GDP Unemployment, Inflation, and Government Budget Balances",
            "1.3 Exchange Rate Regimes, Trade Balances, and Investment Positions",
            "1.4 Business Cycles: Economic Ups and Downs",
            "1.5 International Macroeconomic Institutions: The IMF and the World Bank",
            "Chapter 2: National Income and the Balance of Payments Accounts",
            "2.1 National Income and Product Accounts",
            "2.2 National Income or Product Identity",
            "2.3 U.S. National Income Statistics (2007–2008)",
            "2.4 Balance of Payments Accounts: Definitions",
            "2.5 Recording Transactions on the Balance of Payments",
            "2.6 U.S. Balance of Payments Statistics (2008)",
            "2.7 The Twin-Deficit Identity",
            "2.8 International Investment Position",
            "Chapter 3: The Whole Truth about Trade Imbalances",
            "3.1 Overview of Trade Imbalances",
            "3.2 Trade Imbalances and Jobs",
            "3.3 The National Welfare Effects of Trade Imbalances",
            "3.4 Some Further Complications",
            "3.5 How to Evaluate Trade Imbalances",
            "Chapter 4: Foreign Exchange Markets and Rates of Return",
            "4.1 The Forex: Participants and Objectives",
            "4.2 Exchange Rate: Definitions",
            "4.3 Calculating Rate of Returns on International Investments",
            "4.4 Interpretation of the Rate of Return Formula",
            "4.5 Applying the Rate of Return Formulas",
            "Chapter 5: Interest Rate Parity",
            "5.1 Overview of Interest Rate Parity",
            "5.2 Comparative Statics in the IRP Theory",
            "5.3 Forex Equilibrium with the Rate of Return Diagram",
            "5.4 Exchange Rate Equilibrium Stories with the RoR Diagram",
            "5.5 Exchange Rate Effects of Changes in U.S. Interest Rates Using the RoR Diagram",
            "5.6 Exchange Rate Effects of Changes in Foreign Interest Rates Using the RoR Diagram",
            "5.7 Exchange Rate Effects of Changes in the Expected Exchange Rate Using the RoR Diagram",
            "Chapter 6: Purchasing Power Parity",
            "6.1 Overview of Purchasing Power Parity (PPP)",
            "6.2 The Consumer Price Index (CPI) and PPP",
            "6.3 PPP as a Theory of Exchange Rate Determination",
            "6.4 Problems and Extensions of PPP",
            "6.5 PPP in the Long Run",
            "6.6 Overvaluation and Undervaluation",
            "6.7 PPP and Cross-Country Comparisons",
            "Chapter 7: Interest Rate Determination",
            "7.1 Overview of Interest Rate Determination",
            "7.2 Some Preliminaries",
            "7.3 What Is Money?",
            "7.4 Money Supply Measures",
            "7.5 Controlling the Money Supply",
            "7.6 Money Demand",
            "7.7 Money Functions and Equilibrium",
            "7.8 Money Market Equilibrium Stories",
            "7.9 Effects of a Money Supply Increase",
            "Chapter 8: National Output Determination",
            "8.1 Overview of National Output Determination",
            "8.2 Aggregate Demand for Goods and Services",
            "8.3 Consumption Demand",
            "8.4 Investment Demand",
            "8.5 Government Demand",
            "8.6 Export and Import Demand",
            "8.7 The Aggregate Demand Function",
            "8.8 The Keynesian Cross Diagram",
            "8.9 Goods and Services Market Equilibrium Stories",
            "Chapter 9: The AA-DD Model",
            "9.1 Overview of the AA-DD Model",
            "9.2 Derivation of the DD Curve",
            "9.3 Shifting the DD Curve",
            "Chapter 10: Policy Effects with Floating Exchange Rates",
            "10.1 Overview of Policy with Floating Exchange Rates",
            "10.2 Monetary Policy with Floating Exchange Rates",
            "10.3 Fiscal Policy with Floating Exchange Rates",
            "10.4 Expansionary Monetary Policy with Floating Exchange Rates in the Long Run"
        ]
    
    def extract_pdf(self, pdf_path):
        """Extract PDF content"""
        content = ""
        try:
            # Try pdfplumber first
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n\n"
        except ImportError:
            # Fallback to PyPDF2
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n\n"
            except Exception as e:
                print(f"Error reading PDF with PyPDF2: {e}")
                return ""
        except Exception as e:
            print(f"Error reading PDF with pdfplumber: {e}")
            return ""
        return content
    
    def find_sections(self, content):
        """Find sections in content and extract complete text between them"""
        lines = content.split('\n')
        sections = {}
        
        # Find ALL section boundaries (not just from our outline)
        all_boundaries = []
        
        # First, find all our expected sections
        for i, line in enumerate(lines):
            line_clean = line.strip()
            for section_name in self.outline_order:
                if self.matches_section(line_clean, section_name):
                    all_boundaries.append({
                        'line': i,
                        'name': section_name,
                        'text': line_clean,
                        'is_expected': True
                    })
                    break
        
        # Also find potential section headers that might interrupt our content
        # (like other chapters, appendices, etc.)
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Skip if already found as expected section
            if any(b['line'] == i for b in all_boundaries):
                continue
                
            # Look for other section-like patterns
            if self.looks_like_section_header(line_clean):
                all_boundaries.append({
                    'line': i,
                    'name': f"OTHER_SECTION_{i}",
                    'text': line_clean,
                    'is_expected': False
                })
        
        # Sort boundaries by line number
        all_boundaries.sort(key=lambda x: x['line'])
        
        # Extract content between boundaries
        for i, boundary in enumerate(all_boundaries):
            if not boundary['is_expected']:
                continue  # Skip non-expected sections
                
            section_name = boundary['name']
            start_line = boundary['line'] + 1  # Start after the section header
            
            # Find the next boundary (expected or unexpected)
            end_line = len(lines)  # Default to end of document
            for j in range(i + 1, len(all_boundaries)):
                end_line = all_boundaries[j]['line']
                break
            
            # Extract ALL content between start and end
            section_lines = []
            for line_idx in range(start_line, end_line):
                if line_idx < len(lines):
                    line = lines[line_idx].rstrip()  # Remove trailing whitespace
                    section_lines.append(line)
            
            # Join lines and clean up
            section_content = '\n'.join(section_lines)
            
            # Remove excessive blank lines but preserve paragraph structure
            section_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', section_content)
            section_content = section_content.strip()
            
            # Parse section content into structured parts
            parsed_content = self.parse_section_content(section_content)
            
            sections[section_name] = {
                'title': section_name,
                'learning_objectives': parsed_content['learning_objectives'],
                'content': parsed_content['main_content'],
                'key_takeaways': parsed_content['key_takeaways'],
                'exercises': parsed_content['exercises'],
                'word_count': len(section_content.split()) if section_content else 0,
                'line_start': start_line,
                'line_end': end_line,
                'status': 'found' if section_content else 'empty'
            }
        
        return sections
    
    def clean_item(self, text):
        """Clean individual list items by removing Saylor URLs and trailing punctuation/newlines"""
        # Remove Saylor URL patterns
        text = re.sub(r'Saylor URL:.*?(?:\n\d+\n|\n|$)', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove standalone numbers that look like page numbers
        text = re.sub(r'\n\s*\d+\s*\n', ' ', text)
        
        # Remove trailing punctuation marks, newlines, and whitespace
        text = re.sub(r'[\.,;:!?\s\n]+$', '', text)
        
        # Remove leading bullet points if any remain
        text = re.sub(r'^[•\-\*]\s*', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def parse_section_content(self, content):
        """Parse section content into learning objectives, main content, key takeaways, and exercises"""
        if not content:
            return {
                'learning_objectives': [],
                'main_content': content,
                'key_takeaways': [],
                'exercises': []
            }
        
        learning_objectives = []
        main_content = content
        key_takeaways = []
        exercises = []
        
        # Remove Saylor URLs from entire content first
        content = re.sub(r'Saylor URL:.*?(?:\n\d+\n|\n|$)', '', content, flags=re.IGNORECASE | re.MULTILINE)
        main_content = content
        
        # Find Learning Objectives (case insensitive)
        lo_pattern = r'LEARNING OBJECTIVE[S]?\s*\n(.+?)(?=\n[A-Z][a-z]+|\n\n[A-Z]|\Z)'
        lo_match = re.search(lo_pattern, content, re.IGNORECASE | re.DOTALL)
        if lo_match:
            lo_text = lo_match.group(0).strip()
            # Remove from main content
            main_content = content[:lo_match.start()] + content[lo_match.end():]
            
            # Parse learning objectives into list items (numbered 1. 2. 3. etc.)
            lo_content = lo_match.group(1).strip()
            # Split by numbered items (1. 2. 3. etc.)
            lo_items = re.split(r'\n\s*\d+\.\s+', lo_content)
            # First item might not have a number, so check if the content starts with a number
            if re.match(r'^\d+\.\s+', lo_content):
                # Content starts with a number, so split normally
                lo_items = re.split(r'\d+\.\s+', lo_content)
                learning_objectives = [self.clean_item(item) for item in lo_items if item.strip()]
            else:
                # First item doesn't have number, so it's already in the list
                learning_objectives = [self.clean_item(item) for item in lo_items if item.strip()]
        
        # Find Key Takeaways (case insensitive)
        kt_pattern = r'KEY TAKEAWAY[S]?\s*\n(.+?)(?=EXERCISE[S]?|\Z)'
        kt_match = re.search(kt_pattern, main_content, re.IGNORECASE | re.DOTALL)
        if kt_match:
            kt_text = kt_match.group(0).strip()
            # Remove from main content
            main_content = main_content[:kt_match.start()] + main_content[kt_match.end():]
            
            # Parse key takeaways into list items (separated by bullet points or newlines)
            kt_content = kt_match.group(1).strip()
            # Split by bullet points (• or -) or double newlines
            kt_items = re.split(r'\n\s*[•\-]\s*|\n\n+', kt_content)
            key_takeaways = [self.clean_item(item) for item in kt_items if item.strip() and len(item.strip()) > 10]
        
        # Find Exercises (case insensitive)
        ex_pattern = r'EXERCISE[S]?\s*\n(.+?)(?=\Z)'
        ex_match = re.search(ex_pattern, main_content, re.IGNORECASE | re.DOTALL)
        if ex_match:
            ex_text = ex_match.group(0).strip()
            # Remove from main content
            main_content = main_content[:ex_match.start()].strip()
            
            # Parse exercises into list items (numbered 1. 2. 3. or lettered a. b. c.)
            ex_content = ex_match.group(1).strip()
            # Split by numbered items at the start of lines
            ex_items = re.split(r'\n\s*\d+\.\s+', ex_content)
            # First item might contain the intro text before first numbered item
            if re.search(r'^\d+\.\s+', ex_content):
                ex_items = re.split(r'\d+\.\s+', ex_content)
            exercises = [self.clean_item(item) for item in ex_items if item.strip() and len(item.strip()) > 10]
        
        # Clean up main content - remove Saylor URLs
        main_content = re.sub(r'Saylor URL:.*?(?:\n\d+\n|\n|$)', '', main_content, flags=re.IGNORECASE | re.MULTILINE)
        main_content = main_content.strip()
        
        return {
            'learning_objectives': learning_objectives,
            'main_content': main_content,
            'key_takeaways': key_takeaways,
            'exercises': exercises
        }
    
    def looks_like_section_header(self, line):
        """Check if a line looks like a section header (to find boundaries)"""
        line = line.strip()
        
        # Skip very long lines (likely paragraphs)
        if len(line) > 100:
            return False
            
        # Skip empty lines
        if not line:
            return False
        
        # Common section header patterns
        patterns = [
            r'^Chapter\s+\d+',
            r'^CHAPTER\s+\d+',
            r'^\d+\.\d+\s+\w+',  # Like "1.1 Something"
            r'^Part\s+[IVX]+',
            r'^PART\s+[IVX]+',
            r'^Appendix',
            r'^APPENDIX',
            r'^Bibliography',
            r'^References',
            r'^Index',
            r'^Glossary',
            r'^Summary',
            r'^Conclusion',
            r'^Introduction',
            r'^Preface',
            r'^PREFACE',
            r'^Abstract',
            r'^ABSTRACT'
        ]
        
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def matches_section(self, line, section_name):
        """Check if line matches section name with improved matching"""
        line = line.strip()
        
        # Direct match
        if line == section_name:
            return True
        
        # Case insensitive match
        if line.lower() == section_name.lower():
            return True
        
        # For Preface
        if section_name == "Preface" and line.lower() == "preface":
            return True
        
        # Chapter matching - must be at start of line and match full title
        if section_name.startswith("Chapter"):
            # Extract chapter number
            chapter_match = re.match(r'Chapter (\d+):', section_name)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                # Check if line starts with this chapter and has similar content
                if re.match(rf'^Chapter {chapter_num}:', line, re.IGNORECASE):
                    # Check if major keywords from title appear in line
                    title_words = section_name.split(':')[1].strip().lower().split()[:3]
                    line_lower = line.lower()
                    if any(word in line_lower for word in title_words if len(word) > 3):
                        return True
        
        # Section number matching (e.g., "3.1", "5.2") - must be at start of line
        section_match = re.match(r'^(\d+\.\d+)\s+(.+)', section_name)
        if section_match:
            section_num = section_match.group(1)
            section_title = section_match.group(2).lower()
            
            # Line must start with the section number
            line_match = re.match(rf'^{re.escape(section_num)}\s+(.+)', line)
            if line_match:
                line_title = line_match.group(1).lower()
                # Check if titles are similar (at least first 3 significant words match)
                title_words = [w for w in section_title.split() if len(w) > 3][:3]
                if title_words:
                    matches = sum(1 for word in title_words if word in line_title)
                    if matches >= min(2, len(title_words)):
                        return True
                else:
                    # Short titles, just check if they're similar
                    if section_title[:15] in line_title or line_title[:15] in section_title:
                        return True
        
        return False
    
    def create_ordered_output(self, sections, pdf_path):
        """Create ordered output with nested structure - subsections inside chapters"""
        hierarchical_sections = []
        current_chapter = None
        
        # Process in outline order
        for section_name in self.outline_order:
            if section_name in sections:
                section_data = sections[section_name]
            else:
                # Missing section
                section_data = {
                    'title': section_name,
                    'content': '',
                    'word_count': 0,
                    'status': 'not_found'
                }
            
            # Check if this is a chapter or a subsection
            if section_name.startswith("Chapter") or section_name == "Preface":
                # This is a main chapter/preface - add it to the list
                chapter_entry = {
                    'title': section_data['title'],
                    'learning_objectives': section_data.get('learning_objectives', ''),
                    'content': section_data.get('content', section_data.get('main_content', '')),
                    'key_takeaways': section_data.get('key_takeaways', ''),
                    'exercises': section_data.get('exercises', ''),
                    'word_count': section_data['word_count'],
                    'status': section_data.get('status', 'found'),
                    'subsections': []
                }
                if 'line_start' in section_data:
                    chapter_entry['line_start'] = section_data['line_start']
                    chapter_entry['line_end'] = section_data['line_end']
                
                hierarchical_sections.append(chapter_entry)
                current_chapter = chapter_entry
            else:
                # This is a subsection - add it under the current chapter
                if current_chapter is not None:
                    subsection_entry = {
                        'title': section_data['title'],
                        'learning_objectives': section_data.get('learning_objectives', ''),
                        'content': section_data.get('content', section_data.get('main_content', '')),
                        'key_takeaways': section_data.get('key_takeaways', ''),
                        'exercises': section_data.get('exercises', ''),
                        'word_count': section_data['word_count'],
                        'status': section_data.get('status', 'found')
                    }
                    if 'line_start' in section_data:
                        subsection_entry['line_start'] = section_data['line_start']
                        subsection_entry['line_end'] = section_data['line_end']
                    
                    current_chapter['subsections'].append(subsection_entry)
        
        # Create metadata
        file_path = Path(pdf_path)
        stat = file_path.stat()
        
        # Count all sections (chapters + subsections)
        total_subsections = sum(len(ch['subsections']) for ch in hierarchical_sections)
        all_sections = []
        for ch in hierarchical_sections:
            all_sections.append(ch)
            all_sections.extend(ch['subsections'])
        
        metadata = {
            'filename': file_path.name,
            'total_chapters': len(hierarchical_sections),
            'total_subsections': total_subsections,
            'total_sections_expected': len(self.outline_order),
            'sections_found': len([s for s in all_sections if s['status'] == 'found']),
            'sections_empty': len([s for s in all_sections if s['status'] == 'empty']),
            'sections_missing': len([s for s in all_sections if s['status'] == 'not_found']),
            'total_words': sum(s['word_count'] for s in all_sections),
            'file_size': stat.st_size,
            'structure': 'hierarchical_with_nested_subsections'
        }
        
        return {
            'metadata': metadata,
            'chapters': hierarchical_sections
        }
    
    def save_results(self, results, output_dir="../jsons"):
        """Save ordered results to jsons directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename_base = Path(results['metadata']['filename']).stem
        
        # Save ordered sections
        sections_file = output_path / f"{filename_base}_ordered_sections.json"
        with open(sections_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved ordered sections: {sections_file}")
        return str(sections_file)

def main():
    """ Read International Finance PDF and split based on expected outline order """
    print("📚 International Finance PDF Extractor")
    print("=" * 50)
    print("🎯 Extracting content in exact outline order")
    print()
    
    # Target specific PDF file in docs subdirectory
    pdf_path = Path("docs/International Finance - Theory and Policy.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        print("📁 Please ensure 'International Finance - Theory and Policy.pdf' is in the docs/ subdirectory")
        return
    
    extractor = SimpleOrderedExtractor()
    
    # Process the specific PDF file
    pdf_file = pdf_path
    print(f"🔍 Processing: {pdf_file.name}")
    
    # Task 1: Read International Finance PDF and split based on expected outline order
    print("📖 Step 1: Reading PDF and extracting content...")
    content = extractor.extract_pdf(pdf_file)
    if not content:
        print("❌ Failed to extract PDF content")
        return
    
    print("🔍 Step 2: Finding sections based on outline order...")
    sections = extractor.find_sections(content)
    
    # Task 2: Create a json structure ordered outline
    print("📋 Step 3: Creating ordered JSON structure...")
    results = extractor.create_ordered_output(sections, pdf_file)
    
    # Task 3: Save the JSON structure to disk
    print("💾 Step 4: Saving JSON structure to disk...")
    saved_file = extractor.save_results(results)
    
    # Print summary
    metadata = results['metadata']
    print(f"\n📊 Extraction Summary:")
    print(f"  Expected Sections: {metadata['total_sections_expected']}")
    print(f"  Found Sections: {metadata['sections_found']}")
    print(f"  Empty Sections: {metadata['sections_empty']}")
    print(f"  Missing Sections: {metadata['sections_missing']}")
    print(f"  Total Words: {metadata['total_words']:,}")
    print(f"  Output File: {saved_file}")
    
    # Show first few chapters with subsections
    print(f"\n📖 Chapter Structure:")
    for i, chapter in enumerate(results['chapters'][:5]):
        status_icon = "✅" if chapter['status'] == 'found' else "❌" if chapter['status'] == 'not_found' else "📄"
        print(f"  {i+1}. {status_icon} {chapter['title'][:60]}{'...' if len(chapter['title']) > 60 else ''}")
        print(f"     Words: {chapter['word_count']:,} | Subsections: {len(chapter['subsections'])}")
        
        # Show first few subsections
        for j, subsection in enumerate(chapter['subsections'][:3]):
            sub_icon = "✅" if subsection['status'] == 'found' else "❌" if subsection['status'] == 'not_found' else "📄"
            print(f"       {sub_icon} {subsection['title'][:55]}{'...' if len(subsection['title']) > 55 else ''}")
            print(f"          Words: {subsection['word_count']:,}")
        
        if len(chapter['subsections']) > 3:
            print(f"       ... and {len(chapter['subsections']) - 3} more subsections")
    
    if len(results['chapters']) > 5:
        print(f"  ... and {len(results['chapters']) - 5} more chapters")
    
    print(f"\n🎉 Processing complete!")
    print(f"📁 JSON file saved: {saved_file}")

if __name__ == "__main__":
    main()