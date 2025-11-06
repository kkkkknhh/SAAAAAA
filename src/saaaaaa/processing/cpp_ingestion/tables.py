"""
Table extraction and budget handling.

Extracts tables, KPIs, and budget data from documents with production logic.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def _safe_strip(value: Any) -> str:
    """
    Safely convert a value to a stripped string.
    
    Handles None values and non-string types without raising errors.
    
    Args:
        value: Any value from a table cell
        
    Returns:
        Stripped string representation, or empty string for None
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


class TableExtractor:
    """Production-ready table and budget data extractor."""
    
    # Keywords for table classification
    KPI_KEYWORDS = [
        "indicador", "meta", "línea base", "baseline", "objetivo",
        "resultado esperado", "medición", "seguimiento", "tasa",
        "cobertura", "porcentaje"
    ]
    
    BUDGET_KEYWORDS = [
        "presupuesto", "fuente", "uso", "monto", "budget", "recursos",
        "financiación", "inversión", "costo", "valor", "millones",
        "cop", "$", "pesos"
    ]
    
    def extract(self, raw_objects: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tables and budget data from parsed document.
        
        Args:
            raw_objects: Raw parsed objects from DocumentParser
            
        Returns:
            Subgraph with classified tables, extracted KPIs, and budget data
        """
        tables_subgraph = {
            "tables": [],
            "kpis": [],
            "budgets": [],
        }
        
        # Extract tables from pages
        for page in raw_objects.get("pages", []):
            tables = self._extract_tables_from_page(page)
            tables_subgraph["tables"].extend(tables)
            
            # Classify and extract structured data from each table
            for table in tables:
                if self._is_kpi_table(table):
                    kpis = self._extract_kpis(table)
                    tables_subgraph["kpis"].extend(kpis)
                elif self._is_budget_table(table):
                    budgets = self._extract_budgets(table)
                    tables_subgraph["budgets"].extend(budgets)
        
        # Also extract from document-level tables if available
        doc_tables = raw_objects.get("tables", [])
        for table in doc_tables:
            if table not in tables_subgraph["tables"]:
                tables_subgraph["tables"].append(table)
                
                if self._is_kpi_table(table):
                    kpis = self._extract_kpis(table)
                    tables_subgraph["kpis"].extend(kpis)
                elif self._is_budget_table(table):
                    budgets = self._extract_budgets(table)
                    tables_subgraph["budgets"].extend(budgets)
        
        return tables_subgraph
    
    def _extract_tables_from_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tables from a page.
        
        Production implementation that retrieves tables already parsed
        by DocumentParser (pdfplumber, python-docx, or BeautifulSoup).
        """
        return page.get("tables", [])
    
    def _is_kpi_table(self, table: Dict[str, Any]) -> bool:
        """
        Check if table contains KPI/indicator data.
        
        Looks for indicator-related keywords in headers and content.
        """
        # Convert table to searchable text
        text = self._table_to_text(table).lower()
        
        # Check for KPI keywords
        return any(kw in text for kw in self.KPI_KEYWORDS)
    
    def _is_budget_table(self, table: Dict[str, Any]) -> bool:
        """
        Check if table contains budget/financial data.
        
        Looks for budget-related keywords and numeric patterns.
        """
        text = self._table_to_text(table).lower()
        
        # Check for budget keywords
        has_keywords = any(kw in text for kw in self.BUDGET_KEYWORDS)
        
        # Check for currency patterns
        has_currency = bool(re.search(r'[\$]|cop|pesos|millones', text))
        
        return has_keywords or has_currency
    
    def _extract_kpis(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract KPI data from a classified KPI table.
        
        Attempts to identify indicator name, baseline, target, and unit.
        """
        kpis = []
        headers = table.get("headers", [])
        data_rows = table.get("data_rows", table.get("rows", [])[1:] if len(table.get("rows", [])) > 1 else [])
        
        # Try to identify column indices
        indicator_col = self._find_column(headers, ["indicador", "nombre", "descripción", "indicator"])
        baseline_col = self._find_column(headers, ["línea base", "baseline", "actual", "inicial"])
        target_col = self._find_column(headers, ["meta", "target", "esperado", "objetivo"])
        unit_col = self._find_column(headers, ["unidad", "unit", "medida"])
        year_col = self._find_column(headers, ["año", "year", "periodo"])
        
        # Extract KPI from each data row
        for row in data_rows:
            if not row or len(row) == 0:
                continue
            
            kpi = {
                "table_id": table.get("table_id", "unknown"),
                "page": table.get("page", 0),
            }
            
            # Extract indicator name
            if indicator_col is not None and indicator_col < len(row):
                kpi["indicator"] = _safe_strip(row[indicator_col])
            else:
                # Use first non-empty cell as indicator
                kpi["indicator"] = next((_safe_strip(cell) for cell in row if _safe_strip(cell)), "Unknown")
            
            # Extract baseline
            if baseline_col is not None and baseline_col < len(row):
                kpi["baseline"] = self._parse_numeric(row[baseline_col])
            
            # Extract target
            if target_col is not None and target_col < len(row):
                kpi["target"] = self._parse_numeric(row[target_col])
            
            # Extract unit
            if unit_col is not None and unit_col < len(row):
                kpi["unit"] = _safe_strip(row[unit_col])
            
            # Extract year
            if year_col is not None and year_col < len(row):
                year_text = _safe_strip(row[year_col])
                year_match = re.search(r'20\d{2}', year_text)
                if year_match:
                    kpi["year"] = int(year_match.group())
            
            # Only add if we have at least an indicator
            if kpi.get("indicator"):
                kpis.append(kpi)
        
        return kpis
    
    def _extract_budgets(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract budget data from a classified budget table.
        
        Attempts to identify source, use, amount, and year.
        """
        budgets = []
        headers = table.get("headers", [])
        data_rows = table.get("data_rows", table.get("rows", [])[1:] if len(table.get("rows", [])) > 1 else [])
        
        # Try to identify column indices
        source_col = self._find_column(headers, ["fuente", "source", "origen", "recurso"])
        use_col = self._find_column(headers, ["uso", "destino", "aplicación", "proyecto", "programa"])
        amount_col = self._find_column(headers, ["monto", "valor", "amount", "presupuesto", "costo"])
        year_col = self._find_column(headers, ["año", "year", "vigencia", "periodo"])
        
        # Extract budget from each data row
        for row in data_rows:
            if not row or len(row) == 0:
                continue
            
            budget = {
                "table_id": table.get("table_id", "unknown"),
                "page": table.get("page", 0),
            }
            
            # Extract source
            if source_col is not None and source_col < len(row):
                budget["source"] = _safe_strip(row[source_col])
            else:
                budget["source"] = "Unknown"
            
            # Extract use
            if use_col is not None and use_col < len(row):
                budget["use"] = _safe_strip(row[use_col])
            else:
                # Use first non-empty cell if no use column
                budget["use"] = next((_safe_strip(cell) for cell in row if _safe_strip(cell)), "Unknown")
            
            # Extract amount (look in all cells if no specific column)
            amount = None
            if amount_col is not None and amount_col < len(row):
                amount = self._parse_currency(row[amount_col])
            else:
                # Search all cells for currency values
                for cell in row:
                    parsed = self._parse_currency(cell)
                    if parsed is not None:
                        amount = parsed
                        break
            
            if amount is not None:
                budget["amount"] = amount
            
            # Extract year
            if year_col is not None and year_col < len(row):
                year_text = _safe_strip(row[year_col])
                year_match = re.search(r'20\d{2}', year_text)
                if year_match:
                    budget["year"] = int(year_match.group())
            
            # Only add if we have meaningful data
            if budget.get("amount") or budget.get("source") != "Unknown":
                budgets.append(budget)
        
        return budgets
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table structure to searchable text."""
        parts = []
        
        # Add headers
        headers = table.get("headers", [])
        if headers:
            parts.append(" ".join(str(h) for h in headers))
        
        # Add all rows
        for row in table.get("rows", []):
            parts.append(" ".join(str(cell) for cell in row))
        
        return " ".join(parts)
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords in headers."""
        if not headers:
            return None
        
        for idx, header in enumerate(headers):
            header_lower = str(header).lower()
            for keyword in keywords:
                if keyword in header_lower:
                    return idx
        
        return None
    
    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""
        if not text:
            return None
        
        try:
            # Remove common separators and extract number
            cleaned = re.sub(r'[^\d.,\-]', '', str(text))
            cleaned = cleaned.replace(',', '.')
            
            # Handle percentage
            if '%' in str(text):
                value = float(cleaned)
                return value
            
            return float(cleaned) if cleaned else None
        except (ValueError, AttributeError):
            return None
    
    def _parse_currency(self, text: str) -> Optional[float]:
        """Parse currency value from text."""
        if not text:
            return None
        
        try:
            text_str = str(text).lower()
            
            # Remove currency symbols and text
            cleaned = re.sub(r'[^\d.,\-]', '', text_str)
            cleaned = cleaned.replace(',', '')
            
            if not cleaned:
                return None
            
            value = float(cleaned)
            
            # Adjust for "millones" or "millions"
            if 'millon' in text_str or 'million' in text_str:
                value *= 1_000_000
            
            # Adjust for "miles" or "thousands"
            if 'miles' in text_str or 'thousand' in text_str:
                value *= 1_000
            
            return value
        except (ValueError, AttributeError):
            return None
