"""
Table extraction and budget handling.

Extracts tables, KPIs, and budget data from documents.
"""

from typing import Any, Dict, List


class TableExtractor:
    """Table and budget data extractor."""
    
    def extract(self, raw_objects: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tables and budget data.
        
        Args:
            raw_objects: Raw parsed objects
            
        Returns:
            Subgraph with tables, KPIs, and budgets
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
            
            # Classify tables as KPI or budget
            for table in tables:
                if self._is_kpi_table(table):
                    kpis = self._extract_kpis(table)
                    tables_subgraph["kpis"].extend(kpis)
                elif self._is_budget_table(table):
                    budgets = self._extract_budgets(table)
                    tables_subgraph["budgets"].extend(budgets)
        
        return tables_subgraph
    
    def _extract_tables_from_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables from page."""
        # Simplified: would use pdf-table-extract
        return []
    
    def _is_kpi_table(self, table: Dict[str, Any]) -> bool:
        """Check if table contains KPI data."""
        # Look for KPI keywords
        text = str(table)
        keywords = ["indicador", "meta", "línea base", "baseline"]
        return any(kw in text.lower() for kw in keywords)
    
    def _is_budget_table(self, table: Dict[str, Any]) -> bool:
        """Check if table contains budget data."""
        text = str(table)
        keywords = ["presupuesto", "fuente", "uso", "monto", "budget"]
        return any(kw in text.lower() for kw in keywords)
    
    def _extract_kpis(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract KPI data from table."""
        return []
    
    def _extract_budgets(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract budget data from table."""
        return []
