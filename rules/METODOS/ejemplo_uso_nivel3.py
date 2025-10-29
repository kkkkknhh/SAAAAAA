#!/usr/bin/env python3
"""
Ejemplo de uso del JSON de mapeo de métodos NIVEL 3
Sistema de 416 métodos mapeados a 30 preguntas genéricas
"""

import json
from typing import Dict, List, Optional
from collections import defaultdict


class MethodMapAnalyzer:
    """Analizador del mapeo de métodos"""
    
    def __init__(self, json_path: str = 'metodos_completos_nivel3.json'):
        """Carga el JSON de mapeo"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data['metadata']
        self.dimensions = self.data['dimensions']
    
    def find_question(self, question_id: str) -> Optional[Dict]:
        """
        Encuentra una pregunta por su ID
        
        Args:
            question_id: ID de la pregunta (ej: 'D1-Q1')
        
        Returns:
            Diccionario con info de la pregunta o None
        """
        for dimension in self.dimensions:
            for question in dimension['questions']:
                if question['q'] == question_id:
                    return question
        return None
    
    def get_critical_methods(self, question_id: str) -> List[Dict]:
        """
        Extrae métodos críticos (prioridad 3) de una pregunta
        
        Args:
            question_id: ID de la pregunta
        
        Returns:
            Lista de métodos críticos con metadata
        """
        question = self.find_question(question_id)
        if not question:
            return []
        
        critical = []
        for package in question['p']:
            for i, priority in enumerate(package['pr']):
                if priority == 3:
                    critical.append({
                        'file': package['f'],
                        'file_name': self.metadata['files'][package['f']],
                        'class': package['c'],
                        'method': package['m'][i],
                        'type': package['t'][i],
                        'type_name': self.metadata['types'][package['t'][i]]
                    })
        return critical
    
    def count_methods_by_file(self, question_id: str) -> Dict[str, int]:
        """
        Cuenta métodos por archivo para una pregunta
        
        Args:
            question_id: ID de la pregunta
        
        Returns:
            Diccionario {archivo: cantidad}
        """
        question = self.find_question(question_id)
        if not question:
            return {}
        
        counts = defaultdict(int)
        for package in question['p']:
            counts[package['f']] += len(package['m'])
        return dict(counts)
    
    def get_dimension_stats(self, dimension_id: str) -> Dict:
        """
        Obtiene estadísticas de una dimensión
        
        Args:
            dimension_id: ID de la dimensión (ej: 'D1')
        
        Returns:
            Diccionario con estadísticas
        """
        for dimension in self.dimensions:
            if dimension['id'] == dimension_id:
                total_methods = sum(q['m'] for q in dimension['questions'])
                return {
                    'id': dimension['id'],
                    'name': dimension['name'],
                    'questions': len(dimension['questions']),
                    'total_methods': total_methods,
                    'avg_methods_per_question': total_methods / len(dimension['questions'])
                }
        return {}
    
    def find_shared_methods(self) -> Dict[str, List[str]]:
        """
        Identifica métodos que aparecen en múltiples preguntas
        
        Returns:
            Diccionario {método: [lista de preguntas]}
        """
        method_usage = defaultdict(list)
        
        for dimension in self.dimensions:
            for question in dimension['questions']:
                for package in question['p']:
                    for method in package['m']:
                        full_method = f"{package['f']}.{package['c']}.{method}"
                        method_usage[full_method].append(question['q'])
        
        # Filtrar solo los que aparecen en múltiples preguntas
        shared = {k: v for k, v in method_usage.items() if len(v) > 1}
        return dict(sorted(shared.items(), key=lambda x: len(x[1]), reverse=True))
    
    def get_method_flow(self, question_id: str) -> str:
        """
        Obtiene el flujo de ejecución de una pregunta
        
        Args:
            question_id: ID de la pregunta
        
        Returns:
            String con el flujo
        """
        question = self.find_question(question_id)
        return question['flow'] if question else ""
    
    def analyze_complexity(self) -> Dict:
        """
        Analiza la complejidad de cada dimensión
        
        Returns:
            Diccionario con análisis de complejidad
        """
        complexity = []
        for dimension in self.dimensions:
            total_methods = dimension.get('total_methods', 0)
            complexity.append({
                'id': dimension['id'],
                'name': dimension['name'],
                'total_methods': total_methods,
                'questions': len(dimension['questions']),
                'complexity_score': total_methods / len(dimension['questions'])
            })
        
        return sorted(complexity, key=lambda x: x['complexity_score'], reverse=True)


def demo_basic_usage():
    """Demostración básica de uso"""
    print("="*70)
    print("DEMO: Uso Básico del Analizador de Mapeo de Métodos")
    print("="*70)
    
    analyzer = MethodMapAnalyzer()
    
    # 1. Buscar una pregunta específica
    print("\n1. BUSCAR PREGUNTA D1-Q1")
    print("-"*70)
    q = analyzer.find_question('D1-Q1')
    print(f"ID: {q['q']}")
    print(f"Título: {q['t']}")
    print(f"Total métodos: {q['m']}")
    print(f"Flujo: {q['flow']}")
    
    # 2. Métodos críticos
    print("\n2. MÉTODOS CRÍTICOS (Prioridad ★)")
    print("-"*70)
    critical = analyzer.get_critical_methods('D1-Q1')
    for i, m in enumerate(critical[:5], 1):
        print(f"{i}. ★ {m['file']}.{m['class']}.{m['method']}")
        print(f"   Tipo: {m['type_name']}")
    print(f"   ... y {len(critical)-5} más")
    
    # 3. Distribución por archivo
    print("\n3. DISTRIBUCIÓN POR ARCHIVO")
    print("-"*70)
    counts = analyzer.count_methods_by_file('D1-Q3')
    for file_code, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        file_name = analyzer.metadata['files'][file_code]
        print(f"{file_code} ({file_name:40s}): {count:2d} métodos")


def demo_advanced_analysis():
    """Demostración de análisis avanzado"""
    print("\n\n" + "="*70)
    print("DEMO: Análisis Avanzado")
    print("="*70)
    
    analyzer = MethodMapAnalyzer()
    
    # 1. Análisis de complejidad
    print("\n1. RANKING DE COMPLEJIDAD POR DIMENSIÓN")
    print("-"*70)
    complexity = analyzer.analyze_complexity()
    for i, dim in enumerate(complexity, 1):
        print(f"{i}. {dim['id']} - {dim['name']}")
        print(f"   Métodos: {dim['total_methods']}, Preguntas: {dim['questions']}")
        print(f"   Score complejidad: {dim['complexity_score']:.1f} métodos/pregunta")
    
    # 2. Métodos compartidos
    print("\n2. MÉTODOS MÁS REUTILIZADOS")
    print("-"*70)
    shared = analyzer.find_shared_methods()
    top_shared = list(shared.items())[:5]
    for method, questions in top_shared:
        print(f"\n{method}")
        print(f"  Usado en {len(questions)} preguntas: {', '.join(questions)}")
    
    # 3. Estadísticas por dimensión
    print("\n3. ESTADÍSTICAS DETALLADAS - D6 (La más compleja)")
    print("-"*70)
    stats = analyzer.get_dimension_stats('D6')
    print(f"Dimensión: {stats['name']}")
    print(f"Total preguntas: {stats['questions']}")
    print(f"Total métodos: {stats['total_methods']}")
    print(f"Promedio métodos/pregunta: {stats['avg_methods_per_question']:.1f}")


def demo_orchestrator_simulation():
    """Simula cómo un orquestador usaría el JSON"""
    print("\n\n" + "="*70)
    print("DEMO: Simulación de Orquestador")
    print("="*70)
    
    analyzer = MethodMapAnalyzer()
    
    # Simular consulta del usuario
    user_query = "D1-Q3"  # Asignación de Recursos
    
    print(f"\nUsuario pregunta: {user_query}")
    print("-"*70)
    
    question = analyzer.find_question(user_query)
    print(f"Pregunta: {question['t']}")
    print(f"Requiere {question['m']} métodos")
    print(f"\nFlujo de ejecución:")
    print(f"  {question['flow']}")
    
    print("\n\nPLAN DE EJECUCIÓN (Solo métodos críticos ★):")
    print("-"*70)
    
    critical = analyzer.get_critical_methods(user_query)
    
    # Agrupar por archivo para ejecución eficiente
    by_file = defaultdict(list)
    for m in critical:
        by_file[m['file']].append(m)
    
    for i, (file_code, methods) in enumerate(by_file.items(), 1):
        file_name = analyzer.metadata['files'][file_code]
        print(f"\n{i}. Ejecutar en {file_code} ({file_name}):")
        for m in methods:
            print(f"   ★ {m['class']}.{m['method']}() [{m['type_name']}]")


def demo_special_features():
    """Muestra las características especiales"""
    print("\n\n" + "="*70)
    print("DEMO: Características Especiales")
    print("="*70)
    
    analyzer = MethodMapAnalyzer()
    features = analyzer.data.get('special_features', {})
    
    # Sistema Bicameral
    print("\n1. SISTEMA BICAMERAL")
    print("-"*70)
    bicameral = features.get('bicameral_system', {})
    print(f"Preguntas: {', '.join(bicameral['questions'])}")
    print(f"\nRuta 1 - {bicameral['route_1']['name']}:")
    print(f"  Método: {bicameral['route_1']['method']}")
    print(f"\nRuta 2 - {bicameral['route_2']['name']}:")
    print(f"  Método: {bicameral['route_2']['method']}")
    
    # Derek Beach Tests
    print("\n2. DEREK BEACH PROCESS TRACING")
    print("-"*70)
    beach = features.get('derek_beach_process_tracing', {})
    print(f"Integración: {beach['integration']}")
    print("\nTests Evidenciales:")
    for test in beach['tests']:
        print(f"\n  • {test['name']}")
        print(f"    Tipo: {test['type']}")
        print(f"    Método: {test['method']}")
        print(f"    Lógica: {test['logic']}")
    
    # Anti-Milagro
    print("\n3. VALIDACIÓN ANTI-MILAGRO")
    print("-"*70)
    anti_milagro = features.get('anti_milagro_validation', {})
    print(f"Pregunta: {anti_milagro['question']}")
    print("Categorías de patrones:")
    for cat in anti_milagro['categories']:
        print(f"  • {cat}")


def main():
    """Función principal que ejecuta todas las demos"""
    try:
        demo_basic_usage()
        demo_advanced_analysis()
        demo_orchestrator_simulation()
        demo_special_features()
        
        print("\n" + "="*70)
        print("DEMOS COMPLETADAS ✅")
        print("="*70)
        
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'metodos_completos_nivel3.json'")
        print("Asegúrate de que el archivo esté en el mismo directorio.")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
