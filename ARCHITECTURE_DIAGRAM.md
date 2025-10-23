# AtroZ Dashboard Integration - Architecture Diagram

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   AtroZ Dashboard UI                            │    │
│  │            (deepseek_html_20251022_29a8c3.html)                │    │
│  │                                                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐          │    │
│  │  │  Particle   │  │   PDET      │  │   Evidence   │          │    │
│  │  │   System    │  │  Hexagons   │  │    Ticker    │          │    │
│  │  └─────────────┘  └─────────────┘  └──────────────┘          │    │
│  │                                                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐          │    │
│  │  │   Radar     │  │  Cluster    │  │   Question   │          │    │
│  │  │   Charts    │  │    Bars     │  │    Matrix    │          │    │
│  │  └─────────────┘  └─────────────┘  └──────────────┘          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                 ▲                                        │
│                                 │ DOM Updates                           │
│                                 │                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │          atroz-dashboard-integration.js                         │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  AtrozDashboardIntegration (Main Orchestrator)          │  │    │
│  │  │  • Auto-initialization                                  │  │    │
│  │  │  • Event handling                                       │  │    │
│  │  │  • Auto-refresh (60s default)                          │  │    │
│  │  │  • Export functionality                                 │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  DashboardStateManager                                  │  │    │
│  │  │  • currentView (constellation/macro/meso/micro)        │  │    │
│  │  │  • selectedRegions (Set of IDs)                        │  │    │
│  │  │  • focusMode (boolean)                                 │  │    │
│  │  │  • filters (scoreRange, categories, timeRange)         │  │    │
│  │  │  • History tracking (undo support)                     │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  VisualizationAdapter                                   │  │    │
│  │  │  • adaptPDETRegions()                                   │  │    │
│  │  │  • adaptMunicipalityDetail()                           │  │    │
│  │  │  • formatRadarData()                                    │  │    │
│  │  │  • formatClusterData()                                  │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                 ▲                                        │
│                                 │ Data                                  │
│                                 │                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              atroz-data-service.js                              │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  AtrozDataService (Data Access Layer)                   │  │    │
│  │  │  • fetchPDETRegions()                                   │  │    │
│  │  │  • fetchRegionDetail(id)                                │  │    │
│  │  │  • fetchMunicipalityData(id)                           │  │    │
│  │  │  • fetchEvidenceStream()                                │  │    │
│  │  │  • exportDashboardData(options)                         │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  Cache Layer                                            │  │    │
│  │  │  • Map<cacheKey, data>                                  │  │    │
│  │  │  • Map<cacheKey, timestamp>                             │  │    │
│  │  │  • TTL: 5 minutes (configurable)                        │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  HTTP Client                                            │  │    │
│  │  │  • Retry logic (3 attempts)                             │  │    │
│  │  │  • Authentication (JWT)                                 │  │    │
│  │  │  • Error handling                                       │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  WebSocket Client (Socket.IO)                           │  │    │
│  │  │  • Real-time region updates                             │  │    │
│  │  │  • Auto-reconnect                                       │  │    │
│  │  │  • Event subscriptions                                  │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                │ HTTP/HTTPS + WebSocket
                                │ 
┌───────────────────────────────▼──────────────────────────────────────────┐
│                         API SERVER                                        │
│                         (api_server.py)                                   │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    Flask Application                            │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  Middleware Stack                                               │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │     │
│  │  │ CORS         │→ │ Rate         │→ │ Auth         │         │     │
│  │  │ Handler      │  │ Limiter      │  │ (JWT)        │         │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  REST API Endpoints                                             │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │  GET  /api/v1/health                                     │  │     │
│  │  │  POST /api/v1/auth/token                                 │  │     │
│  │  │  GET  /api/v1/pdet/regions                               │  │     │
│  │  │  GET  /api/v1/pdet/regions/<id>                          │  │     │
│  │  │  GET  /api/v1/municipalities/<id>                        │  │     │
│  │  │  GET  /api/v1/evidence/stream                            │  │     │
│  │  │  POST /api/v1/export/dashboard                           │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  WebSocket Server (Socket.IO)                                  │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │  Events:                                                 │  │     │
│  │  │  • connect                                               │  │     │
│  │  │  • disconnect                                            │  │     │
│  │  │  • subscribe_region                                      │  │     │
│  │  │  • region_update (emit)                                  │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  Cache Layer                                                    │     │
│  │  • In-memory cache (Dict)                                      │     │
│  │  • TTL: 5 minutes                                              │     │
│  │  • Cache invalidation                                          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  DataService (Business Logic)                                  │     │
│  │  • get_pdet_regions()                                          │     │
│  │  • get_region_detail(id)                                       │     │
│  │  • get_municipality_data(id)                                   │     │
│  │  • get_evidence_stream()                                       │     │
│  │  • _get_cluster_breakdown(id)                                  │     │
│  │  • _get_question_matrix(id)                                    │     │
│  │  • _get_recommendations(id)                                    │     │
│  └────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 │ Python Integration
                                 │
┌────────────────────────────────▼──────────────────────────────────────────┐
│                    SAAAAAA ORCHESTRATOR                                    │
│                    (orchestrator.py)                                       │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  PolicyAnalysisOrchestrator                                     │      │
│  │  ┌──────────────────────────────────────────────────────────┐  │      │
│  │  │  execute_chess_strategy()                                │  │      │
│  │  │  • OPENING: Execute all 300 questions                    │  │      │
│  │  │  • MIDDLE GAME: Analyze 6 modalities                     │  │      │
│  │  │  • ENDGAME: MICRO → MESO → MACRO synthesis              │  │      │
│  │  └──────────────────────────────────────────────────────────┘  │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  ExecutionChoreographer (choreographer.py)                      │      │
│  │  • execute_question(spec, doc, metadata)                        │      │
│  │  • Method-level granularity (584 methods)                       │      │
│  │  • Deterministic pipeline execution                             │      │
│  │  • Complete provenance tracking                                 │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  9 Producer Modules (7 Producers + 1 Aggregator + 1 Analyzer)  │      │
│  │  ┌──────────────────────────────────────────────────────────┐  │      │
│  │  │  1. dereck_beach.py (99 methods)                         │  │      │
│  │  │     • BeachEvidentialTest                                │  │      │
│  │  │     • CausalExtractor                                    │  │      │
│  │  │     • BayesianMechanismInference                         │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  2. policy_processor.py (32 methods)                     │  │      │
│  │  │     • IndustrialPolicyProcessor                          │  │      │
│  │  │     • BayesianEvidenceScorer                             │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  3. embedding_policy.py (36 methods)                     │  │      │
│  │  │     • AdvancedSemanticChunker                            │  │      │
│  │  │     • BayesianNumericalAnalyzer                          │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  4. semantic_chunking_policy.py (15 methods)             │  │      │
│  │  │     • SemanticProcessor                                  │  │      │
│  │  │     • BayesianEvidenceIntegrator                         │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  5. teoria_cambio.py (30 methods)                        │  │      │
│  │  │     • TeoriaCambio                                       │  │      │
│  │  │     • AdvancedDAGValidator                               │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  6. contradiction_deteccion.py (62 methods)              │  │      │
│  │  │     • PolicyContradictionDetector                        │  │      │
│  │  │     • TemporalLogicVerifier                              │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  7. financiero_viabilidad_tablas.py (65 methods)         │  │      │
│  │  │     • PDETMunicipalPlanAnalyzer                          │  │      │
│  │  │     • FinancialAuditor                                   │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  8. report_assembly.py (43 methods)                      │  │      │
│  │  │     • ReportAssembler                                    │  │      │
│  │  │     • MICRO/MESO/MACRO generators                        │  │      │
│  │  │  ──────────────────────────────────────────────────────  │  │      │
│  │  │  9. Analyzer_one.py (34 methods)                         │  │      │
│  │  │     • MunicipalAnalyzer                                  │  │      │
│  │  │     • SemanticAnalyzer                                   │  │      │
│  │  └──────────────────────────────────────────────────────────┘  │      │
│  │                    TOTAL: 584 METHODS                           │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  Canonical Truth Model                                          │      │
│  │  • cuestionario_FIXED.json (300 questions)                      │      │
│  │  • execution_mapping.yaml (method chains)                       │      │
│  │  • COMPLETE_METHOD_CLASS_MAP.json (method registry)             │      │
│  └────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence

### 1. Initial Load

```
User → Browser → index.html loads
                ↓
        atroz-data-service.js loads
                ↓
        atroz-dashboard-integration.js loads
                ↓
        Auto-initialization triggers
                ↓
        fetchPDETRegions() called
                ↓
        HTTP GET /api/v1/pdet/regions
                ↓
        API Server → DataService.get_pdet_regions()
                ↓
        (Future: Orchestrator integration)
                ↓
        Response → Cache → Client
                ↓
        VisualizationAdapter.adaptPDETRegions()
                ↓
        Update PDET hexagons in DOM
```

### 2. Region Detail Click

```
User clicks PDET hexagon
        ↓
openMunicipalityDetail(region) triggered
        ↓
fetchRegionDetail(region.id) called
        ↓
Check cache → if miss:
        ↓
HTTP GET /api/v1/pdet/regions/{id}
        ↓
API Server → DataService.get_region_detail()
        ↓
Response with detailed_analysis
        ↓
VisualizationAdapter.adaptMunicipalityDetail()
        ↓
Populate modal with:
    • Radar chart
    • Cluster bars
    • Question matrix (44 items)
    • Recommendations
```

### 3. Real-Time Update

```
Data changes on server
        ↓
WebSocket emit 'region_update'
        ↓
Client receives via Socket.IO
        ↓
handleRegionUpdate(data) triggered
        ↓
Cache invalidated for region
        ↓
Notify subscribers
        ↓
Automatic DOM update
```

### 4. Export Workflow

```
User clicks Export button
        ↓
integration.exportData(options) called
        ↓
HTTP POST /api/v1/export/dashboard
        {
            format: 'json',
            regions: ['alto-patia', 'arauca'],
            include_evidence: true
        }
        ↓
API Server collects data:
    • Region details
    • Evidence stream
    • Analysis results
        ↓
Response with complete export
        ↓
downloadExport() triggers browser download
```

## Technology Stack

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Custom properties, animations, grid/flexbox
- **Vanilla JavaScript**: No framework dependencies
- **Socket.IO Client**: WebSocket support
- **Canvas API**: Particle system & neural connections

### Backend
- **Python 3.10+**: Core language
- **Flask 3.0**: Web framework
- **Flask-CORS**: CORS handling
- **Flask-SocketIO**: WebSocket server
- **PyJWT**: JWT authentication
- **PyYAML**: Configuration parsing

### Integration
- **REST API**: HTTP/JSON communication
- **WebSocket**: Real-time bidirectional updates
- **JWT**: Stateless authentication
- **JSON Schemas**: Data validation

## Security Layers

```
┌─────────────────────────────────────┐
│  1. CORS Validation                 │
│     • Whitelist origins             │
│     • Credential support            │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  2. Rate Limiting                   │
│     • Per-IP tracking               │
│     • 1000 req / 15 min             │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  3. JWT Authentication              │
│     • Token validation              │
│     • Expiration check (24h)        │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  4. Input Validation                │
│     • Type checking                 │
│     • Range validation              │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  5. Error Handling                  │
│     • Graceful degradation          │
│     • No sensitive info leak        │
└─────────────────────────────────────┘
```

---

**Last Updated:** 2025-10-22  
**Version:** 1.0.0
