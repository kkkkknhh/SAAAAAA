#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AtroZ Dashboard API Server - REST API Integration Layer
========================================================

Provides REST API endpoints for AtroZ Dashboard integration with the SAAAAAA orchestrator.

ARCHITECTURE:
- Flask-based REST API server
- CORS-enabled for dashboard access
- JWT authentication support
- Rate limiting and caching
- WebSocket support for real-time updates
- Integration with orchestrator.py for data processing

ENDPOINTS:
- /api/v1/pdet/regions - Get all PDET regions with scores
- /api/v1/pdet/regions/<id> - Get specific region detail
- /api/v1/municipalities/<id> - Get municipality analysis
- /api/v1/analysis/clusters/<region_id> - Get cluster analysis
- /api/v1/questions/matrix/<municipality_id> - Get question matrix
- /api/v1/evidence/stream - Get evidence stream for ticker
- /api/v1/export/dashboard - Export dashboard data

Author: Integration Team
Version: 1.0.0
Python: 3.10+
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.exceptions import HTTPException
import jwt

# Import orchestrator components
from orchestrator import PolicyAnalysisOrchestrator, OrchestratorConfig
from report_assembly import MicroLevelAnswer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class APIConfig:
    """API Server Configuration"""
    SECRET_KEY = os.getenv('ATROZ_API_SECRET', 'dev-secret-key-change-in-production')
    JWT_SECRET = os.getenv('ATROZ_JWT_SECRET', 'jwt-secret-key-change-in-production')
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('ATROZ_CORS_ORIGINS', '*').split(',')
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv('ATROZ_RATE_LIMIT', 'true').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.getenv('ATROZ_RATE_LIMIT_REQUESTS', '1000'))
    RATE_LIMIT_WINDOW = int(os.getenv('ATROZ_RATE_LIMIT_WINDOW', '900'))  # 15 minutes
    
    # Cache Configuration
    CACHE_ENABLED = os.getenv('ATROZ_CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_TTL = int(os.getenv('ATROZ_CACHE_TTL', '300'))  # 5 minutes
    
    # Data Paths
    DATA_DIRECTORY = os.getenv('ATROZ_DATA_DIR', 'output')
    CACHE_DIRECTORY = os.getenv('ATROZ_CACHE_DIR', 'cache')


# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = APIConfig.SECRET_KEY

# Enable CORS
CORS(app, origins=APIConfig.CORS_ORIGINS, supports_credentials=True)

# Enable WebSocket
socketio = SocketIO(app, cors_allowed_origins=APIConfig.CORS_ORIGINS)

# Initialize cache
cache = {}
cache_timestamps = {}

# Initialize rate limiter
request_counts = {}


# ============================================================================
# MIDDLEWARE & DECORATORS
# ============================================================================

def generate_jwt_token(client_id: str) -> str:
    """Generate JWT token for client authentication"""
    payload = {
        'client_id': client_id,
        'exp': datetime.utcnow() + timedelta(hours=APIConfig.JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, APIConfig.JWT_SECRET, algorithm=APIConfig.JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, APIConfig.JWT_SECRET, algorithms=[APIConfig.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(f):
    """Decorator for JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.jwt_payload = payload
        return f(*args, **kwargs)
    
    return decorated_function


def rate_limit(f):
    """Decorator for rate limiting"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not APIConfig.RATE_LIMIT_ENABLED:
            return f(*args, **kwargs)
        
        client_ip = request.remote_addr
        current_time = datetime.now().timestamp()
        
        # Initialize or clean up request counter
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        
        # Remove old requests outside the window
        request_counts[client_ip] = [
            ts for ts in request_counts[client_ip]
            if current_time - ts < APIConfig.RATE_LIMIT_WINDOW
        ]
        
        # Check if limit exceeded
        if len(request_counts[client_ip]) >= APIConfig.RATE_LIMIT_REQUESTS:
            return jsonify({
                'error': 'Rate limit exceeded',
                'limit': APIConfig.RATE_LIMIT_REQUESTS,
                'window': APIConfig.RATE_LIMIT_WINDOW
            }), 429
        
        # Add current request
        request_counts[client_ip].append(current_time)
        
        return f(*args, **kwargs)
    
    return decorated_function


def cached(ttl: int = APIConfig.CACHE_TTL):
    """Decorator for caching responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not APIConfig.CACHE_ENABLED:
                return f(*args, **kwargs)
            
            # Generate cache key from function name and arguments
            cache_key = f"{f.__name__}:{request.path}:{request.query_string.decode()}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            current_time = datetime.now().timestamp()
            
            # Check cache
            if cache_hash in cache:
                timestamp = cache_timestamps.get(cache_hash, 0)
                if current_time - timestamp < ttl:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cache[cache_hash]
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Store in cache
            cache[cache_hash] = result
            cache_timestamps[cache_hash] = current_time
            
            logger.debug(f"Cache miss: {cache_key}")
            return result
        
        return decorated_function
    return decorator


# ============================================================================
# MOCK DATA SERVICE (Replace with real orchestrator integration)
# ============================================================================

class DataService:
    """Service layer for data retrieval and transformation"""
    
    def __init__(self):
        """Initialize data service with orchestrator"""
        self.orchestrator = None
        self.data_cache = {}
        logger.info("DataService initialized")
    
    def get_pdet_regions(self) -> List[Dict[str, Any]]:
        """
        Get all PDET regions with scores
        
        Returns data in format expected by AtroZ dashboard
        """
        # PDET regions from Colombian government definition
        regions = [
            {
                'id': 'alto-patia',
                'name': 'ALTO PATÍA Y NORTE DEL CAUCA',
                'coordinates': {'x': 25, 'y': 20},
                'metadata': {
                    'municipalities': 24,
                    'population': 450000,
                    'area': 12500
                },
                'scores': {
                    'overall': 72,
                    'governance': 68,
                    'social': 74,
                    'economic': 70,
                    'environmental': 75,
                    'lastUpdated': datetime.now().isoformat()
                },
                'connections': ['pacifico-medio', 'sur-tolima'],
                'indicators': {
                    'alignment': 0.72,
                    'implementation': 0.68,
                    'impact': 0.75
                }
            },
            {
                'id': 'arauca',
                'name': 'ARAUCA',
                'coordinates': {'x': 75, 'y': 15},
                'metadata': {
                    'municipalities': 4,
                    'population': 95000,
                    'area': 23818
                },
                'scores': {
                    'overall': 68,
                    'governance': 65,
                    'social': 70,
                    'economic': 67,
                    'environmental': 71,
                    'lastUpdated': datetime.now().isoformat()
                },
                'connections': ['catatumbo'],
                'indicators': {
                    'alignment': 0.68,
                    'implementation': 0.65,
                    'impact': 0.70
                }
            },
            {
                'id': 'bajo-cauca',
                'name': 'BAJO CAUCA Y NORDESTE ANTIOQUEÑO',
                'coordinates': {'x': 45, 'y': 25},
                'metadata': {
                    'municipalities': 13,
                    'population': 280000,
                    'area': 8485
                },
                'scores': {
                    'overall': 65,
                    'governance': 62,
                    'social': 66,
                    'economic': 64,
                    'environmental': 68,
                    'lastUpdated': datetime.now().isoformat()
                },
                'connections': ['sur-cordoba', 'sur-bolivar'],
                'indicators': {
                    'alignment': 0.65,
                    'implementation': 0.62,
                    'impact': 0.67
                }
            },
            # Add remaining 13 PDET regions...
        ]
        
        return regions
    
    def get_region_detail(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific region"""
        regions = self.get_pdet_regions()
        for region in regions:
            if region['id'] == region_id:
                # Add detailed analysis
                region['detailed_analysis'] = {
                    'cluster_breakdown': self._get_cluster_breakdown(region_id),
                    'question_matrix': self._get_question_matrix(region_id),
                    'recommendations': self._get_recommendations(region_id),
                    'evidence': self._get_evidence_for_region(region_id)
                }
                return region
        return None
    
    def _get_cluster_breakdown(self, region_id: str) -> List[Dict[str, Any]]:
        """Get cluster analysis for region"""
        return [
            {'name': 'GOBERNANZA', 'value': 72, 'trend': 0.05},
            {'name': 'SOCIAL', 'value': 68, 'trend': 0.02},
            {'name': 'ECONÓMICO', 'value': 81, 'trend': -0.03},
            {'name': 'AMBIENTAL', 'value': 76, 'trend': 0.07}
        ]
    
    def _get_question_matrix(self, region_id: str) -> List[Dict[str, Any]]:
        """Get question matrix (44 questions) for region"""
        import random
        questions = []
        for i in range(1, 45):
            score = random.uniform(0.4, 1.0)
            questions.append({
                'id': i,
                'text': f'Pregunta {i}',
                'score': score,
                'category': f'D{(i-1)//7 + 1}',
                'evidence': [f'PDT Sección {i//10 + 1}'],
                'recommendations': [f'Recomendación {i}'] if score < 0.7 else []
            })
        return questions
    
    def _get_recommendations(self, region_id: str) -> List[Dict[str, Any]]:
        """Get strategic recommendations for region"""
        return [
            {
                'priority': 'ALTA',
                'text': 'Fortalecer mecanismos de participación ciudadana',
                'category': 'GOBERNANZA',
                'impact': 'HIGH'
            },
            {
                'priority': 'ALTA',
                'text': 'Implementar sistema de monitoreo continuo',
                'category': 'SEGUIMIENTO',
                'impact': 'HIGH'
            },
            {
                'priority': 'MEDIA',
                'text': 'Mejorar articulación interinstitucional',
                'category': 'INSTITUCIONAL',
                'impact': 'MEDIUM'
            }
        ]
    
    def _get_evidence_for_region(self, region_id: str) -> List[Dict[str, Any]]:
        """Get evidence items for region"""
        return [
            {
                'source': 'PDT Sección 3.2',
                'page': 45,
                'text': 'Implementación de estrategias municipales',
                'relevance': 0.92
            },
            {
                'source': 'PDT Capítulo 4',
                'page': 67,
                'text': 'Articulación con Decálogo DDHH',
                'relevance': 0.88
            }
        ]
    
    def get_evidence_stream(self) -> List[Dict[str, Any]]:
        """Get evidence stream for ticker display"""
        return [
            {
                'source': 'PDT Sección 3.2',
                'page': 45,
                'text': 'Implementación de estrategias municipales',
                'timestamp': datetime.now().isoformat()
            },
            {
                'source': 'PDT Capítulo 4',
                'page': 67,
                'text': 'Articulación con Decálogo DDHH',
                'timestamp': datetime.now().isoformat()
            },
            {
                'source': 'Anexo Técnico',
                'page': 112,
                'text': 'Indicadores de cumplimiento',
                'timestamp': datetime.now().isoformat()
            }
        ]


# Initialize data service
data_service = DataService()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/v1/auth/token', methods=['POST'])
@rate_limit
def get_auth_token():
    """Get authentication token"""
    data = request.get_json()
    client_id = data.get('client_id')
    client_secret = data.get('client_secret')
    
    # Validate credentials (implement proper validation in production)
    if not client_id or not client_secret:
        return jsonify({'error': 'Missing credentials'}), 400
    
    # Generate token
    token = generate_jwt_token(client_id)
    
    return jsonify({
        'access_token': token,
        'token_type': 'Bearer',
        'expires_in': APIConfig.JWT_EXPIRATION_HOURS * 3600
    })


@app.route('/api/v1/pdet/regions', methods=['GET'])
@rate_limit
@cached(ttl=300)
def get_pdet_regions():
    """
    Get all PDET regions with scores
    
    Returns:
        List of PDET regions with metadata and scores
    """
    try:
        regions = data_service.get_pdet_regions()
        
        return jsonify({
            'status': 'success',
            'data': regions,
            'count': len(regions),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Failed to get PDET regions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/pdet/regions/<region_id>', methods=['GET'])
@rate_limit
@cached(ttl=300)
def get_region_detail(region_id: str):
    """
    Get detailed information for a specific PDET region
    
    Args:
        region_id: Region identifier (e.g., 'alto-patia')
    
    Returns:
        Detailed region data with analysis
    """
    try:
        region = data_service.get_region_detail(region_id)
        
        if not region:
            return jsonify({'error': 'Region not found'}), 404
        
        return jsonify({
            'status': 'success',
            'data': region,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Failed to get region detail: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/municipalities/<municipality_id>', methods=['GET'])
@rate_limit
@cached(ttl=300)
def get_municipality_data(municipality_id: str):
    """
    Get municipality analysis data
    
    Args:
        municipality_id: Municipality identifier
    
    Returns:
        Municipality analysis with scores and recommendations
    """
    try:
        # Mock data - integrate with orchestrator for real analysis
        municipality_data = {
            'id': municipality_id,
            'name': f'Municipality {municipality_id}',
            'region_id': 'alto-patia',
            'analysis': {
                'radar': {
                    'dimensions': ['Gobernanza', 'Social', 'Económico', 'Ambiental', 'Institucional', 'Territorial'],
                    'scores': [72, 68, 81, 76, 70, 74]
                },
                'clusters': data_service._get_cluster_breakdown('alto-patia'),
                'questions': data_service._get_question_matrix('alto-patia')
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': municipality_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Failed to get municipality data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/evidence/stream', methods=['GET'])
@rate_limit
@cached(ttl=60)
def get_evidence_stream():
    """
    Get evidence stream for ticker display
    
    Returns:
        List of evidence items with sources and timestamps
    """
    try:
        evidence = data_service.get_evidence_stream()
        
        return jsonify({
            'status': 'success',
            'data': evidence,
            'count': len(evidence),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Failed to get evidence stream: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/export/dashboard', methods=['POST'])
@rate_limit
def export_dashboard_data():
    """
    Export dashboard data in various formats
    
    Request body:
        {
            "format": "json|csv|pdf",
            "regions": ["region_id1", "region_id2"],
            "include_evidence": true
        }
    
    Returns:
        Exported data file
    """
    try:
        data = request.get_json()
        export_format = data.get('format', 'json')
        region_ids = data.get('regions', [])
        include_evidence = data.get('include_evidence', False)
        
        # Collect data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'regions': [],
            'evidence': [] if include_evidence else None
        }
        
        # Get region data
        for region_id in region_ids:
            region = data_service.get_region_detail(region_id)
            if region:
                export_data['regions'].append(region)
        
        # Get evidence if requested
        if include_evidence:
            export_data['evidence'] = data_service.get_evidence_stream()
        
        # Format response based on requested format
        if export_format == 'json':
            return jsonify({
                'status': 'success',
                'data': export_data
            })
        else:
            return jsonify({'error': f'Format {export_format} not yet implemented'}), 400
    
    except Exception as e:
        logger.error(f"Failed to export dashboard data: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEBSOCKET HANDLERS FOR REAL-TIME UPDATES
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_region')
def handle_subscribe_region(data):
    """Subscribe to region updates"""
    region_id = data.get('region_id')
    logger.info(f"Client {request.sid} subscribed to region: {region_id}")
    
    # Send initial data
    region = data_service.get_region_detail(region_id)
    emit('region_update', region)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handle HTTP exceptions"""
    return jsonify({
        'error': e.description,
        'status_code': e.code
    }), e.code


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run API server"""
    logger.info("=" * 80)
    logger.info("AtroZ Dashboard API Server")
    logger.info("=" * 80)
    logger.info(f"CORS Origins: {APIConfig.CORS_ORIGINS}")
    logger.info(f"Rate Limiting: {APIConfig.RATE_LIMIT_ENABLED}")
    logger.info(f"Caching: {APIConfig.CACHE_ENABLED}")
    logger.info("=" * 80)
    
    # Run server
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('ATROZ_API_PORT', '5000')),
        debug=os.getenv('ATROZ_DEBUG', 'false').lower() == 'true'
    )


if __name__ == '__main__':
    main()
