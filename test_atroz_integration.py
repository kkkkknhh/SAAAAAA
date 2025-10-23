#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AtroZ Dashboard Integration - Test Suite
=========================================

Tests for API server, data service, and integration components.

Run with:
    pytest test_atroz_integration.py -v

Or:
    python test_atroz_integration.py
"""

import pytest
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api_server import app, APIConfig, DataService

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def data_service():
    """Create data service instance"""
    return DataService()


# ============================================================================
# API SERVER TESTS
# ============================================================================

class TestAPIServer:
    """Test API server endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v1/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_auth_token_missing_credentials(self, client):
        """Test authentication with missing credentials"""
        response = client.post(
            '/api/v1/auth/token',
            json={}
        )
        
        assert response.status_code == 400
    
    def test_auth_token_valid_credentials(self, client):
        """Test authentication with valid credentials"""
        response = client.post(
            '/api/v1/auth/token',
            json={
                'client_id': 'test-client',
                'client_secret': 'test-secret'
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data
        assert data['token_type'] == 'Bearer'
    
    def test_get_pdet_regions(self, client):
        """Test PDET regions endpoint"""
        response = client.get('/api/v1/pdet/regions')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'data' in data
        assert isinstance(data['data'], list)
        assert len(data['data']) > 0
        
        # Validate first region structure
        region = data['data'][0]
        assert 'id' in region
        assert 'name' in region
        assert 'coordinates' in region
        assert 'scores' in region
        assert 'metadata' in region
    
    def test_get_region_detail(self, client):
        """Test region detail endpoint"""
        response = client.get('/api/v1/pdet/regions/alto-patia')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'data' in data
        
        region = data['data']
        assert region['id'] == 'alto-patia'
        assert 'detailed_analysis' in region
    
    def test_get_region_not_found(self, client):
        """Test region detail with invalid ID"""
        response = client.get('/api/v1/pdet/regions/invalid-region')
        
        assert response.status_code == 404
    
    def test_get_municipality_data(self, client):
        """Test municipality data endpoint"""
        response = client.get('/api/v1/municipalities/12345')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'data' in data
        
        municipality = data['data']
        assert 'analysis' in municipality
        assert 'radar' in municipality['analysis']
        assert 'clusters' in municipality['analysis']
    
    def test_get_evidence_stream(self, client):
        """Test evidence stream endpoint"""
        response = client.get('/api/v1/evidence/stream')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'data' in data
        assert isinstance(data['data'], list)
    
    def test_export_dashboard_data(self, client):
        """Test export endpoint"""
        response = client.post(
            '/api/v1/export/dashboard',
            json={
                'format': 'json',
                'regions': ['alto-patia'],
                'include_evidence': True
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'data' in data


# ============================================================================
# DATA SERVICE TESTS
# ============================================================================

class TestDataService:
    """Test data service methods"""
    
    def test_get_pdet_regions(self, data_service):
        """Test PDET regions retrieval"""
        regions = data_service.get_pdet_regions()
        
        assert isinstance(regions, list)
        assert len(regions) > 0
        
        # Validate region structure
        region = regions[0]
        assert 'id' in region
        assert 'name' in region
        assert 'coordinates' in region
        assert 'x' in region['coordinates']
        assert 'y' in region['coordinates']
        assert 'scores' in region
        assert 'overall' in region['scores']
    
    def test_get_region_detail(self, data_service):
        """Test region detail retrieval"""
        detail = data_service.get_region_detail('alto-patia')
        
        assert detail is not None
        assert detail['id'] == 'alto-patia'
        assert 'detailed_analysis' in detail
    
    def test_get_region_detail_invalid(self, data_service):
        """Test region detail with invalid ID"""
        detail = data_service.get_region_detail('invalid-id')
        
        assert detail is None
    
    def test_get_cluster_breakdown(self, data_service):
        """Test cluster breakdown generation"""
        clusters = data_service._get_cluster_breakdown('alto-patia')
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        
        cluster = clusters[0]
        assert 'name' in cluster
        assert 'value' in cluster
        assert 'trend' in cluster
    
    def test_get_question_matrix(self, data_service):
        """Test question matrix generation"""
        questions = data_service._get_question_matrix('alto-patia')
        
        assert isinstance(questions, list)
        assert len(questions) == 44  # 44 questions
        
        question = questions[0]
        assert 'id' in question
        assert 'score' in question
        assert 'category' in question
    
    def test_get_recommendations(self, data_service):
        """Test recommendations generation"""
        recs = data_service._get_recommendations('alto-patia')
        
        assert isinstance(recs, list)
        assert len(recs) > 0
        
        rec = recs[0]
        assert 'priority' in rec
        assert 'text' in rec
        assert 'category' in rec
    
    def test_get_evidence_stream(self, data_service):
        """Test evidence stream retrieval"""
        evidence = data_service.get_evidence_stream()
        
        assert isinstance(evidence, list)
        assert len(evidence) > 0
        
        item = evidence[0]
        assert 'source' in item
        assert 'page' in item
        assert 'text' in item


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test end-to-end integration"""
    
    def test_complete_workflow(self, client):
        """Test complete workflow from regions to export"""
        # 1. Get regions
        response = client.get('/api/v1/pdet/regions')
        assert response.status_code == 200
        regions = response.get_json()['data']
        
        # 2. Get detail for first region
        first_region_id = regions[0]['id']
        response = client.get(f'/api/v1/pdet/regions/{first_region_id}')
        assert response.status_code == 200
        
        # 3. Get evidence stream
        response = client.get('/api/v1/evidence/stream')
        assert response.status_code == 200
        
        # 4. Export data
        response = client.post(
            '/api/v1/export/dashboard',
            json={
                'format': 'json',
                'regions': [first_region_id],
                'include_evidence': True
            }
        )
        assert response.status_code == 200
    
    def test_cache_behavior(self, client):
        """Test caching behavior"""
        # First request
        response1 = client.get('/api/v1/pdet/regions')
        assert response1.status_code == 200
        
        # Second request (should be cached)
        response2 = client.get('/api/v1/pdet/regions')
        assert response2.status_code == 200
        
        # Data should be identical
        assert response1.get_json() == response2.get_json()


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration"""
    
    def test_api_config_defaults(self):
        """Test API configuration defaults"""
        assert APIConfig.SECRET_KEY is not None
        assert APIConfig.JWT_SECRET is not None
        assert APIConfig.JWT_ALGORITHM == 'HS256'
        assert APIConfig.CACHE_TTL > 0
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        assert isinstance(APIConfig.CORS_ORIGINS, list)


# ============================================================================
# RUN STANDALONE
# ============================================================================

if __name__ == '__main__':
    """Run tests standalone"""
    print("=" * 80)
    print("AtroZ Dashboard Integration - Test Suite")
    print("=" * 80)
    print()
    
    # Run with pytest if available
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v', '--color=yes'])
        sys.exit(exit_code)
    except ImportError:
        print("⚠ pytest not installed, running basic tests...")
        print()
        
        # Basic test run
        from api_server import app
        
        with app.test_client() as client:
            # Test health check
            print("Testing health check...")
            response = client.get('/api/v1/health')
            assert response.status_code == 200
            print("✓ Health check passed")
            
            # Test PDET regions
            print("Testing PDET regions...")
            response = client.get('/api/v1/pdet/regions')
            assert response.status_code == 200
            print("✓ PDET regions passed")
            
            # Test evidence stream
            print("Testing evidence stream...")
            response = client.get('/api/v1/evidence/stream')
            assert response.status_code == 200
            print("✓ Evidence stream passed")
        
        print()
        print("=" * 80)
        print("✓ All basic tests passed!")
        print("=" * 80)
        print()
        print("For complete testing, install pytest:")
        print("  pip install pytest")
        print("  pytest test_atroz_integration.py -v")
