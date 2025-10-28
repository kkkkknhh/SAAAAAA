"""Tests for JSONContractLoader."""
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.contract_loader import (
    LoadError,
    LoadResult,
    JSONContractLoader,
)


def test_load_error_creation():
    """Test creating LoadError objects."""
    error = LoadError(
        file_path="/path/to/file.json",
        error_type="JSONDecodeError",
        message="Invalid JSON",
        line_number=5,
    )
    
    assert error.file_path == "/path/to/file.json"
    assert error.error_type == "JSONDecodeError"
    assert error.message == "Invalid JSON"
    assert error.line_number == 5
    
    # Test string representation
    error_str = str(error)
    assert "/path/to/file.json:5" in error_str
    assert "JSONDecodeError" in error_str
    
    print("✓ LoadError creation works")


def test_load_result_creation():
    """Test creating LoadResult objects."""
    result = LoadResult(success=True, data={"key": "value"})
    
    assert result.success is True
    assert result.data == {"key": "value"}
    assert len(result.errors) == 0
    assert len(result.files_loaded) == 0
    
    print("✓ LoadResult creation works")


def test_load_result_add_error():
    """Test adding errors to LoadResult."""
    result = LoadResult(success=True)
    
    result.add_error(
        file_path="test.json",
        error_type="TestError",
        message="Test error message",
    )
    
    assert len(result.errors) == 1
    assert result.errors[0].file_path == "test.json"
    assert result.errors[0].error_type == "TestError"
    
    print("✓ LoadResult add_error works")


def test_load_result_merge():
    """Test merging LoadResults."""
    result1 = LoadResult(success=True, data={"a": 1})
    result1.files_loaded.append("file1.json")
    
    result2 = LoadResult(success=True, data={"b": 2})
    result2.files_loaded.append("file2.json")
    result2.add_error("file2.json", "Warning", "Test warning")
    
    result1.merge(result2)
    
    assert result1.data == {"a": 1, "b": 2}
    assert len(result1.files_loaded) == 2
    assert len(result1.errors) == 1
    
    print("✓ LoadResult merge works")


def test_loader_initialization():
    """Test JSONContractLoader initialization."""
    loader = JSONContractLoader()
    
    assert len(loader.base_paths) > 0
    assert loader.validate_schema is False
    assert loader.schema_validator is None
    
    # With custom base paths
    loader2 = JSONContractLoader(base_paths=[Path("/tmp")])
    assert Path("/tmp") in loader2.base_paths
    
    print("✓ JSONContractLoader initialization works")


def test_path_resolution():
    """Test path resolution logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create a test file
        test_file = tmppath / "test.json"
        test_file.write_text('{"test": true}')
        
        # Create loader with base path
        loader = JSONContractLoader(base_paths=[tmppath])
        
        # Test resolving relative path
        resolved = loader._resolve_path("test.json")
        assert resolved is not None
        assert resolved.exists()
        
        # Test non-existent file
        resolved = loader._resolve_path("nonexistent.json")
        assert resolved is None
        
        print("✓ Path resolution works")


def test_read_payload_valid_json():
    """Test reading valid JSON payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create valid JSON file
        json_file = tmppath / "valid.json"
        json_file.write_text('{"key": "value", "number": 42}')
        
        loader = JSONContractLoader()
        data, error = loader._read_payload(json_file)
        
        assert error is None
        assert data == {"key": "value", "number": 42}
        
        print("✓ Reading valid JSON payload works")


def test_read_payload_invalid_json():
    """Test reading invalid JSON payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create invalid JSON file
        json_file = tmppath / "invalid.json"
        json_file.write_text('{"key": invalid}')
        
        loader = JSONContractLoader()
        data, error = loader._read_payload(json_file)
        
        assert data is None
        assert error is not None
        assert error.error_type == "JSONDecodeError"
        
        print("✓ Reading invalid JSON detects errors")


def test_read_payload_not_object():
    """Test reading JSON that's not an object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create JSON file with array instead of object
        json_file = tmppath / "array.json"
        json_file.write_text('[1, 2, 3]')
        
        loader = JSONContractLoader()
        data, error = loader._read_payload(json_file)
        
        assert data is None
        assert error is not None
        assert error.error_type == "InvalidFormat"
        assert "dict" in error.message
        
        print("✓ Reading non-object JSON detects format error")


def test_load_single_file():
    """Test loading a single JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test file
        json_file = tmppath / "config.json"
        json_file.write_text('{"setting": "value"}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_file("config.json")
        
        assert result.success is True
        assert result.data == {"setting": "value"}
        assert len(result.files_loaded) == 1
        assert len(result.errors) == 0
        
        print("✓ Loading single file works")


def test_load_file_not_found():
    """Test loading non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_file("nonexistent.json")
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "PathResolution" in result.errors[0].error_type or "FileNotFound" in result.errors[0].error_type
        
        print("✓ Loading non-existent file detects error")


def test_load_directory_basic():
    """Test loading all JSON files from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create multiple JSON files
        (tmppath / "config1.json").write_text('{"id": 1}')
        (tmppath / "config2.json").write_text('{"id": 2}')
        (tmppath / "config3.json").write_text('{"id": 3}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_directory(tmppath)
        
        assert result.success is True
        assert len(result.data) == 3
        assert "config1" in result.data
        assert "config2" in result.data
        assert "config3" in result.data
        assert result.data["config1"]["id"] == 1
        assert len(result.files_loaded) == 3
        
        print("✓ Loading directory works")


def test_load_directory_with_pattern():
    """Test loading directory with custom pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create files with different extensions
        (tmppath / "data1.json").write_text('{"type": "json"}')
        (tmppath / "data2.txt").write_text('not json')
        (tmppath / "config.json").write_text('{"type": "config"}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_directory(tmppath, pattern="data*.json")
        
        assert result.success is True
        assert len(result.data) == 1
        assert "data1" in result.data
        assert "config" not in result.data
        
        print("✓ Loading directory with pattern works")


def test_load_directory_recursive():
    """Test recursive directory loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create nested structure
        subdir = tmppath / "subdir"
        subdir.mkdir()
        
        (tmppath / "root.json").write_text('{"level": "root"}')
        (subdir / "nested.json").write_text('{"level": "nested"}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        
        # Non-recursive should only find root file
        result = loader.load_directory(tmppath, recursive=False)
        assert len(result.data) == 1
        assert "root" in result.data
        
        # Recursive should find both
        result = loader.load_directory(tmppath, recursive=True)
        assert result.success is True
        assert len(result.data) == 2
        assert "root" in result.data
        assert "nested" in result.data
        
        print("✓ Recursive directory loading works")


def test_load_directory_error_aggregation():
    """Test error aggregation during directory loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create mix of valid and invalid files
        (tmppath / "valid.json").write_text('{"valid": true}')
        (tmppath / "invalid.json").write_text('{invalid json}')
        (tmppath / "array.json").write_text('[1, 2, 3]')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        
        # With error aggregation (default)
        result = loader.load_directory(tmppath, aggregate_errors=True)
        assert result.success is False  # Has errors
        assert len(result.data) == 1  # One valid file loaded
        assert len(result.errors) == 2  # Two errors
        assert "valid" in result.data
        
        print("✓ Error aggregation works")


def test_load_directory_deterministic_order():
    """Test that files are loaded in deterministic order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create files in non-alphabetical order
        (tmppath / "zebra.json").write_text('{"order": 3}')
        (tmppath / "alpha.json").write_text('{"order": 1}')
        (tmppath / "beta.json").write_text('{"order": 2}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_directory(tmppath)
        
        # Files should be loaded in alphabetical order
        files = result.files_loaded
        assert len(files) == 3
        
        # Extract filenames
        filenames = [Path(f).name for f in files]
        assert filenames == sorted(filenames)
        
        print("✓ Deterministic loading order works")


def test_load_multiple_paths():
    """Test loading multiple files/directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create directory and file
        subdir = tmppath / "configs"
        subdir.mkdir()
        
        (tmppath / "root.json").write_text('{"location": "root"}')
        (subdir / "nested.json").write_text('{"location": "nested"}')
        
        loader = JSONContractLoader(base_paths=[tmppath])
        result = loader.load_multiple([
            "root.json",
            "configs",
        ])
        
        assert result.success is True
        assert "root" in result.data
        assert "nested" in result.data
        
        print("✓ Loading multiple paths works")


def test_schema_validation():
    """Test schema validation hook."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test file
        json_file = tmppath / "test.json"
        json_file.write_text('{"required_field": "value"}')
        
        # Define schema validator
        def validator(data, file_path):
            if "required_field" not in data:
                return False, "Missing required_field"
            return True, None
        
        loader = JSONContractLoader(
            base_paths=[tmppath],
            validate_schema=True,
            schema_validator=validator,
        )
        
        # Should pass validation
        result = loader.load_file("test.json")
        assert result.success is True
        
        # Create file that fails validation
        invalid_file = tmppath / "invalid.json"
        invalid_file.write_text('{"other_field": "value"}')
        
        result = loader.load_file("invalid.json")
        assert result.success is False
        assert len(result.errors) > 0
        assert "SchemaValidation" in result.errors[0].error_type
        
        print("✓ Schema validation works")


def test_format_errors():
    """Test error formatting."""
    loader = JSONContractLoader()
    
    result = LoadResult(success=False)
    result.add_error("file1.json", "Error1", "Message 1")
    result.add_error("file2.json", "Error2", "Message 2")
    
    formatted = loader.format_errors(result)
    
    assert "2 error(s)" in formatted
    assert "file1.json" in formatted
    assert "file2.json" in formatted
    assert "Error1" in formatted
    assert "Error2" in formatted
    
    print("✓ Error formatting works")


if __name__ == "__main__":
    print("Running JSONContractLoader tests...\n")
    
    try:
        test_load_error_creation()
        test_load_result_creation()
        test_load_result_add_error()
        test_load_result_merge()
        test_loader_initialization()
        test_path_resolution()
        test_read_payload_valid_json()
        test_read_payload_invalid_json()
        test_read_payload_not_object()
        test_load_single_file()
        test_load_file_not_found()
        test_load_directory_basic()
        test_load_directory_with_pattern()
        test_load_directory_recursive()
        test_load_directory_error_aggregation()
        test_load_directory_deterministic_order()
        test_load_multiple_paths()
        test_schema_validation()
        test_format_errors()
        
        print("\n✅ All JSONContractLoader tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
