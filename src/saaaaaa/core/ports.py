"""
Port interfaces for dependency injection.

Ports define abstract interfaces for external interactions (I/O, time, environment).
These are implemented by adapters in the infrastructure layer.

This follows the Ports and Adapters (Hexagonal) architecture pattern:
- Ports are in the core layer (no dependencies)
- Adapters are in the infrastructure layer (can import anything)
- Core modules depend on ports (abstractions), not adapters (implementations)

Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class FilePort(Protocol):
    """Port for file system operations.
    
    Implementations provide access to file reading and writing.
    Core modules receive a FilePort instance via dependency injection.
    """

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from a file.
        
        Args:
            path: File path to read
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File contents as string
            
        Raises:
            FileNotFoundError: If file does not exist
            PermissionError: If file cannot be read
        """
        ...

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text to a file.
        
        Args:
            path: File path to write
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            
        Raises:
            PermissionError: If file cannot be written
        """
        ...

    def read_bytes(self, path: str) -> bytes:
        """Read bytes from a file.
        
        Args:
            path: File path to read
            
        Returns:
            File contents as bytes
            
        Raises:
            FileNotFoundError: If file does not exist
            PermissionError: If file cannot be read
        """
        ...

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file.
        
        Args:
            path: File path to write
            content: Bytes content to write
            
        Raises:
            PermissionError: If file cannot be written
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if path exists, False otherwise
        """
        ...

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory.
        
        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            exist_ok: Don't raise error if directory exists
            
        Raises:
            FileExistsError: If directory exists and exist_ok is False
        """
        ...


class JsonPort(Protocol):
    """Port for JSON serialization/deserialization.
    
    Separates JSON operations from file I/O for better composability.
    """

    def loads(self, text: str) -> Any:
        """Parse JSON from string.
        
        Args:
            text: JSON string
            
        Returns:
            Parsed Python object
            
        Raises:
            ValueError: If JSON is invalid
        """
        ...

    def dumps(self, obj: Any, indent: Optional[int] = None) -> str:
        """Serialize object to JSON string.
        
        Args:
            obj: Python object to serialize
            indent: Indentation spaces (None for compact)
            
        Returns:
            JSON string
            
        Raises:
            TypeError: If object is not serializable
        """
        ...


class EnvPort(Protocol):
    """Port for environment variable access.
    
    Allows core modules to access configuration without direct os.environ coupling.
    """

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Environment variable value or default
        """
        ...

    def get_required(self, key: str) -> str:
        """Get required environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If environment variable is not set
        """
        ...

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Boolean value (true/false/yes/no/1/0)
        """
        ...


class ClockPort(Protocol):
    """Port for time operations.
    
    Allows core modules to get current time without direct datetime.now() calls.
    Enables time manipulation in tests.
    """

    def now(self) -> datetime:
        """Get current datetime.
        
        Returns:
            Current datetime
        """
        ...

    def utcnow(self) -> datetime:
        """Get current UTC datetime.
        
        Returns:
            Current UTC datetime
        """
        ...


class LogPort(Protocol):
    """Port for logging operations.
    
    Allows core modules to log without coupling to specific logging framework.
    """

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...


__all__ = [
    'FilePort',
    'JsonPort',
    'EnvPort',
    'ClockPort',
    'LogPort',
]
