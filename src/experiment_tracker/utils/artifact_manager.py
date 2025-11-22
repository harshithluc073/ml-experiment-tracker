"""
Artifact Manager for ML Experiment Tracker.

Handles comprehensive artifact management including:
- File organization and copying
- Metadata tracking
- Artifact versioning
- Multiple artifact types (models, datasets, plots, logs, etc.)
"""

import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json


class ArtifactType:
    """Standard artifact types."""
    
    MODEL = "model"
    DATASET = "dataset"
    PLOT = "plot"
    IMAGE = "image"
    LOG = "log"
    CONFIG = "config"
    CHECKPOINT = "checkpoint"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    OTHER = "other"


class Artifact:
    """
    Represents a single artifact with metadata.
    
    Tracks all information about an artifact including its location,
    type, size, hash, and custom metadata.
    """
    
    def __init__(
        self,
        path: Path,
        artifact_type: str,
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an artifact.
        
        Args:
            path: Path to the artifact file/directory
            artifact_type: Type of artifact (from ArtifactType)
            name: Optional custom name
            description: Optional description
            metadata: Optional custom metadata dictionary
        """
        self.path = Path(path)
        self.artifact_type = artifact_type
        self.name = name or self.path.name
        self.description = description
        self.metadata = metadata or {}
        
        # Compute artifact info
        self.size_bytes = self._get_size()
        self.hash = self._compute_hash()
        self.created_at = datetime.now().isoformat()
        
        # Store whether it's a file or directory
        self.is_directory = self.path.is_dir()
    
    def _get_size(self) -> int:
        """Get total size in bytes."""
        if not self.path.exists():
            return 0
        
        if self.path.is_file():
            return self.path.stat().st_size
        elif self.path.is_dir():
            return sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        
        return 0
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of the artifact."""
        if not self.path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        
        if self.path.is_file():
            with open(self.path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        elif self.path.is_dir():
            # Hash all files in directory
            for file_path in sorted(self.path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'type': self.artifact_type,
            'description': self.description,
            'size_bytes': self.size_bytes,
            'size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'hash': self.hash,
            'is_directory': self.is_directory,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        return f"Artifact(name='{self.name}', type='{self.artifact_type}', size={size_mb:.2f}MB)"


class ArtifactManager:
    """
    Manages artifacts for experiment runs.
    
    Handles copying, organizing, and tracking artifacts with full metadata.
    Creates a structured artifact directory for each run.
    """
    
    def __init__(self, artifacts_dir: Union[str, Path]):
        """
        Initialize artifact manager.
        
        Args:
            artifacts_dir: Base directory for storing artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all artifacts
        self.artifacts: List[Artifact] = []
        
        # Artifact index file
        self.index_file = self.artifacts_dir / "artifacts_index.json"
    
    def add_artifact(
        self,
        source_path: Union[str, Path],
        artifact_type: str = ArtifactType.OTHER,
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        copy: bool = True
    ) -> Artifact:
        """
        Add an artifact to the manager.
        
        Args:
            source_path: Path to the artifact to add
            artifact_type: Type of artifact
            name: Optional custom name
            description: Optional description
            metadata: Optional custom metadata
            copy: If True, copy artifact to artifacts directory
        
        Returns:
            Artifact object
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact not found: {source_path}")
        
        # Determine destination
        if copy:
            # Create type-specific subdirectory
            type_dir = self.artifacts_dir / artifact_type
            type_dir.mkdir(exist_ok=True)
            
            # Copy to artifacts directory
            dest_path = type_dir / source_path.name
            
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
            elif source_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
        else:
            # Reference original location
            dest_path = source_path
        
        # Create artifact
        artifact = Artifact(
            path=dest_path,
            artifact_type=artifact_type,
            name=name,
            description=description,
            metadata=metadata
        )
        
        self.artifacts.append(artifact)
        self._save_index()
        
        return artifact
    
    def add_model(
        self,
        model_path: Union[str, Path],
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Add a model artifact."""
        return self.add_artifact(
            model_path,
            artifact_type=ArtifactType.MODEL,
            name=name,
            description=description,
            metadata=metadata
        )
    
    def add_dataset(
        self,
        dataset_path: Union[str, Path],
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Add a dataset artifact."""
        return self.add_artifact(
            dataset_path,
            artifact_type=ArtifactType.DATASET,
            name=name,
            description=description,
            metadata=metadata
        )
    
    def add_plot(
        self,
        plot_path: Union[str, Path],
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Add a plot/visualization artifact."""
        return self.add_artifact(
            plot_path,
            artifact_type=ArtifactType.PLOT,
            name=name,
            description=description,
            metadata=metadata
        )
    
    def add_log(
        self,
        log_path: Union[str, Path],
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Add a log file artifact."""
        return self.add_artifact(
            log_path,
            artifact_type=ArtifactType.LOG,
            name=name,
            description=description,
            metadata=metadata
        )
    
    def add_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: Optional[int] = None,
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Add a training checkpoint artifact."""
        if metadata is None:
            metadata = {}
        
        if epoch is not None:
            metadata['epoch'] = epoch
        
        return self.add_artifact(
            checkpoint_path,
            artifact_type=ArtifactType.CHECKPOINT,
            name=name,
            description=description,
            metadata=metadata
        )
    
    def get_artifacts_by_type(self, artifact_type: str) -> List[Artifact]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts if a.artifact_type == artifact_type]
    
    def get_artifact_by_name(self, name: str) -> Optional[Artifact]:
        """Get artifact by name."""
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None
    
    def get_total_size(self) -> int:
        """Get total size of all artifacts in bytes."""
        return sum(a.size_bytes for a in self.artifacts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all artifacts."""
        total_size = self.get_total_size()
        
        # Count by type
        type_counts = {}
        for artifact in self.artifacts:
            type_counts[artifact.artifact_type] = type_counts.get(artifact.artifact_type, 0) + 1
        
        return {
            'total_artifacts': len(self.artifacts),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'artifacts_by_type': type_counts,
            'artifacts': [a.to_dict() for a in self.artifacts]
        }
    
    def _save_index(self) -> None:
        """Save artifact index to JSON file."""
        summary = self.get_summary()
        
        with open(self.index_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_index(self) -> Dict[str, Any]:
        """Load artifact index from JSON file."""
        if not self.index_file.exists():
            return {'total_artifacts': 0, 'artifacts': []}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def export_metadata(self, output_path: Union[str, Path]) -> None:
        """Export artifact metadata to JSON file."""
        output_path = Path(output_path)
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def __len__(self) -> int:
        """Return number of artifacts."""
        return len(self.artifacts)
    
    def __repr__(self) -> str:
        size_mb = self.get_total_size() / (1024 * 1024)
        return f"ArtifactManager(artifacts={len(self.artifacts)}, total_size={size_mb:.2f}MB)"