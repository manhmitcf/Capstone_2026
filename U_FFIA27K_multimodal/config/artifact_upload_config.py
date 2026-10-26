import json
import logging
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ArtifactUploadConfig(BaseModel):
    """
    Configuration for uploading the completed training artifact to Hugging Face.
    """
    enabled: bool = Field(
        default=True,
        description="Whether to zip and upload the project after training completes successfully."
    )
    source_dir: str = Field(
        default="/marimo/Capstone_2026_Fish_Feeding_Intensity",
        description="Directory to zip after training completes."
    )
    zip_path: str = Field(
        default="/marimo/Results_final_video_tie_breakers.zip",
        description="Path of the zip file to create."
    )
    repo_id: str = Field(
        default="manhmitcf/Results_final_video_tie_breakers",
        description="Hugging Face repository ID, for example 'username/repo-name'."
    )
    repo_type: str = Field(
        default="dataset",
        description="Hugging Face repository type: 'dataset', 'model', or 'space'."
    )
    path_in_repo: str = Field(
        default="Results_final_video_tie_breakers.zip",
        description="Destination file path inside the Hugging Face repository."
    )
    create_repo: bool = Field(
        default=True,
        description="Create the Hugging Face repository if it does not exist."
    )

    @classmethod
    def from_json(cls, path: str = "config/artifact_upload_config.json") -> "ArtifactUploadConfig":
        from pathlib import Path
        config_path = Path(path)
        if not config_path.is_file():
            pkg_fallback = Path(__file__).resolve().parent / Path(path).name
            if pkg_fallback.is_file():
                config_path = pkg_fallback
            else:
                root_fallback = Path(__file__).resolve().parent.parent / path
                if root_fallback.is_file():
                    config_path = root_fallback
        logger.info(f"Loading artifact upload configuration from JSON: '{config_path}'")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
