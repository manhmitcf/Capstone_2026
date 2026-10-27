import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def discover_hf_token() -> Optional[str]:
    """
    Search for a Hugging Face authentication token across environment variables,
    dedicated token files, and fallback configuration text files.
    """
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip().lstrip('\ufeff')

    dedicated_files = [
        Path("token.txt"),
        Path("../token.txt"),
        Path("/marimo/token.txt"),
        Path(__file__).resolve().parent.parent / "token.txt",
        Path(__file__).resolve().parent.parent.parent / "token.txt",
    ]
    for dtf in dedicated_files:
        if dtf.is_file():
            try:
                content = dtf.read_text(encoding="utf-8-sig").strip().lstrip('\ufeff')
                match = re.search(r'(hf_[A-Za-z0-9_]+)', content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass

    try:
        from huggingface_hub import get_token
        hub_token = get_token()
        if hub_token:
            return hub_token.strip().lstrip('\ufeff')
    except Exception:
        pass

    return None


def upload_audio_case_to_hf(
    case_name: str,
    case_dir: str,
    repo_id: str = "manhmitcf/audio_depth_tkeo",
    branch_name: str = "main",
    repo_type: str = "dataset",
) -> bool:
    """
    Upload an individual ablation run directory to Hugging Face on the specified branch.

    Args:
        case_name: Name of the ablation case (e.g. 'audio_d1_notkeo').
        case_dir: Local folder path containing the case outputs.
        repo_id: Hugging Face repository ID.
        branch_name: Target git branch on Hugging Face (default: 'ablations/audio_depth_tkeo').
        repo_type: 'dataset' or 'model'.

    Returns:
        bool: True if uploaded successfully, False otherwise.
    """
    token = discover_hf_token()
    if not token:
        logger.warning(f"[{case_name}] HF_TOKEN not found. Skipping Hugging Face upload.")
        return False

    case_path = Path(case_dir)
    if not case_path.is_dir():
        logger.warning(f"[{case_name}] Directory '{case_dir}' does not exist. Skipping upload.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # 1. Ensure target repository exists
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        except Exception as exc:
            logger.debug(f"Repo check: {exc}")

        # 2. Ensure target branch exists on Hugging Face
        try:
            api.create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type, exist_ok=True)
            logger.info(f"[{case_name}] Verified branch '{branch_name}' on HF repo '{repo_id}'.")
        except Exception as exc:
            logger.debug(f"Branch verification: {exc}")

        # 3. Upload folder to audio_ablations/<case_name>/
        path_in_repo = f"audio_ablations/{case_name}"
        logger.info(f"[{case_name}] Uploading output files to HF branch '{branch_name}' -> '{path_in_repo}'...")

        api.upload_folder(
            folder_path=str(case_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=branch_name,
            commit_message=f"Upload audio ablation results for {case_name}"
        )

        logger.info(f"[{case_name}] Successfully uploaded to Hugging Face branch '{branch_name}'!")
        return True

    except ImportError:
        logger.warning("huggingface_hub is not installed. Install via `pip install huggingface_hub`.")
        return False
    except Exception as exc:
        logger.warning(f"[{case_name}] Hugging Face upload failed: {exc}. Training will continue unaffected.")
        return False


def upload_summary_file_to_hf(
    file_path: str,
    repo_id: str = "manhmitcf/audio_depth_tkeo",
    branch_name: str = "main",
    repo_type: str = "dataset",
) -> bool:
    """Upload consolidated summary table to the root of the ablation branch."""
    token = discover_hf_token()
    if not token:
        return False

    local_file = Path(file_path)
    if not local_file.is_file():
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        try:
            api.create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type, exist_ok=True)
        except Exception:
            pass

        path_in_repo = f"audio_ablations/{local_file.name}"
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=branch_name,
            commit_message=f"Update consolidated audio ablation summary ({local_file.name})"
        )
        logger.info(f"Uploaded summary '{local_file.name}' to HF branch '{branch_name}'.")
        return True
    except Exception as exc:
        logger.warning(f"Could not upload summary file to HF: {exc}")
        return False
