"""
Packages the ZentroMart VoiceOps final submission into team_name_voiceops.zip
matching the exact Capstone specification:

team_name_voiceops.zip
├── README.md
├── requirements.txt
├── run.py
├── src/
├── report.pdf
├── demo.mp4
└── sample_outputs/
"""
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP_NAME = ROOT / "team_name_voiceops.zip"


def create_demo_placeholder() -> Path:
    """Create demo.mp4 placeholder if not already present."""
    demo_file = ROOT / "demo.mp4"
    if not demo_file.exists():
        content = (
            "ZentroMart VoiceOps 2026 - Capstone Demo Video Placeholder\n"
            "Replace this file with your 2-minute video walkthrough (MP4 format) "
            "demonstrating Gnani STT voice transcription, agent reasoning, "
            "and Gnani TTS speech synthesis."
        )
        demo_file.write_text(content, encoding="utf-8")
        print("Created demo.mp4 placeholder.")
    return demo_file


def package_zip() -> None:
    create_demo_placeholder()

    if ZIP_NAME.exists():
        ZIP_NAME.unlink()

    included_entries = [
        "README.md",
        "requirements.txt",
        "run.py",
        "report.pdf",
        "demo.mp4",
        "src",
        "sample_outputs",
    ]

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for entry_name in included_entries:
            entry_path = ROOT / entry_name
            if not entry_path.exists():
                print(f"Warning: {entry_name} not found!")
                continue

            if entry_path.is_file():
                zipf.write(entry_path, arcname=entry_name)
                print(f"Added file: {entry_name}")
            elif entry_path.is_dir():
                for root, dirs, files in os.walk(entry_path):
                    # Filter out __pycache__ and hidden files
                    dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                    for file in files:
                        if file.endswith((".pyc", ".pyo")) or file.startswith("."):
                            continue
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(ROOT)
                        zipf.write(full_path, arcname=str(rel_path).replace("\\", "/"))
                        print(f"Added file: {rel_path}")

    print(f"\nSuccessfully created submission archive: {ZIP_NAME.name} ({ZIP_NAME.stat().st_size:,} bytes)")


def verify_zip() -> None:
    print(f"\nVerifying contents of {ZIP_NAME.name}:")
    with zipfile.ZipFile(ZIP_NAME, "r") as zipf:
        infolist = zipf.infolist()
        for info in infolist:
            print(f"  {info.filename} ({info.file_size:,} bytes)")
    print(f"Total files in archive: {len(infolist)}")


if __name__ == "__main__":
    package_zip()
    verify_zip()
