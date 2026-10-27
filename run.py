"""
ZentroMart VoiceOps 2026 - Batch Execution & Evaluation Pipeline.

Processes a folder of audio/text queries or evaluation datasets (practice/questions.csv,
practice/conversations.csv), executes the agent brain, and saves:
- Output transcripts
- Audio replies synthesized via Gnani TTS (model='timbre-v2.5', voice='Kaveri')
- Retrieved policy sources and decision metrics
into sample_outputs/
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.agent import VoiceOpsAgent
from src.db_loader import DatabaseLoader
from src.kb_loader import KnowledgeBaseLoader
from src.voice import VoiceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceOpsRunner")
console = Console()


def process_audio_directory(
    audio_dir: Path,
    agent: VoiceOpsAgent,
    voice_service: VoiceService,
    output_dir: Path,
    generate_tts: bool = True,
) -> List[Dict[str, Any]]:
    """Process all audio files in a directory through STT -> Agent -> TTS."""
    results = []
    audio_extensions = {".wav", ".mp3", ".m4a", ".ogg"}
    audio_files = sorted([f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions])

    logger.info(f"Found {len(audio_files)} audio file(s) in {audio_dir}")
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, audio_file in enumerate(audio_files, 1):
        item_id = audio_file.stem
        logger.info(f"[{idx}/{len(audio_files)}] Processing audio: {audio_file.name}")

        # 1. Transcribe with Gnani STT
        transcript = voice_service.speech_to_text(audio_file, language_code="en-IN")
        if not transcript:
            transcript = audio_file.stem.replace("_", " ")

        # 2. Agent Execution
        conv_id = f"audio_batch_{item_id}"
        agent_out = agent.process_turn(transcript, conversation_id=conv_id)

        # 3. Synthesize reply with Gnani TTS
        audio_reply_path = None
        if generate_tts and agent_out["reply"]:
            audio_reply_file = audio_out_dir / f"{item_id}_reply.wav"
            voice_service.text_to_speech(
                text=agent_out["reply"],
                voice="Kaveri",
                model="timbre-v2.5",
                output_file=audio_reply_file,
            )
            audio_reply_path = str(audio_reply_file)

        results.append({
            "id": item_id,
            "type": "audio",
            "source_file": str(audio_file),
            "transcript": transcript,
            "decision": agent_out["decision"],
            "reply": agent_out["reply"],
            "tool_calls": agent_out["tool_calls"],
            "retrieved_evidence": agent_out["evidence"],
            "audio_reply_file": audio_reply_path or "",
        })

    return results


def process_questions_dataset(
    csv_path: Path,
    agent: VoiceOpsAgent,
    voice_service: VoiceService,
    output_dir: Path,
    generate_tts_sample_limit: int = 5,
) -> List[Dict[str, Any]]:
    """Process questions from practice/questions.csv."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    results = []
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(df)} questions from {csv_path.name}...")

    for idx, row in df.iterrows():
        qid = row.get("id", f"Q{idx}")
        # Use asr_text if available to simulate realistic voice speech input
        prompt = row.get("asr_text") or row.get("question", "")

        conv_id = f"single_turn_{qid}"
        agent_out = agent.process_turn(prompt, conversation_id=conv_id)

        # Generate sample TTS audio replies for the first few questions
        audio_reply_path = ""
        if idx < generate_tts_sample_limit and agent_out["reply"]:
            audio_reply_file = audio_out_dir / f"{qid}_reply.wav"
            voice_service.text_to_speech(
                text=agent_out["reply"],
                voice="Kaveri",
                model="timbre-v2.5",
                output_file=audio_reply_file,
            )
            audio_reply_path = str(audio_reply_file)

        results.append({
            "id": qid,
            "type": "question",
            "input_text": row.get("question", ""),
            "asr_input": row.get("asr_text", ""),
            "decision": agent_out["decision"],
            "expected_decision": row.get("expected_decision", ""),
            "reply": agent_out["reply"],
            "reference_answer": row.get("reference_answer", ""),
            "tool_calls": "; ".join(agent_out["tool_calls"]),
            "retrieved_evidence": "; ".join(agent_out["evidence"]),
            "gold_evidence": row.get("gold_evidence", ""),
            "audio_reply_file": audio_reply_path,
        })

    return results


def process_conversations_dataset(
    csv_path: Path,
    agent: VoiceOpsAgent,
    voice_service: VoiceService,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Process multi-turn conversations from practice/conversations.csv."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    results = []

    logger.info(f"Processing multi-turn conversations from {csv_path.name}...")

    # Group by conversation_id ordered by turn
    for cid, group in df.groupby("conversation_id"):
        agent.reset_session(cid)
        sorted_turns = group.sort_values(by="turn", key=lambda col: col.astype(int))

        for _, row in sorted_turns.iterrows():
            turn_no = row.get("turn", "1")
            user_msg = row.get("customer_message", "")

            agent_out = agent.process_turn(user_msg, conversation_id=cid)

            results.append({
                "conversation_id": cid,
                "turn": turn_no,
                "scenario": row.get("scenario", ""),
                "customer_message": user_msg,
                "decision": agent_out["decision"],
                "expected_decision": row.get("expected_decision", ""),
                "agent_reply": agent_out["reply"],
                "tool_calls": "; ".join(agent_out["tool_calls"]),
                "expected_tools": row.get("expected_tool_calls", ""),
                "retrieved_evidence": "; ".join(agent_out["evidence"]),
                "gold_evidence": row.get("gold_evidence", ""),
            })

    return results


def main() -> None:
    """Main CLI entry point for batch processing."""
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps Batch Runner")
    parser.add_argument("--input-dir", type=str, help="Directory containing audio or text query files")
    parser.add_argument("--input-file", type=str, help="Single CSV or text file to process")
    parser.add_argument("--output-dir", type=str, default="sample_outputs", help="Directory to store outputs")
    parser.add_argument("--tts-limit", type=int, default=5, help="Number of audio replies to synthesize via Gnani TTS")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS synthesis")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    db_loader = DatabaseLoader(data_dir=project_root / "data")
    kb_loader = KnowledgeBaseLoader(kb_dir=project_root / "kb")
    voice_service = VoiceService()
    agent = VoiceOpsAgent(db_loader=db_loader, kb_loader=kb_loader)

    all_results: Dict[str, Any] = {}

    # Case 1: Custom audio or text input directory
    if args.input_dir:
        in_dir = Path(args.input_dir)
        if in_dir.exists() and in_dir.is_dir():
            logger.info(f"Running batch on directory: {in_dir}")
            audio_results = process_audio_directory(
                audio_dir=in_dir,
                agent=agent,
                voice_service=voice_service,
                output_dir=output_dir,
                generate_tts=not args.no_tts,
            )
            all_results["audio_queries"] = audio_results

    # Case 2: Custom single input file
    elif args.input_file:
        in_file = Path(args.input_file)
        if in_file.exists():
            if "question" in in_file.name.lower():
                q_res = process_questions_dataset(
                    in_file, agent, voice_service, output_dir,
                    generate_tts_sample_limit=0 if args.no_tts else args.tts_limit,
                )
                all_results["questions"] = q_res
            elif "conversation" in in_file.name.lower():
                c_res = process_conversations_dataset(in_file, agent, voice_service, output_dir)
                all_results["conversations"] = c_res
            else:
                logger.warning(f"Unrecognized CSV schema for {in_file.name}")

    # Case 3: Default evaluation run on practice/ datasets
    else:
        logger.info("No custom inputs specified. Executing evaluation on practice/ datasets...")
        questions_file = project_root / "practice" / "questions.csv"
        conversations_file = project_root / "practice" / "conversations.csv"

        if questions_file.exists():
            q_res = process_questions_dataset(
                questions_file, agent, voice_service, output_dir,
                generate_tts_sample_limit=0 if args.no_tts else args.tts_limit,
            )
            all_results["questions"] = q_res

        if conversations_file.exists():
            c_res = process_conversations_dataset(conversations_file, agent, voice_service, output_dir)
            all_results["conversations"] = c_res

    # Save Results to sample_outputs/
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed execution records saved to {json_path}")

    # Save summary tables to CSV
    if "questions" in all_results:
        q_df = pd.DataFrame(all_results["questions"])
        q_csv = output_dir / "questions_evaluation_summary.csv"
        q_df.to_csv(q_csv, index=False, encoding="utf-8")
        logger.info(f"Questions summary saved to {q_csv}")

    if "conversations" in all_results:
        c_df = pd.DataFrame(all_results["conversations"])
        c_csv = output_dir / "conversations_evaluation_summary.csv"
        c_df.to_csv(c_csv, index=False, encoding="utf-8")
        logger.info(f"Conversations summary saved to {c_csv}")

    # Print summary table to console
    table = Table(title="ZentroMart VoiceOps Execution Summary")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Total Evaluated", style="magenta")
    table.add_column("Outputs Saved", style="green")

    if "questions" in all_results:
        table.add_row("Single Questions", str(len(all_results["questions"])), str(output_dir / "questions_evaluation_summary.csv"))
    if "conversations" in all_results:
        table.add_row("Conversation Turns", str(len(all_results["conversations"])), str(output_dir / "conversations_evaluation_summary.csv"))
    if "audio_queries" in all_results:
        table.add_row("Audio Queries", str(len(all_results["audio_queries"])), str(output_dir / "audio"))

    console.print(table)


if __name__ == "__main__":
    main()
