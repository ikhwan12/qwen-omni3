"""
Helper script to prepare and validate L2-ARCTIC v5.0 dataset structure
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List
import json


def scan_l2arctic_dataset(data_dir: str) -> Dict:
    """Scan and analyze L2-ARCTIC v5.0 dataset structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return {
            'exists': False,
            'error': f"Directory {data_dir} does not exist"
        }
    
    # Get speaker directories
    speaker_dirs = [d for d in data_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') 
                   and d.name not in ['l2arctic_audio']]
    
    speakers = []
    speaker_stats = {}
    
    for speaker_dir in speaker_dirs:
        speaker = speaker_dir.name
        speakers.append(speaker)
        
        # Count files in each subdirectory
        wav_dir = speaker_dir / "wav"
        transcript_dir = speaker_dir / "transcript"
        
        wav_count = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
        txt_count = len(list(transcript_dir.glob("*.txt"))) if transcript_dir.exists() else 0
        
        speaker_stats[speaker] = {
            'wav_files': wav_count,
            'transcript_files': txt_count,
            'has_wav_dir': wav_dir.exists(),
            'has_transcript_dir': transcript_dir.exists()
        }
    
    total_wav = sum(s['wav_files'] for s in speaker_stats.values())
    total_txt = sum(s['transcript_files'] for s in speaker_stats.values())
    
    return {
        'exists': True,
        'num_speakers': len(speakers),
        'speakers': sorted(speakers),
        'total_wav_files': total_wav,
        'total_transcript_files': total_txt,
        'speaker_stats': speaker_stats
    }


def validate_l2arctic_dataset(data_dir: str) -> List[str]:
    """Validate L2-ARCTIC dataset and return list of issues"""
    issues = []
    info = scan_l2arctic_dataset(data_dir)
    
    if not info['exists']:
        issues.append(f"❌ {info['error']}")
        return issues
    
    print("📊 L2-ARCTIC Dataset Statistics:")
    print(f"  • Speakers: {info['num_speakers']}")
    print(f"  • Total WAV files: {info['total_wav_files']}")
    print(f"  • Total transcript files: {info['total_transcript_files']}")
    
    if info['num_speakers'] > 0:
        print(f"\n👥 Speakers: {', '.join(info['speakers'])}")
    
    # Detailed per-speaker stats
    print("\n📁 Per-Speaker Statistics:")
    print(f"{'Speaker':<15} {'WAV Files':<12} {'Transcripts':<12} {'Status'}")
    print("-" * 60)
    
    for speaker in sorted(info['speakers']):
        stats = info['speaker_stats'][speaker]
        status = "✓" if stats['wav_files'] == stats['transcript_files'] else "⚠️"
        print(f"{speaker:<15} {stats['wav_files']:<12} {stats['transcript_files']:<12} {status}")
        
        # Check for issues
        if not stats['has_wav_dir']:
            issues.append(f"❌ {speaker}: Missing 'wav' directory")
        if not stats['has_transcript_dir']:
            issues.append(f"❌ {speaker}: Missing 'transcript' directory")
        if stats['wav_files'] == 0:
            issues.append(f"⚠️  {speaker}: No WAV files found")
        if stats['transcript_files'] == 0:
            issues.append(f"⚠️  {speaker}: No transcript files found")
        if stats['wav_files'] != stats['transcript_files']:
            issues.append(f"⚠️  {speaker}: Mismatch - {stats['wav_files']} WAV but {stats['transcript_files']} transcripts")
    
    # Overall validation
    if info['num_speakers'] == 0:
        issues.append("❌ No speaker directories found")
    
    if info['total_wav_files'] == 0:
        issues.append("❌ No audio files (.wav) found in dataset")
    
    if info['total_transcript_files'] == 0:
        issues.append("⚠️  No reference transcript files (.txt) found - WER evaluation will not work")
    
    return issues


def check_sample_files(data_dir: str, speaker: str = "ABA", num_samples: int = 3):
    """Check sample files to verify content"""
    data_path = Path(data_dir)
    speaker_dir = data_path / speaker
    
    if not speaker_dir.exists():
        print(f"\nSpeaker directory not found: {speaker}")
        return
    
    print(f"\n📄 Sample Files from Speaker '{speaker}':")
    print("-" * 60)
    
    transcript_dir = speaker_dir / "transcript"
    wav_dir = speaker_dir / "wav"
    
    for i in range(1, num_samples + 1):
        filename = f"arctic_a{i:04d}"
        txt_file = transcript_dir / f"{filename}.txt"
        wav_file = wav_dir / f"{filename}.wav"
        
        print(f"\n{filename}:")
        
        if wav_file.exists():
            size = wav_file.stat().st_size / 1024  # KB
            print(f"  WAV: ✓ ({size:.1f} KB)")
        else:
            print(f"  WAV: ✗ Not found")
        
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            print(f"  TXT: ✓")
            print(f"  Content: {content[:100]}{'...' if len(content) > 100 else ''}")
        else:
            print(f"  TXT: ✗ Not found")


def export_dataset_info(data_dir: str, output_file: str = "dataset_info.json"):
    """Export dataset information to JSON"""
    info = scan_l2arctic_dataset(data_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset information exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate L2-ARCTIC v5.0 dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="E:/Dataset/LNV/L2-ARCTIC/l2arctic_release_v5.0",
        help="Path to L2-ARCTIC dataset directory"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset structure"
    )
    parser.add_argument(
        "--check_samples",
        action="store_true",
        help="Check sample files"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="ABA",
        help="Speaker to check samples for (default: ABA)"
    )
    parser.add_argument(
        "--export_info",
        type=str,
        default=None,
        help="Export dataset info to JSON file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("L2-ARCTIC v5.0 Dataset Preparation Tool")
    print("=" * 60)
    print()
    
    # Default: validate the dataset
    if not args.check_samples and not args.export_info:
        args.validate = True
    
    if args.validate:
        issues = validate_l2arctic_dataset(args.data_dir)
        
        print("\n" + "=" * 60)
        if issues:
            print("⚠️  Validation Issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✓ Dataset validation passed!")
        print("=" * 60)
    
    if args.check_samples:
        check_sample_files(args.data_dir, args.speaker)
    
    if args.export_info:
        export_dataset_info(args.data_dir, args.export_info)


if __name__ == "__main__":
    main()
