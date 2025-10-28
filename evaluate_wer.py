"""
WER (Word Error Rate) Evaluation for L2-ARCTIC Dataset using werpy
Includes normalization for both hypothesis and ground truth
"""
import os
import json
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from werpy import wer
from tqdm import tqdm


def cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    CER = (Substitutions + Insertions + Deletions) / (Number of reference characters)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # Convert to lists of characters
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # Use dynamic programming to compute Levenshtein distance
    m, n = len(ref_chars), len(hyp_chars)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1   # Substitution
                )
    
    # CER = edit distance / number of reference characters
    edit_distance = dp[m][n]
    cer_score = edit_distance / len(ref_chars)
    
    return cer_score


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def normalize_text(text: str, config: Dict) -> str:
    """
    Normalize text for WER evaluation
    Applies to both hypothesis (test) and ground truth (reference)
    """
    if not config['evaluation']['normalize_text']:
        return text
    
    # Convert to lowercase
    if config['evaluation']['lowercase']:
        text = text.lower()
    
    # Remove punctuation
    if config['evaluation']['remove_punctuation']:
        # Remove all punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace and normalize spaces
    text = ' '.join(text.split())
    
    return text.strip()


def load_references_l2arctic(data_dir: str, transcript_subdir: str = "transcript") -> Dict[str, str]:
    """
    Load reference transcriptions for L2-ARCTIC dataset
    Structure: {speaker}/transcript/arctic_aXXXX.txt
    """
    references = {}
    data_path = Path(data_dir)
    
    # Get all speaker directories
    speaker_dirs = [d for d in data_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') 
                   and d.name not in ['l2arctic_audio']]
    
    for speaker_dir in speaker_dirs:
        transcript_dir = speaker_dir / transcript_subdir
        if not transcript_dir.exists():
            continue
        
        # Read all transcript files
        for txt_file in transcript_dir.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Create key: speaker/filename
            key = f"{speaker_dir.name}/{txt_file.stem}"
            references[key] = text
    
    return references


def match_transcription_to_reference(
    transcriptions: List[Dict],
    references: Dict[str, str]
) -> List[Tuple[str, str, str, str, str]]:
    """
    Match transcriptions with their reference texts
    Returns list of (audio_path, speaker, filename, hypothesis, reference) tuples
    """
    matched_pairs = []
    unmatched = []
    
    for trans in transcriptions:
        audio_path = trans['audio_path']
        speaker = trans['speaker']
        filename = trans['filename']
        hypothesis = trans['transcription']
        
        # Create key to match with reference
        key = f"{speaker}/{filename}"
        
        reference = references.get(key)
        
        if reference is not None:
            matched_pairs.append((audio_path, speaker, filename, hypothesis, reference))
        else:
            unmatched.append((speaker, filename))
    
    if unmatched:
        print(f"\nWarning: {len(unmatched)} audio files have no matching reference text")
        print("First few unmatched files:")
        for speaker, filename in unmatched[:5]:
            print(f"  - {speaker}/{filename}")
    
    return matched_pairs


def calculate_metrics_werpy(
    hypothesis: str,
    reference: str
) -> Dict[str, float]:
    """Calculate WER and CER metrics using werpy"""
    try:
        # werpy expects lists of words for WER
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        
        # Calculate WER (word error rate)
        wer_score = wer(ref_words, hyp_words)
        
        # Calculate CER (character error rate)
        cer_score = cer(reference, hypothesis)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        wer_score = 1.0
        cer_score = 1.0
    
    return {
        'wer': wer_score * 100,  # Convert to percentage
        'cer': cer_score * 100
    }


def evaluate_wer(
    transcriptions_file: str,
    config: Dict
) -> Dict:
    """Main evaluation function"""
    
    # Load transcriptions
    print(f"Loading transcriptions from: {transcriptions_file}")
    with open(transcriptions_file, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
    
    # Load references
    data_dir = config['dataset']['data_dir']
    transcript_subdir = config['dataset'].get('transcript_subdir', 'transcript')
    print(f"Loading references from: {data_dir}")
    references = load_references_l2arctic(data_dir, transcript_subdir)
    
    if not references:
        print("\nERROR: No reference transcriptions found!")
        print("Please ensure your L2-ARCTIC dataset has transcript files.")
        print(f"Expected location: {{speaker}}/{transcript_subdir}/arctic_aXXXX.txt")
        return None
    
    print(f"Found {len(references)} reference transcriptions")
    
    # Match transcriptions to references
    print("\nMatching transcriptions with references...")
    matched_pairs = match_transcription_to_reference(transcriptions, references)
    
    if not matched_pairs:
        print("ERROR: No matching pairs found!")
        return None
    
    print(f"Successfully matched {len(matched_pairs)} pairs")
    
    # Calculate metrics for each pair
    detailed_results = []
    all_hypotheses_words = []
    all_references_words = []
    all_hypotheses_chars = []
    all_references_chars = []
    
    print("\nCalculating WER metrics...")
    print("Normalizing both hypothesis (test) and ground truth (reference)...")
    
    for audio_path, speaker, filename, hypothesis, reference in tqdm(matched_pairs):
        # Normalize both texts
        norm_hypothesis = normalize_text(hypothesis, config)
        norm_reference = normalize_text(reference, config)
        
        # Calculate metrics for this pair
        metrics = calculate_metrics_werpy(norm_hypothesis, norm_reference)
        
        detailed_results.append({
            'audio_path': audio_path,
            'speaker': speaker,
            'filename': filename,
            'hypothesis_original': hypothesis,
            'reference_original': reference,
            'hypothesis_normalized': norm_hypothesis,
            'reference_normalized': norm_reference,
            'wer': metrics['wer'],
            'cer': metrics['cer']
        })
        
        # Collect for overall metrics
        all_hypotheses_words.extend(norm_hypothesis.split())
        all_references_words.extend(norm_reference.split())
        all_hypotheses_chars.append(norm_hypothesis)
        all_references_chars.append(norm_reference)
    
    # Calculate overall metrics
    print("\nCalculating overall metrics...")
    overall_wer = wer(all_references_words, all_hypotheses_words) * 100
    
    # For CER, concatenate all texts
    overall_cer = cer(''.join(all_references_chars), ''.join(all_hypotheses_chars)) * 100
    
    # Calculate per-speaker metrics
    df = pd.DataFrame(detailed_results)
    
    speaker_metrics = {}
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        
        # Collect all words and chars for this speaker
        speaker_hyp_words = []
        speaker_ref_words = []
        speaker_hyp_chars = []
        speaker_ref_chars = []
        
        for _, row in speaker_df.iterrows():
            speaker_hyp_words.extend(row['hypothesis_normalized'].split())
            speaker_ref_words.extend(row['reference_normalized'].split())
            speaker_hyp_chars.append(row['hypothesis_normalized'])
            speaker_ref_chars.append(row['reference_normalized'])
        
        # Calculate speaker-level metrics
        speaker_wer = wer(speaker_ref_words, speaker_hyp_words) * 100
        speaker_cer = cer(''.join(speaker_ref_chars), ''.join(speaker_hyp_chars)) * 100
        
        speaker_metrics[speaker] = {
            'wer': speaker_wer,
            'cer': speaker_cer,
            'count': len(speaker_df)
        }
    
    results = {
        'overall': {
            'wer': overall_wer,
            'cer': overall_cer,
            'num_samples': len(matched_pairs)
        },
        'per_speaker': speaker_metrics,
        'detailed': detailed_results,
        'normalization_applied': {
            'lowercase': config['evaluation']['lowercase'],
            'remove_punctuation': config['evaluation']['remove_punctuation']
        }
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS - L2-ARCTIC WER Evaluation")
    print("="*70)
    
    # Show normalization settings
    norm = results['normalization_applied']
    print("\nNormalization Settings:")
    print(f"  Lowercase: {norm['lowercase']}")
    print(f"  Remove Punctuation: {norm['remove_punctuation']}")
    
    overall = results['overall']
    print(f"\nOverall Metrics (n={overall['num_samples']}):")
    print(f"  Word Error Rate (WER): {overall['wer']:.2f}%")
    print(f"  Character Error Rate (CER): {overall['cer']:.2f}%")
    
    if results['per_speaker']:
        print("\n" + "-"*70)
        print("Per-Speaker Results:")
        print("-"*70)
        print(f"{'Speaker':<15} {'WER (%)':<12} {'CER (%)':<12} {'Samples':<10}")
        print("-"*70)
        
        # Sort by WER
        sorted_speakers = sorted(
            results['per_speaker'].items(),
            key=lambda x: x[1]['wer']
        )
        
        for speaker, metrics in sorted_speakers:
            print(f"{speaker:<15} {metrics['wer']:>10.2f}  {metrics['cer']:>10.2f}  {metrics['count']:>9}")
        
        print("-"*70)
        
        # Calculate average WER across speakers
        avg_wer = sum(m['wer'] for m in results['per_speaker'].values()) / len(results['per_speaker'])
        avg_cer = sum(m['cer'] for m in results['per_speaker'].values()) / len(results['per_speaker'])
        print(f"{'Average':<15} {avg_wer:>10.2f}  {avg_cer:>10.2f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER for L2-ARCTIC transcriptions using werpy")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--transcriptions",
        type=str,
        default=None,
        help="Path to transcriptions JSON file (default: outputs/transcriptions.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (default: outputs/evaluation_results.json)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set default paths
    transcriptions_file = args.transcriptions or os.path.join(
        config['dataset']['output_dir'],
        'transcriptions.json'
    )
    
    output_file = args.output or os.path.join(
        config['dataset']['output_dir'],
        'evaluation_results.json'
    )
    
    # Check if transcriptions file exists
    if not os.path.exists(transcriptions_file):
        print(f"ERROR: Transcriptions file not found: {transcriptions_file}")
        print("Please run transcribe.py first to generate transcriptions.")
        return
    
    # Run evaluation
    results = evaluate_wer(transcriptions_file, config)
    
    if results is None:
        print("\nEvaluation failed!")
        return
    
    # Print results
    print_results(results)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary to CSV
    csv_file = Path(output_file).with_suffix('.csv')
    df = pd.DataFrame(results['detailed'])
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Detailed results CSV saved to: {csv_file}")
    
    # Save per-speaker summary
    speaker_csv = Path(output_file).parent / "per_speaker_results.csv"
    speaker_df = pd.DataFrame.from_dict(results['per_speaker'], orient='index')
    speaker_df.index.name = 'speaker'
    speaker_df.to_csv(speaker_csv, encoding='utf-8')
    print(f"Per-speaker summary saved to: {speaker_csv}")


if __name__ == "__main__":
    main()
