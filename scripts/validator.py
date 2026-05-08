import pandas as pd
import os
from typing import Tuple

def check_basic_rules(sequence: str) -> Tuple[bool, str]:
    """
    Check if the amino acid sequence meets basic requirements.
    
    Args:
        sequence (str): Amino acid sequence to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    # Check if sequence starts with 'M'
    if not sequence.startswith('M'):
        return False, f"Sequence must start with 'M', found: '{sequence[0]}'"
    
    # Check sequence length
    if len(sequence) < 220 or len(sequence) > 250:
        return False, f"Sequence length must be between 220 and 250, found: {len(sequence)}"
    
    # Check for valid amino acids
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    for i, aa in enumerate(sequence):
        if aa not in valid_amino_acids:
            return False, f"Invalid amino acid '{aa}' at position {i+1}"
    
    # Check for stop codon
    if '*' in sequence:
        return False, "Sequence contains stop codon ('*')"
    
    return True, "Valid"

def check_exclusion_list(sequence: str, csv_path: str = None) -> Tuple[bool, str]:
    """
    Check if the sequence is in the exclusion list.
    
    Args:
        sequence (str): Amino acid sequence to check
        csv_path (str, optional): Path to exclusion list CSV. Defaults to None.
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    if csv_path is None:
        # Use the exclusion list in the project root
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Exclusion_List.csv')
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 'sequence' column exists
        if 'sequence' not in df.columns:
            return False, f"'sequence' column not found in exclusion list CSV: {csv_path}"
        
        # Check if sequence exists in exclusion list
        if sequence in df['sequence'].values:
            return False, "Sequence found in exclusion list"
            
        return True, "Valid"
    except Exception as e:
        return False, f"Error checking exclusion list: {str(e)}"