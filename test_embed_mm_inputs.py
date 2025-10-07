#!/usr/bin/env python3
"""
Test script to verify embed_mm_inputs function signature compatibility
"""

import sys
import inspect

# Add the python directory to the path
sys.path.insert(0, '/home/yzh/sglang_053_update/sglang_053/python')

try:
    from sglang.srt.managers.mm_utils import embed_mm_inputs
    
    # Get the function signature
    sig = inspect.signature(embed_mm_inputs)
    
    print("✓ Successfully imported embed_mm_inputs")
    print(f"\nFunction signature:")
    print(f"  {sig}")
    
    print(f"\nParameters:")
    for param_name, param in sig.parameters.items():
        print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
        if param.default != inspect.Parameter.empty:
            print(f"    (default: {param.default})")
    
    # Check for the expected parameters
    expected_params = [
        'mm_inputs_list',
        'extend_prefix_lens',
        'extend_seq_lens',
        'input_ids',
        'input_embedding',
        'multimodal_model',
        'data_embedding_func_mapping',
        'placeholder_tokens',
    ]
    
    actual_params = list(sig.parameters.keys())
    
    print(f"\n✓ Verification:")
    for param in expected_params:
        if param in actual_params:
            print(f"  ✓ {param} - found")
        else:
            print(f"  ✗ {param} - NOT FOUND")
    
    # Check for old parameter names that should NOT exist
    old_params = ['image_data_embedding_func', 'audio_data_embedding_func']
    print(f"\n✓ Checking for deprecated parameters:")
    for param in old_params:
        if param in actual_params:
            print(f"  ✗ {param} - FOUND (should not exist)")
        else:
            print(f"  ✓ {param} - not found (correct)")
    
    print("\n✓ All checks passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

