#!/usr/bin/env python3
"""
Quick fix for the typing error in main.py
"""
import os

def fix_typing_error():
    """Fix the Dict typing error in main.py"""
    print("🔧 Fixing typing error in main.py")
    print("=" * 60)
    
    if not os.path.exists('main.py'):
        print("❌ main.py not found!")
        return
    
    # Read current file
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Check if already has typing import
    if 'from typing import' in content:
        print("✅ Already has typing imports")
        return
    
    # Fix by adding the import
    lines = content.split('\n')
    
    # Find where to insert (after other imports)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('from datetime import'):
            insert_pos = i + 1
            break
    
    # Insert the typing import
    lines.insert(insert_pos, 'from typing import Dict, Optional')
    
    # Write back
    with open('main.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Fixed typing import in main.py")
    print("\nYou can now run:")
    print("  python main.py")
    
    # Also check if they need the other files
    print("\n🔍 Checking other files...")
    
    issues = []
    
    # Check llm_engine
    if os.path.exists('core/llm_engine.py'):
        with open('core/llm_engine.py', 'r') as f:
            content = f.read()
        if 'Overbought with volume' in content:
            issues.append("core/llm_engine.py - Still using OLD version")
    else:
        issues.append("core/llm_engine.py - File missing")
    
    # Check bot_engine
    if os.path.exists('core/bot_engine.py'):
        with open('core/bot_engine.py', 'r') as f:
            content = f.read()
        if 'stop_loss_pct' not in content:
            issues.append("core/bot_engine.py - Still using OLD version")
    else:
        issues.append("core/bot_engine.py - File missing")
    
    if issues:
        print("\n⚠️  Other issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n📋 To fix completely:")
        print("1. Copy 'Fixed LLM Engine' artifact → core/llm_engine.py")
        print("2. Copy 'Fixed Bot Engine' artifact → core/bot_engine.py")
    else:
        print("\n✅ All files look good!")

if __name__ == "__main__":
    fix_typing_error()