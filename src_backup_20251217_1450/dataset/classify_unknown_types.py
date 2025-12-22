#!/usr/bin/env python3
"""
自动分类 'unknown' 类型的 SPC 对
使用增强的模式匹配
"""

import json
import re
from pathlib import Path

def classify_unknown_pairs(input_file, output_file):
    """使用增强的规则对 unknown 类型进行分类"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    print("="*70)
    print("🔍 Classifying Unknown Vulnerability Types")
    print("="*70)
    
    unknown_pairs = [p for p in pairs if p.get('vulnerability_type') == 'unknown']
    print(f"\nFound {len(unknown_pairs)} unknown pairs")
    
    # 增强的分类规则
    classification_rules = {
        'reentrancy': [
            (r'\.call\{value:', 'external call with value'),
            (r'\.call\.value\(', 'old-style external call'),
            (r'msg\.sender\.call', 'call to msg.sender'),
            (r'\.transfer\(', 'transfer function'),
            (r'\.send\(', 'send function'),
            (r'balance\[.*\]\s*-=.*\n.*\.call', 'balance update after call'),
            (r'ReentrancyGuard', 'uses ReentrancyGuard'),
            (r'nonReentrant', 'uses nonReentrant modifier'),
        ],
        'overflow': [
            (r'SafeMath', 'uses SafeMath'),
            (r'\.add\(', 'SafeMath add'),
            (r'\.sub\(', 'SafeMath sub'),
            (r'\.mul\(', 'SafeMath mul'),
            (r'\buint\d*\s+\w+\s*\+=', 'uint addition'),
            (r'\buint\d*\s+\w+\s*\*=', 'uint multiplication'),
            (r'pragma solidity \^0\.8', 'Solidity 0.8+ (built-in overflow check)'),
        ],
        'access_control': [
            (r'onlyOwner', 'uses onlyOwner modifier'),
            (r'Ownable', 'inherits Ownable'),
            (r'require\(msg\.sender\s*==\s*owner', 'checks owner'),
            (r'function.*destroy|kill|selfdestruct', 'has selfdestruct'),
            (r'AccessControl', 'uses AccessControl'),
        ],
        'tx_origin': [
            (r'tx\.origin', 'uses tx.origin'),
            (r'require\(tx\.origin', 'checks tx.origin'),
        ],
        'unchecked_call': [
            (r'\(bool\s+\w+,\s*\)\s*=.*\.call', 'checks call return value'),
            (r'require\(.*\.call', 'requires call success'),
            (r'\.delegatecall\(', 'uses delegatecall'),
        ],
        'timestamp': [
            (r'block\.timestamp', 'uses block.timestamp'),
            (r'\bnow\b', 'uses now keyword'),
            (r'block\.number', 'uses block.number'),
        ],
        'dos': [
            (r'for\s*\(.*;\s*\w+\s*<\s*\w+\.length', 'loops over array'),
            (r'while\s*\(', 'uses while loop'),
        ],
    }
    
    classified = 0
    classification_reasons = {}
    
    for pair in unknown_pairs:
        code_before = pair.get('code_before', '')
        code_after = pair.get('code_after', '')
        combined_code = code_before + '\n' + code_after
        
        # 尝试分类
        scores = {}
        reasons = {}
        
        for vuln_type, rules in classification_rules.items():
            score = 0
            matched_reasons = []
            
            for pattern, reason in rules:
                if re.search(pattern, combined_code, re.IGNORECASE):
                    score += 1
                    matched_reasons.append(reason)
            
            if score > 0:
                scores[vuln_type] = score
                reasons[vuln_type] = matched_reasons
        
        # 选择得分最高的类型
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] >= 2:  # 至少匹配2个规则
                pair['vulnerability_type'] = best_type
                pair['_classification_reason'] = reasons[best_type]
                pair['_classification_confidence'] = scores[best_type]
                classified += 1
                classification_reasons[best_type] = classification_reasons.get(best_type, 0) + 1
    
    print(f"\n✅ Successfully classified: {classified}/{len(unknown_pairs)} ({classified/len(unknown_pairs)*100:.0f}%)")
    
    if classification_reasons:
        print(f"\n📊 New classifications:")
        for vtype, count in sorted(classification_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   {vtype:20s}: {count}")
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_file}")
    
    # 更新统计
    final_unknown = sum(1 for p in pairs if p['vulnerability_type'] == 'unknown')
    print(f"\nRemaining unknown: {final_unknown}/{len(pairs)} ({final_unknown/len(pairs)*100:.0f}%)")
    
    return pairs


def main():
    input_file = "data/spc_data/processed/bootstrap_spc_dataset.json"
    output_file = "data/spc_data/processed/bootstrap_classified.json"
    
    pairs = classify_unknown_pairs(input_file, output_file)
    
    print("\n" + "="*70)
    print("✅ Classification Complete!")
    print("="*70)
    print("\n💡 Tip: Review pairs with '_classification_reason' field")
    print("   These were auto-classified from 'unknown' type")


if __name__ == "__main__":
    main()