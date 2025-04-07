import json
import re
from typing import Dict, List
from pprint import pprint
import treys


evaluator = treys.Evaluator()
#Card = treys.Card()
def convert_card_to_treys(card: str) -> int:
    """将扑克牌转换为 treys 格式"""
    return treys.Card.new(card)

def evaluate_hand_strength(hole_cards: List[str], community_cards: List[str]) -> str:
    """使用 treys 评估手牌强度"""
    
    
    # 转换手牌和公共牌为 treys 格式
    treys_hole = [convert_card_to_treys(card) for card in hole_cards]
    treys_board = [convert_card_to_treys(card) for card in community_cards] if community_cards else []
    
    # 如果没有公共牌，只返回手牌等级（1-169，1最强）
    if not treys_board:
        return ''
    
    # 计算手牌强度（越低越强）
    hand_value = evaluator.evaluate(treys_board, treys_hole)
    # 转换为百分比（0-100，100最强）
    hand_class = evaluator.get_rank_class(hand_value)
    hand_percentage = (7462 - hand_value) / 7462 * 100
    
    return evaluator.class_to_string(hand_class)

def standardize_action(action: str, pot_size: float) -> str:
    """Convert PokerStars action to standard format"""
    if 'folds' in action:
        return "Fold"
    elif 'calls' in action or 'checks' in action:
        return "CheckCall"
    elif 'raises' in action or 'bets' in action:
        amount = float(re.search(r'\$(\d+)', action).group(1))
        if amount >= pot_size * 3:  # All-in or very large raise
            return "RaiseMax"
        elif amount >= pot_size * 2:
            return "RaiseTwoPot"
        elif amount >= pot_size:
            return "RaisePot"
        else:
            return "RaiseHalfPot"
    return "CheckCall"  # default case

def extract_cards(hand_text: str) -> tuple[List[str], List[str]]:
    """从手牌历史中提取 DeepStack 的手牌和公共牌"""
    hole_cards = []
    community_cards = []
    
    lines = hand_text.strip().split('\n')
    for line in lines:
        # 提取 DeepStack 的手牌
        if line.startswith('Dealt to DeepStack'):
            cards = re.findall(r'\[(.*?)\]', line)
            if cards:
                hole_cards = cards[0].split()
        # 提取公共牌
        elif '*** FLOP ***' in line:
            cards = re.findall(r'\[(.*?)\]', line)
            if cards:
                community_cards.extend(cards[0].split())
        elif '*** TURN ***' in line:
            cards = re.findall(r'\[(.*?)\]', line)
            if cards:
                community_cards.append(cards[0].strip())
        elif '*** RIVER ***' in line:
            cards = re.findall(r'\[(.*?)\]', line)
            if cards:
                community_cards.append(cards[0].strip())
    
    return hole_cards, community_cards

def extract_decision_points(hand_text: str) -> List[Dict]:
    """Extract decision points from hand history"""
    decision_points = []
    current_pot = 150  # Start with blinds
    lines = hand_text.strip().split('\n')
    
    # 获取手牌和公共牌
    hole_cards, community_cards = extract_cards(hand_text)

    # Track current state
    for i, line in enumerate(lines):
        # When we find a DeepStack action (except posting blinds), create a decision point
        if line.startswith('DeepStack: ') and not 'posts' in line:
            # Get the action
            action = line.split(': ', 1)[1]
            
            # Get all previous lines as situation
            situation = lines[:i]
            
            # 添加手牌和当前可见的公共牌信息到决策点
            current_community_cards = []
            for prev_line in lines[:i]:
                if '*** FLOP ***' in prev_line:
                    current_community_cards = community_cards[:3]
                elif '*** TURN ***' in prev_line:
                    current_community_cards = community_cards[:4]
                elif '*** RIVER ***' in prev_line:
                    current_community_cards = community_cards[:5]
            
            hand_strength = evaluate_hand_strength(hole_cards, current_community_cards)
            instruction_str = '\nYou are Deepstack, a specialist in playing heads up No Limit Texas Holdem. The following will be a game scenario and you need to make the opimal decision.\n\nHere is a game summary:\n'
            input_str = "\n".join(situation)
            input_str += "\nDeepstack's holecards are:" + str(hole_cards)
            if current_community_cards:
                input_str += "\nDeepstack's community cards are:" + str(current_community_cards)
                input_str += "\nDeepstack now has:" + hand_strength
            input_str += "\nNow it is Deepstack's turn to make a move.\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:"
            # Create decision point
            decision_point = {
                "instruction": instruction_str,
                "input": input_str,
                "output": standardize_action(action, current_pot),
                "hole_cards": ' '.join(hole_cards),
                "community_cards": ' '.join(current_community_cards),
                "hand_strength": hand_strength,
            }
            #pprint(decision_point)
            # 只添加非 showdown 的决策点
            if not any('SHOW DOWN' in l for l in situation):
                decision_points.append(decision_point)
            
            # Update pot size for next decision
            if 'raises' in action or 'bets' in action:
                amount = float(re.search(r'\$(\d+)', action).group(1))
                current_pot += amount
            elif 'calls' in action:
                amount = float(re.search(r'\$(\d+)', action).group(1))
                current_pot += amount
        
        # Update pot size from opponent actions
        elif ': ' in line and not line.startswith('DeepStack:') and not 'posts' in line:
            action = line.split(': ', 1)[1]
            if 'raises' in action or 'bets' in action:
                amount = float(re.search(r'\$(\d+)', action).group(1))
                current_pot += amount
            elif 'calls' in action:
                amount = float(re.search(r'\$(\d+)', action).group(1))
                current_pot += amount
    
    return decision_points

def convert_hands_to_llm_format(input_file: str, output_file: str):
    """Convert multiple hands to LLM format"""
    llm_data = []
    
    with open(input_file, 'r') as f:
        content = f.read()
        hands = content.split("\n\n")
        
        for hand in hands:
            if not hand.strip():
                continue
            
            decision_points = extract_decision_points(hand)
            llm_data.extend(decision_points)
    
    with open(output_file, 'w') as f:
        json.dump(llm_data, f, indent=2)

if __name__ == "__main__":
    input_file = "data/all_hands.log"
    output_file = "data/llm_training_data.json"
    convert_hands_to_llm_format(input_file, output_file)
