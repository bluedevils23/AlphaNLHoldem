import treys
import json
from typing import Dict

evaluator = treys.Evaluator()

def simulate_poker_hand() -> Dict:
    """模拟一手扑克牌并判断胜负"""
    # 创建一副新牌
    deck = treys.Deck()
    
    # 为两个玩家发牌
    player1_cards = deck.draw(2)
    player2_cards = deck.draw(2)
    
    # 发公共牌
    board = deck.draw(5)
    
    # 计算双方牌力
    player1_score = evaluator.evaluate(board, player1_cards)
    player2_score = evaluator.evaluate(board, player2_cards)
    
    # 转换成可读的牌面
    player1_cards_str = [treys.Card.int_to_str(card) for card in player1_cards]
    player2_cards_str = [treys.Card.int_to_str(card) for card in player2_cards]
    board_str = [treys.Card.int_to_str(card) for card in board]
    
    # 判断胜负（分数越低越好）
    if player1_score == player2_score:
        winner = "Draw game" 
    elif player1_score < player2_score:
        winner = "Player 1"
    elif player1_score > player2_score:
        winner = "Player 2"
    
    # 获取双方牌型
    player1_class = evaluator.get_rank_class(player1_score)
    player2_class = evaluator.get_rank_class(player2_score)
    
    return {
        "player1_cards": player1_cards_str,
        "player1_hand": evaluator.class_to_string(player1_class),
        "player2_cards": player2_cards_str,
        "player2_hand": evaluator.class_to_string(player2_class),
        "board": board_str,
        "winner": winner
    }

def convert_to_llm_format(result: Dict) -> Dict:
    """将模拟结果转换为LLM训练数据格式"""
    instruction_str = '\nYou are a poker expert. Given the following game situation, determine who has the winning hand.\n'
    input_str = f"Community cards: {result['board']}\n"
    input_str += f"Player 1 has: {result['player1_cards']} ({result['player1_hand']})\n"
    input_str += f"Player 2 has: {result['player2_cards']} ({result['player2_hand']})\n"
    input_str += "Who wins this hand?"
    
    return {
        "instruction": instruction_str,
        "input": input_str,
        "output": result['winner'],
        "player1_cards": ' '.join(result['player1_cards']),
        "player2_cards": ' '.join(result['player2_cards']),
        "board": ' '.join(result['board']),
        "player1_hand": result['player1_hand'],
        "player2_hand": result['player2_hand']
    }

if __name__ == "__main__":
    # 生成多个模拟结果
    llm_data = []
    num_hands = 10000  # 生成10w手牌
    
    for _ in range(num_hands):
        result = simulate_poker_hand()
        llm_data.append(convert_to_llm_format(result))
    
    # 保存为JSON文件
    with open('data/poker_hand_result_eval.json', 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    # 打印第一条记录作为示例
    print("\n生成的训练数据示例：")
    print(json.dumps(llm_data[0], indent=2, ensure_ascii=False))