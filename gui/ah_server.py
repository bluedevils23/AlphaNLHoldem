import asyncio
import json
import sys
import re
from enum import Enum
sys.path.append("../")
from ray.rllib.models import ModelCatalog
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.nl_holdem_net import NlHoldemNet
ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
import numpy as np
from tqdm import tqdm
import pandas as pd
from agi.evaluation_tools import NNAgent,death_match

#%%

conf = eval(open("../confs/nl_holdem.py").read().strip())

#%%

env = NlHoldemEnvWrapper(
        conf
)

#%%
print(env)
i = 1048
nn_agent = NNAgent(env.observation_space,
                       env.action_space,
                       conf,
                       f"../weights/c_{i}.pkl",
                       f"oppo_c{i}")

color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

#%%

# for i in tqdm(range(1)):
#     obs = env.reset()
    
    
#     d = False
#     while not d:
#         action_ind = nn_agent.make_action(obs)
#         obs,r,d,i = env.step(action_ind)

#         print(obs.keys())
#         print('card_info:\n')
#         print(obs['card_info'])
#         print('action_info:\n')
#         print(obs['action_info'])
#         print('legal_moves:\n')
#         print(obs['legal_moves'])
#         print('extra_info:\n')
#         print(obs['extra_info'])
#     #break

# #%%



# print(
#     env.env.get_state(0)["raw_obs"]["hand"],\
#     env.env.get_state(1)["raw_obs"]["hand"],\
#     env.env.get_state(1)["raw_obs"]["public_cards"],\
#     env.env.get_state(1)["action_record"]
#     )


# #%%

# print(env.env.get_payoffs())


#%%

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    ALL_IN = 3
    CHECK = 4

def parse_acpc_protocol(acpc_string):
    """将ACPC协议字符串转换为history对象
    
    Args:
        acpc_string: ACPC协议格式的字符串
        
    Returns:
        list: history对象，格式为[[stage1_actions], [stage2_actions], [stage3_actions], [stage4_actions]]
        每个action格式为[player_id, action_id, legal_actions]
    """
    # 初始化history
    history = [[], [], [], []]
    
    # 解析ACPC字符串
    parts = acpc_string.split(':')
    if len(parts) < 4:
        return history
        
    position = int(parts[1])  # 玩家位置
    betting_sequence = parts[3]  # 下注序列
    
    # 如果没有下注序列，返回空history
    if not betting_sequence:
        return history
        
    # 分割不同阶段的动作
    stages = betting_sequence.split('/')
    
    # 处理每个阶段的动作
    for stage_idx, stage_actions in enumerate(stages):
        if not stage_actions:
            continue
            
        # 解析这个阶段的所有动作
        actions = re.findall(r'([cr])(\d+)?', stage_actions)
        current_player = position  # 跟踪当前玩家
        
        for action in actions:
            action_type, amount = action
            
            # 确定动作类型
            if action_type == 'c':
                action_id = Action.CALL.value
            else:  # 'r'
                action_id = Action.RAISE.value
                
            # 所有可能的动作都是合法的
            legal_actions = [action.value for action in Action]
            
            # 添加到history中
            history[stage_idx].append([
                current_player,
                action_id,
                legal_actions
            ])
            
            # 切换到另一个玩家
            current_player = 1 - current_player
            
    return history

def history_to_acpc(history):
    """将history对象转换回ACPC格式字符串
    
    Args:
        history: history对象
        
    Returns:
        str: ACPC格式的动作序列
    """
    stages = []
    
    for stage_actions in history:
        stage_str = ''
        for player_id, action_id, _ in stage_actions:
            if action_id == Action.CALL.value:
                stage_str += 'c'
            elif action_id == Action.RAISE.value:
                # 注意：这里需要添加实际的raise金额
                stage_str += 'r???'  # 实际使用时需要替换为真实金额
                
        stages.append(stage_str)
        
    return '/'.join(stages)



    """
    Create a valid observation object for the poker environment.
    
    Args:
        hand_cards (list): List of hole cards e.g. ['AS', 'KH']
        public_cards (list): List of community cards e.g. ['JD', 'TD', '3C']
        legal_actions (list): List of legal action indices
        stakes (tuple): Tuple of (player_stake, opponent_stake)
        current_player (int): Current player ID (0 or 1)
        
    Returns:
        dict: Observation dictionary with required structure
    """

    if hand_cards is None:
        hand_cards = []
    if public_cards is None:
        public_cards = []
    if legal_actions is None:
        legal_actions = list(range(5))  # All actions legal by default
    if stakes is None:
        stakes = (0, 0)

    card_info = np.zeros([4,13,6],np.uint8)
    action_info = np.zeros([4,self.action_num,4 * 6 + 1],np.uint8) # 25 channel
    extra_info = np.zeros([2],np.uint8) # 25 channel
    legal_actions_info = np.zeros([self.action_num],np.uint8) # 25 channel
    
    hold_card = hand_cards
    public_card = public_cards
    current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
    
    for ind in current_legal_actions:
        legal_actions_info[ind] = 1
    
    flop_card = public_card[:3]
    turn_card = public_card[3:4]
    river_card = public_card[4:5]
    
    for one_card in hold_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][0] = 1
        
    for one_card in flop_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][1] = 1
        
    for one_card in turn_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][2] = 1
        
    for one_card in river_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][3] = 1
        
    for one_card in public_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][4] = 1
        
    for one_card in public_card + hold_card:
        card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][5] = 1
        
    
    for ind_round,one_history in enumerate(self.history):
        for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[:6]):
            action_info[player_id,action_id,ind_round * 6 + ind_h] = 1
            action_info[2,action_id,ind_round * 6 + ind_h] = 1
            
            for la_ind in legal_actions:
                action_info[3,la_ind,ind_round * 6 + ind_h] = 1
                
    action_info[:,:,-1] = self.my_agent()
    
    extra_info[0] = obs[0]["raw_obs"]["stakes"][0]
    extra_info[1] = obs[0]["raw_obs"]["stakes"][1]
    
    return {
        "card_info": card_info,
        "action_info": action_info,
        "legal_moves": legal_actions_info,
        "extra_info": extra_info,
    }



def create_obs(hand_cards=None, public_cards=None, history=None, legal_actions=None, stakes=None, current_player=0):
    """
    Create a valid observation object for the poker environment.
    
    Args:
        hand_cards (list): List of hole cards e.g. ['AS', 'KH']
        public_cards (list): List of community cards e.g. ['JD', 'TD', '3C']
        legal_actions (list): List of legal action indices
        stakes (tuple): Tuple of (player_stake, opponent_stake)
        current_player (int): Current player ID (0 or 1)
        
    Returns:
        dict: Observation dictionary with required structure
    """

    if hand_cards is None:
        hand_cards = []
    if public_cards is None:
        public_cards = []
    if legal_actions is None:
        legal_actions = list(range(5))  # All actions legal by default
    if stakes is None:
        stakes = (0, 0)
        
    # Initialize observation components
    card_info = np.zeros([4, 13, 6], np.uint8)
    action_info = np.zeros([4, 5, 25], np.uint8)
    extra_info = np.zeros([2], np.uint8)
    legal_actions_info = np.zeros([5], np.uint8)
    
    # Set legal actions
    for ind in legal_actions:
        legal_actions_info[ind] = 1
    
    # Process cards
    flop_cards = public_cards[:3]
    turn_cards = public_cards[3:4]
    river_cards = public_cards[4:5]
    
    # Process hole cards
    for card in hand_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][0] = 1
                
    # Process flop cards
    for card in flop_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][1] = 1
                
    # Process turn card
    for card in turn_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][2] = 1
                
    # Process river card
    for card in river_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][3] = 1
                
    # Process all public cards
    for card in public_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][4] = 1
                
    # Process all visible cards
    for card in public_cards + hand_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][5] = 1
    
    #Process action Info 
    for ind_round,one_history in enumerate(history):
        print('ind_round', ind_round)
        print('one_history', one_history)
        for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[:6]):
            #action_id = action_id.value
            action_info[player_id,action_id,ind_round * 6 + ind_h] = 1
            action_info[2,action_id,ind_round * 6 + ind_h] = 1
            
            for la_ind in legal_actions:
                action_info[3,la_ind,ind_round * 6 + ind_h] = 1
    # Set current player in action info
    action_info[:,:,-1] = current_player
    
    # Set stakes info
    extra_info[0] = stakes[0]
    extra_info[1] = stakes[1]
    
    return {
        "card_info": card_info,
        "action_info": action_info,
        "legal_moves": legal_actions_info,
        "extra_info": extra_info,
    }

def create_obs_from_env_state(env):
    """从环境状态创建observation对象
    
    Args:
        env: 游戏环境实例
        
    Returns:
        dict: observation对象
    """
    # 从env.env获取状态信息
    obs=env.last_obs
    hand_cards = obs[0]["raw_obs"]["hand"]
    print('hand_cards:', hand_cards)
    public_cards = obs[0]["raw_obs"]["public_cards"]
    print('public_cards:', public_cards)
    current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
    print('current_legal_actions:', current_legal_actions)
    # 获取动作历史
    action_record = obs[0]["action_record"]
    print('action record:', action_record)
    # 将action_record转换为history格式
    history = [[], [], [], []]  # preflop, flop, turn, river
    current_stage = 0
    
    for action in action_record:
        # 假设action_record中的每个动作是一个tuple (player_id, action_type)
        player_id, action_type = action
        
        # 添加到history
        history[current_stage].append([
            player_id,
            action_type,
            [0,1,2,3,4]  # 所有可能的动作
        ])
    print('history:', history)
    print('env_history:', env.history)
    # 获取当前合法动作
    legal_actions = [action.value for action in obs[0]["raw_obs"]["legal_actions"]]
    
    # 获取双方筹码
    stakes = obs[0]["raw_obs"]["stakes"]
    
    # 获取当前玩家
    current_player = obs[0]["raw_obs"]["current_player"] if "current_player" in obs[0]["raw_obs"] else 0
    print('current player:', current_player)
    
    # 使用create_obs函数创建observation
    return create_obs(
        hand_cards=hand_cards,
        public_cards=public_cards,
        history=env.history,
        legal_actions=legal_actions,
        stakes=stakes,
        current_player=current_player
    )

def test_obs_creation():
    """测试observation创建函数，逐个比较observation的各个部分"""
    import numpy as np
    
    # 创建一个环境实例
    env = NlHoldemEnvWrapper(
        conf
    )
    
    # 获取一个正常的observation作为参考
    
    for i in tqdm(range(1)):
        original_obs = env.reset()
        d = False
        while not d:
            action_ind = nn_agent.make_action(original_obs)
            original_obs,r,d,i = env.step(action_ind)
            print('legal_moves', original_obs['legal_moves'])
            # 使用环境状态创建新的observation
            new_obs = create_obs_from_env_state(env)
            
            # 逐个比较各个部分
            components = ["card_info", "action_info", "legal_moves", "extra_info"]
            
            for comp in components:
                print(f"\nComparing {comp}:")
                print(f"Original shape: {original_obs[comp].shape}")
                print(f"New shape: {new_obs[comp].shape}")
                
                if original_obs[comp].shape != new_obs[comp].shape:
                    print(f"Shape mismatch in {comp}!")
                    continue
                    
                # 检查数值是否相等
                is_equal = np.array_equal(original_obs[comp], new_obs[comp])
                print(f"Values are equal: {is_equal}")
                
                if not is_equal:
                    # 找出不同的位置
                    diff_indices = np.where(original_obs[comp] != new_obs[comp])
                    print(f"Differences found at indices: {diff_indices}")
                    print("Original values at these positions:", original_obs[comp][diff_indices])
                    print("New values at these positions:", new_obs[comp][diff_indices])
                    
                    # 如果是多维数组，打印第一个不同的位置的完整切片
                    if len(diff_indices[0]) > 0:
                        first_diff = tuple(index[0] for index in diff_indices)
                        print(f"\nFirst difference at index {first_diff}")
                        if len(original_obs[comp].shape) > 1:
                            print("Original slice:")
                            print(original_obs[comp][first_diff[0]])
                            print("New slice:")
                            print(new_obs[comp][first_diff[0]])
        #break
    
    
    return original_obs, new_obs


def convert_acpc_to_obs(acpc_state_str):
    """
    Convert ACPC protocol game state string to standardized observation space format.
    
    ACPC state format: MATCHSTATE:PLAYER:ROUND:ACTIONS:CARDS
    Returns observation in format compatible with NlHoldemEnv
    
    Args:
        acpc_state_str (str): Game state string in ACPC protocol format
    
    Returns:
        dict: Standardized observation space with card_info, action_info, legal_moves, extra_info
    """
    
    try:
        # Split ACPC state into components
        parts = acpc_state_str.split(':')
        
        # Ensure we have at least 5 parts
        while len(parts) < 5:
            parts.append('')
        
        # Parse components
        matchstate, player, round_num, actions, cards = parts
        player = int(player)
        
        # Initialize observation tensors
        card_info = np.zeros([4, 13, 6], np.uint8)
        action_info = np.zeros([4, 5, 25], np.uint8)
        legal_actions_info = np.zeros([5], np.uint8)
        extra_info = np.zeros([2], np.uint8)
        
        # Parse cards
        hand_p0, hand_p1, public_cards = [], [], []
        if cards:
            # Split player and community cards
            player_cards_part = cards.split('|')[0] if '|' in cards else cards
            community_cards_part = cards.split('|')[1] if '|' in cards else ''
            
            # Handle player cards
            player_card_rounds = player_cards_part.split('/')
            hand_p0 = player_card_rounds[0].split(',') if player_card_rounds and player_card_rounds[0] else []
            hand_p1 = player_card_rounds[1].split(',') if len(player_card_rounds) > 1 else []
            
            # Handle community cards
            community_card_rounds = community_cards_part.split('/')
            for round_cards in community_card_rounds:
                if round_cards:
                    public_cards.extend(round_cards.split(','))
        
        # Fill card info tensor
        # Player's hole cards

        for card in (hand_p0 if player == 0 else hand_p1):
            if len(card) >= 2:
                # ACPC format: rank then suit (e.g. "2h")
                rank, suit = card[0], card[1].lower()
                if suit in color2ind and rank in rank2ind:
                    card_info[color2ind[suit]][rank2ind[rank]][0] = 1
        
        # Public cards
        for i, card in enumerate(public_cards):
            if len(card) >= 2:
                # ACPC format: rank then suit (e.g. "2h")
                rank, suit = card[0], card[1].lower()
                if suit in color2ind and rank in rank2ind:
                    # Flop cards
                    if i < 3:
                        card_info[color2ind[suit]][rank2ind[rank]][1] = 1
                    # Turn card
                    elif i == 3:
                        card_info[color2ind[suit]][rank2ind[rank]][2] = 1
                    # River card
                    elif i == 4:
                        card_info[color2ind[suit]][rank2ind[rank]][3] = 1
                    # All visible cards
                    card_info[color2ind[suit]][rank2ind[rank]][4] = 1
        
        # Parse actions and fill action info tensor
        if actions:
            round_actions = actions.split('/')
            action_records = []
            current_round = 0
            
            for round_actions in round_actions:
                if round_actions:
                    for pos, action in enumerate(round_actions):
                        player_id = pos % 2
                        action2ind = {'c': 0, 'f': 1, 'r': 2, 'k': 3, 'b': 4}
                        if action in action2ind:
                            action_id = action2ind[action]
                            action_info[player_id, action_id, current_round * 6 + pos] = 1
                            action_info[2, action_id, current_round * 6 + pos] = 1  # Global action history
                    current_round += 1
        
        # Set legal actions
        legal_actions = _get_legal_actions(actions)
        for action in legal_actions:
            if isinstance(action, int) and 0 <= action < 5:
                legal_actions_info[action] = 1
        
        # Set extra info (stakes)
        # Note: In ACPC format, we might need to calculate stakes from action history
        # For now using placeholder values
        extra_info[0] = 1  # Placeholder for player stake
        extra_info[1] = 1  # Placeholder for opponent stake
        
        return {
            "card_info": card_info,
            "action_info": action_info,
            "legal_moves": legal_actions_info,
            "extra_info": extra_info,
        }
    
    except Exception as e:
        print(f"Error converting ACPC state: {e}")
        return None

def _get_legal_actions(actions_str):
    """
    Derive legal actions based on current game state
    
    Args:
        actions_str (str): Actions string from ACPC state
    
    Returns:
        list: Possible legal actions
    """
    legal_actions = []
    
    # Basic legal actions
    base_actions = [
        'check/call',  # 'c'
        'fold',        # 'f'
        'raise'        # 'r'
    ]
    
    # If no actions, all actions are legal
    if not actions_str:
        return base_actions
    
    # Last action determines next possible actions
    last_action = actions_str.split('/')[-1][-1] if actions_str else ''
    
    if last_action == 'c':  # If last action was call
        legal_actions = ['check/call', 'raise', 'fold']
    elif last_action == 'r':  # If last action was raise
        legal_actions = ['call', 'raise', 'fold']
    else:
        legal_actions = base_actions
    
    return legal_actions

# Comprehensive test function
def test_acpc_to_obs_conversion():
    test_cases = [
        "MATCHSTATE:0:30::9s8h|",
        "MATCHSTATE:0:30:c:9s8h|",
        "MATCHSTATE:0:30:c:9s8h|:c",
        "MATCHSTATE:0:30:cc/:9s8h|/8c8d5c",
        "MATCHSTATE:0:30:cc/r250:9s8h|/8c8d5c",
        "MATCHSTATE:0:30:cc/r250c/:9s8h|/8c8d5c/6s",
        "MATCHSTATE:1:31:r300r900c/r1800r3600r9000c/:|JdTc/6dJc9c/Kh",
        "MATCHSTATE:1:31:r300r900c/r1800r3600r9000c/r20000:|JdTc/6dJc9c/Kh:c",
        "MATCHSTATE:1:31:r300r900c/r1800r3600r9000c/r20000c/:KsJs|JdTc/6dJc9c/Kh/Qc"
    ]
    
    for acpc_state in test_cases:
        obs_state = convert_acpc_to_obs(acpc_state)
        print(f"ACPC State: {acpc_state}")
        print("Converted OBS State:", obs_state)
        print("---")

# Run the test
#test_acpc_to_obs_conversion()

#test_obs_creation()

#*************************
# # 示例ACPC字符串
# acpc_str = "MATCHSTATE:0:30:cc/r250c/r500c/r1250c:9s8h|9c6h/8c8d5c/6s/2d"

# # 转换为history对象
# history = parse_acpc_protocol(acpc_str)

# # 打印结果
# for stage_idx, stage in enumerate(history):
#     print(f"Stage {stage_idx}:", stage)
#*************************


#*************************
# obs = create_obs(
#     hand_cards=['SA', 'HK'],  # 黑桃A和红心K
#     public_cards=['DJ', 'DT', 'C3'],  # 公共牌：方块J、方块10、梅花3
#     legal_actions=[0, 1, 2],  # 可用动作
#     stakes=(10, 20),  # 双方赌注
#     current_player=0  # 当前玩家
# )
# print('card_info:')
# print(obs['card_info'])
# print('action_info:')
# print(obs['action_info'])
# print('legal_moves:')
# print(obs['legal_moves'])
# print('extra_info:')
# print(obs['extra_info'])
#*************************

def convert_card_format(card):
    """将ACPC格式的牌转换为我们的格式
    
    Args:
        card: ACPC格式的牌，例如'Ah'（A♥），格式是[rank][suit]
    
    Returns:
        str: 我们的格式，例如'HA'（♥A），格式是[suit][rank]
    """
    # ACPC格式: [rank][suit]，例如'Ah'表示A♥
    # 我们的格式: [suit][rank]，例如'HA'表示♥A
    rank_map = {'t': '10', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'}
    
    if not card or len(card) != 2:
        return None
        
    rank, suit = card[0], card[1]
    
    # 转换rank
    if rank.lower() in rank_map:
        rank = rank_map[rank.lower()]
    else:
        rank = rank.upper()
    
    # 转换suit
    suit = suit.upper()
    
    # 返回[suit][rank]格式
    return suit + rank

async def handle_acpc_connection(reader, writer):
    """处理单个ACPC连接的协程，持续接收数据直到连接关闭
    
    Args:
        reader: StreamReader对象，用于读取数据
        writer: StreamWriter对象，用于发送数据
    """
    addr = writer.get_extra_info('peername')
    print(f"New connection from {addr}")
    
    try:
        while True:  # 持续接收数据
            # 读取数据直到遇到换行符
            data = await reader.readline()
            if not data:  # 如果客户端关闭连接
                print(f"Connection closed by {addr}")
                break
                
            acpc_string = data.decode().strip()
            print(f"Received from {addr}: {acpc_string}")
            
            # 1. 将ACPC字符串转换为history对象
            history = parse_acpc_protocol(acpc_string)
            for stage_idx, stage in enumerate(['preflop', 'flop', 'turn', 'river']):
                print(f"{stage}: {history[stage_idx]}")
            
            # 2. 从ACPC字符串中提取手牌和公共牌信息
            parts = acpc_string.split(':')
            if len(parts) >= 5:  # 确保有足够的部分
                cards_part = parts[4]
                card_sections = cards_part.split('/')
                
                # 1. 获取手牌并转换格式
                hole_cards = card_sections[0].split('|')[0].strip()
                acpc_hand_cards = [hole_cards[i:i+2] for i in range(0, len(hole_cards), 2)] if hole_cards else []
                hand_cards = [convert_card_format(card) for card in acpc_hand_cards]
                print(f"Hand cards (ACPC format): {acpc_hand_cards}")
                print(f"Hand cards (our format): {hand_cards}")

                # 2. 获取公共牌并转换格式
                public_cards = []
                acpc_public_cards = []
                for section in card_sections[1:]:
                    cards = section.strip()
                    if cards:
                        section_cards = [cards[i:i+2] for i in range(0, len(cards), 2)]
                        acpc_public_cards.extend(section_cards)
                        public_cards.extend([convert_card_format(card) for card in section_cards])
                print(f"Public cards (ACPC format): {acpc_public_cards}")
                print(f"Public cards (our format): {public_cards}")

                # 3. 获取当前玩家
                current_player = int(parts[1])
                print(f"Current player: {current_player}")

                # 4. 获取合法动作
                legal_actions = _get_legal_actions(parts[3])
                print(f"Legal actions: {legal_actions}")
                
                print("\nCalling create_obs with parameters:")
                print(f"hand_cards: {hand_cards}")
                print(f"public_cards: {public_cards}")
                print(f"history: {history}")
                print(f"legal_actions: {legal_actions}")
                print(f"stakes: (0, 0)")
                print(f"current_player: {current_player}")
                
                # 5. 创建observation
                obs = create_obs(
                    hand_cards=hand_cards,
                    public_cards=public_cards,
                    history=history,
                    legal_actions=legal_actions,
                    stakes=(0, 0),  # 这里可能需要从ACPC字符串中提取实际的stakes
                    current_player=current_player
                )
    except Exception as e:
            # 发生错误时发送错误信息
            print(f"Error handling connection from {addr}: {e}")
            error_response = json.dumps({"status": "error", "message": str(e)}) + "\n"
            try:
                writer.write(error_response.encode())
                await writer.drain()
            except:
                pass
            
    finally:
        # 关闭连接
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass
        print(f"Connection to {addr} closed")


async def run_acpc_server():
    """运行ACPC服务器"""
    server = await asyncio.start_server(
        handle_acpc_connection, 
        '0.0.0.0',  # 监听所有网络接口
        8888        # 端口号
    )

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()

def start_acpc_server():
    """启动ACPC服务器的入口函数"""
    try:
        # 在Windows上需要使用不同的事件循环策略
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 运行服务器
        asyncio.run(run_acpc_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

# 测试用的客户端函数
async def test_acpc_client(acpc_string):
    """测试ACPC客户端
    
    Args:
        acpc_string: 要发送的ACPC协议字符串
    """
    try:
        reader, writer = await asyncio.open_connection(
            '127.0.0.1', 
            8888
        )
        
        # 发送ACPC字符串
        writer.write((acpc_string + "\n").encode())
        await writer.drain()
        
        # 接收响应
        data = await reader.readline()
        response = json.loads(data.decode())
        print(f"Received: {response}")
        
        # 关闭连接
        writer.close()
        await writer.wait_closed()
        
        return response
        
    except Exception as e:
        print(f"Client error: {e}")
        return None

def test_server():
    """测试持续运行的服务器"""
    # 在新线程中启动服务器
    import threading
    server_thread = threading.Thread(target=start_acpc_server)
    server_thread.daemon = True  # 设置为守护线程
    server_thread.start()
    
    # 等待服务器启动
    import time
    time.sleep(1)
    
    # 测试多次发送ACPC字符串
    async def run_test_client():
        test_strings = [
            "MATCHSTATE:0:0:cr300c/r900c/r2700c/:Ah2h|/8s6h4c/Kd/9c",
            "MATCHSTATE:0:1:cc/r250c/r500c/:Ks9d|/8c8d5c/6s/",
            "MATCHSTATE:0:2:cr400c//:As2s|/7h4d2c",
        ]
        
        for test_string in test_strings:
            print(f"\nSending: {test_string}")
            response = await test_acpc_client(test_string)
            print(f"Response received: {response}")
            time.sleep(1)  # 等待1秒再发送下一个
    
    # 运行测试客户端
    asyncio.run(run_test_client())

#test_server()

if __name__ == "__main__":
    # 如果直接运行此文件，启动服务器
    start_acpc_server()