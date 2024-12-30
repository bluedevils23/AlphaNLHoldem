import OpenHoldem
import math
import sys
import time
#import slumbot_util
import socket
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# 使用FileHandler输出到文件
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.settimeout(100)
#client_socket.connect(('10.164.142.211', 8888))

class Main:
    gotcaught = False
    ibluffed = False
    inv = -1
    phr = -1
    serverBigBlind = 20
    startStack = 1000
    smallStack = 1000
    rate = 0
    oh = {
        'betround': -1,
        'handrank169': -1,
        'prwin': -1,
        'prtie': -1,
        'prlos': -1,
        'nplayersplaying': -1,
        'call': -1,
        'currentbet': -1,
        'BetSize' : -1,
        'balance': -1,
        'bblind': -1,
        'sblind': -1,
        'didfold': -1,
        'didchec': -1,
        'didcall': -1,
        'didrais': -1,
        'didbetsize': -1,
        'didalli': -1,
        #### added by yangliu ####
        'Bets': -1,
        'Calls': -1,
        'Checks': -1,
        'Raises': -1,
        'userchair': -1,
        'bigblindchair': -1,
        'balance0': -1,
        'balance1': -1,
        'smallblindchair': -1,
        'lastcallerchair': -1,
        'lastraiserchair': -1,
        'AmountToCall' : -1,
        'pot' : -1,
        'lastraised1' : -1,
        'nbetsround' : -1,
        #### newly added by yangliu ####
        'StackSize' : -1,
        'PotSize' : -1,
        'MaxOpponentStackSize' : -1,
        'MaxStacksizeOfActiveOpponents' : -1,

        #### hand card and suit values ####
        '$$pr0' : -1,
        '$$pr1' : -1,
        '$$ps0' : -1,
        '$$ps1' : -1,
        #### board cards and suit values ####
        '$$cr0' : -1,
        '$$cr1' : -1,
        '$$cr2' : -1,
        '$$cr3' : -1,
        '$$cr4' : -1,
        '$$cs0' : -1,
        '$$cs1' : -1,
        '$$cs2' : -1,
        '$$cs3' : -1,
        '$$cs4' : -1,

    }

    def __init__(self):
        self.gotcaught = False
        self.ibluffed = False
        self.inv = 0
        self.phr = 0
        self.lastHandNo = 0
        self.lastHole = ''
        self.previousBetround = 1
        self.actionsAll = ''
        for k, v in self.oh.items():
            self.oh[k] = 0

    def updateVars(self):
        for k, v in self.oh.items():
            self.oh[k] = OpenHoldem.getSymbol(k)
            # if k in ['betround','currentbet','bblind','sblind','pot']:
            #     logger.info('%s:%s', k, self.oh[k])
        self.phr = (170.0 - self.oh['handrank169'])/169.0
        self.inv = 1.0/self.oh["nplayersplaying"]
        #logger.info(f'1/nplayers: {self.inv}')
        self.oh['bblind'] = self.oh['sblind'] * 2
        self.betround = int(self.oh["betround"])
        if self.oh["betround"] == 1:
            self.gotcaught = False
            self.ibluffed = False
        if self.oh["betround"] > 1 and self.timesActed() > 0 and self.ibluffed == True:
            self.gotcaught = True

    def timesActed(self):
        return int(self.oh["didfold"] + self.oh["didchec"] + self.oh["didcall"] + self.oh["didrais"] + self.oh["didbetsize"])

    def callExpectedValue(self):
        ev = self.oh["prwin"]*self.oh["pot"] + self.oh["prtie"]*self.inv*self.oh["pot"] - self.oh["prlos"]*self.oh["call"]
        logger.info(f'ev: {ev}')
        return ev

    def preFlopDecision(self):
        decision = 0.0
        logger.info(f'phr: {self.phr}')
        if 0.95 < self.phr:
            if self.timesActed() == 0:
                decision = OpenHoldem.getSymbol("RaiseHalfPot")
            else:
                decision = OpenHoldem.getSymbol("RaiseMax")
            logger.info('-> 0.95')
        elif 0.85 < self.phr and self.oh["call"] <= 13.0*self.oh["bblind"]:
            if self.timesActed() == 0:
                decision = OpenHoldem.getSymbol("RaiseHalfPot")
            else:
                decision = OpenHoldem.getSymbol("Call")
            logger.info('-> 0.85')
        elif 0.70 < self.phr and self.oh["call"] <= 3.0*self.oh["bblind"]:
            if self.timesActed() == 0:
                decision = OpenHoldem.getSymbol("Call")
            logger.info('-> 0.70')
        #### debug action ####
        #decision = OpenHoldem.getSymbol("Call")
        return decision

    def postFlopDecision(self):
        decision = 0.0
        min_bet = max(2.0*self.oh["call"], self.oh["bblind"])
        if 0.40 < self.oh["prwin"] - self.inv:
            if self.timesActed() == 0:
                decision = OpenHoldem.getSymbol("RaisePot")
            else:
                decision = OpenHoldem.getSymbol("RaiseMax")
        elif 0.1 < self.oh["prwin"] - self.inv and math.isclose(0, self.oh["call"], rel_tol=1e-6) and self.gotcaught == False:
            decision = OpenHoldem.getSymbol("RaiseHalfPot")
            self.ibluffed = True
        elif self.oh["call"] < self.callExpectedValue():
            decision = OpenHoldem.getSymbol("Call")
        return decision

    def getDecision2(self):
        #return OpenHoldem.getSymbol("f$betpot_1_2")
        decision = 0.0
        position = -1
        self.pot = -1
        self.updateVars()
        self.rate = self.oh['bblind']/self.serverBigBlind
        # if self.oh['pot'] == 0:
        #     self.pot = self.oh['bblind'] + self.oh['sblind']
        # else:
        #     self.pot = self.oh['pot']
        self.pot = self.oh['pot']
        if self.oh['bigblindchair'] == self.oh['userchair']:
            position = 1
            self.opponentChair = self.oh['smallblindchair']
        else:
            position = 0
            self.opponentChair = self.oh['bigblindchair']

        hole = self.getHand()
        board = self.getBoard()

        # logger.info(OpenHoldem.GetHandnumber())
        # handNo = int(OpenHoldem.GetHandnumber())
        ### New hand here
        # if self.lastHandNo != handNo:
        #     self.lastHandNo = handNo
        #     self.actionsAll = ''
        ### 888poker doesnt support handnumber, calc handnumber by our code.
        ### TODO: We may get 2 same hands.
        if self.lastHole != hole:
            self.lastHole = hole
            self.lastHandNo = self.lastHandNo + 1
            self.actionsAll = ''
            logger.info('\n\n')
            logger.info('Starting new handNo:%d', self.lastHandNo)
        handNo = self.lastHandNo

        logger.info('---------------------')
        logger.info('handNo:%s' % handNo)
        logger.info('betround:%s' % self.betround)
        logger.info('position:%s' % position)
        logger.info('sblind:%s' % self.oh['sblind'])
        logger.info('bblind:%s' % self.oh['bblind'])
        logger.info('hand:%s' % hole)
        logger.info('board:%s' % str(board))
        logger.info('pot:%s' % self.oh['pot'])
        logger.info('previousBetround:%d' % self.previousBetround)
        logger.info('---------------------')

        ### Add opponent actions to self.actionsAll
        actions = ''
        if self.betround == 1:
            if self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 0:
                actions = ''
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 1:
                if self.oh['Checks'] == 1 and self.oh['AmountToCall'] == 0.5:
                    actions = 'c'
                elif self.oh['Calls'] == 1 and self.oh['AmountToCall'] == 0:
                    actions = 'c'
                elif self.oh['Raises'] == 1 and self.oh['AmountToCall'] > 0:
                    actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind) )
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] > 1:
                if self.oh['AmountToCall'] == 0:
                    actions = 'c'
                else:
                    actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
        else:
            if self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 0:
                logger.info('Act:0')
                if self.actionsAll[-1] != '/':
                    actions = 'c/'
                else:
                    actions = ''
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 1:
                logger.info('Act:1')
                if self.oh['AmountToCall'] == 0:
                    if self.actionsAll[-1] != '/':
                        actions = 'c/c'
                    else:
                        actions = 'c'
                else:
                    if self.actionsAll[-1] != '/' and self.oh['didchec'] + self.oh['didcall'] + self.oh['didrais'] + self.oh['didbetsize']== 0:
                        actions = 'c/r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                    else:
                        actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
            else:
                logger.info('Act:>1')
                if self.oh['AmountToCall'] == 0:
                    actions = 'c'
                else:
                    actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))

        # if self.oh['lastcallerchair'] == -1 and self.oh['lastraiserchair'] == -1:
        #     if self.betround != 1 and self.actionsAll[-1] != '/':
        #         actions = 'c/'
        #     else:
        #         if self.betround == 1 and self.oh['AmountToCall'] == 0.5:
        #             actions = 'c'
        # if self.oh['lastcallerchair'] == self.opponentChair:
        #         actions = 'c'
        # elif self.oh['lastraiserchair'] == self.opponentChair:
        #     if self.betround == 1 and self.oh['AmountToCall'] == 0.5:
        #         actions = 'c'
        #     else:
        #         actions = 'r'+str(int(self.oh['AmountToCall'] * self.serverBigBlind))
        self.actionsAll = self.actionsAll + actions

        # if self.oh["betround"] == 1:
        #     if position == 0:
        #         if self.oh['AmountToCall'] == 0.5:
        #             actions = ''
        #         else:
        #             actions = 'r'+ str(self.oh['AmountToCall'] * 20)
        #     elif position == 1:
        #         if self.oh['AmountToCall'] == 0.0:
        #             actions = 'c'
        #         else:
        #             actions = 'r'+ str(self.oh['AmountToCall'] * 20)
        msg = ''
        if self.betround == 1 and self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] < 2:
            if self.oh['balance'] > self.startStack:
                self.smallStack = 2 * self.startStack - self.oh['balance']
            else:
                self.smallStack = self.oh['balance']
            msg = str(math.floor(self.smallStack / self.rate))+';'

        msg = msg + "MATCHSTATE:" + str(position) + ":" + \
            str(handNo) + ":" + self.actionsAll + ":"
        if position == 0:
            msg += hole + "|"
        elif position == 1:
            msg += "|" + hole
        if len(board) > 0:
            msg += "/" + board
        msg += "\n"

        client_socket.send(msg.encode())
        logger.info('sent:%s', msg.strip())
        time_start = time.time()
        try:
            advice = client_socket.recv(100).decode()
            logger.info('received:%s', advice)
        except socket.timeout:
            logger.warn('Socket timeout')
            advice = 'c'
        time_end = time.time()
        logger.info(advice)
        logger.info('time for betround %d:%.3f sec', self.betround, time_end - time_start)

        if advice == 'c':
            decision = OpenHoldem.getSymbol("Call")
        elif advice == 'f':
            decision = OpenHoldem.getSymbol("Fold")
        elif advice.strip().isdigit():
            raiseAmount = int(advice.strip()) * self.rate
            logger.info('Raise:%d' % raiseAmount)

            if raiseAmount / self.pot <= 0.5:
                logger.info('RaiseHalfPot')
                decision = OpenHoldem.getSymbol("RaiseHalfPot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 0.67:
                logger.info('RaiseTwoThirdPot')
                decision = OpenHoldem.getSymbol("RaiseTwoThirdPot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 0.75:
                logger.info('RaiseThreeFourthPot')
                decision = OpenHoldem.getSymbol("RaiseThreeFourthPot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 1:
                logger.info('RaisePot')
                decision = OpenHoldem.getSymbol("RaisePot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 1.5:
                logger.info('RaisePotOneAndHalf')
                decision = OpenHoldem.getSymbol("RaisePot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 2:
                logger.info('RaisePot2')
                decision = OpenHoldem.getSymbol("RaisePot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 3:
                logger.info('RaisePot3')
                decision = OpenHoldem.getSymbol("RaisePot")
                logger.info(decision)
            elif raiseAmount / self.pot <= 10:
                logger.info('RaiseMax')
                logger.info(self.pot)
                decision = OpenHoldem.getSymbol("RaiseMax")
                logger.info(decision)

            # elif raiseAmount <= 60:
            #     logger.info('Raise60')
            #     decision = OpenHoldem.getSymbol("Raise60")
            #     logger.info(decision)
            # elif raiseAmount <= 80:
            #     logger.info('Raise80')
            #     decision = OpenHoldem.getSymbol("Raise80")
            #     logger.info(decision)
            # elif raiseAmount <= 100:
            #     logger.info('Raise100')
            #     decision = OpenHoldem.getSymbol("Raise100")
            #     logger.info(decision)
            # elif raiseAmount <= 100:
            #     logger.info('Raise100')
            #     decision = OpenHoldem.getSymbol("Raise100")
            #     logger.info(decision)
            # elif raiseAmount <= 200:
            #     logger.info('Raise200')
            #     decision = OpenHoldem.getSymbol("Raise200")
            #     logger.info(decision)
            # elif raiseAmount <= 300:
            #     logger.info('Raise300')
            #     decision = OpenHoldem.getSymbol("Raise300")
            #     logger.info(decision)
            # elif raiseAmount <= 400:
            #     logger.info('Raise400')
            #     decision = OpenHoldem.getSymbol("Raise400")
            #     logger.info(decision)
            # elif raiseAmount <= 500:
            #     logger.info('Raise500')
            #     decision = OpenHoldem.getSymbol("Raise500")
            #     logger.info(decision)
            # elif raiseAmount <= 600:
            #     logger.info('Raise600')
            #     decision = OpenHoldem.getSymbol("Raise600")
            #     logger.info(decision)
            # elif raiseAmount <= 700:
            #     logger.info('Raise700')
            #     decision = OpenHoldem.getSymbol("Raise700")
            #     logger.info(decision)
            # elif raiseAmount <= 800:
            #     logger.info('Raise800')
            #     decision = OpenHoldem.getSymbol("Raise800")
            #     logger.info(decision)
            # elif raiseAmount <= 900:
            #     logger.info('Raise900')
            #     decision = OpenHoldem.getSymbol("Raise900")
            #     logger.info(decision)
            # elif raiseAmount <= 1000:
            #     logger.info('Raise1000')
            #     decision = OpenHoldem.getSymbol("Raise1000")
            #     logger.info(decision)
            else:
                logger.info('RaiseDefault:RaisePot')
                decision = OpenHoldem.getSymbol("RaisePot")
        else:
            logger.info('Check')
            decision = OpenHoldem.getSymbol("Check")

        # if self.oh["betround"] == 1:
        #     if self.oh["prwin"] > self.inv:
        #         decision = self.preFlopDecision()
        # else:
        #     decision = self.postFlopDecision()
        # logger.info(f'decision: {decision}')
        logger.info('decision:%s' % self.convertDecision(decision))

        ### Add bot actions to self.actionAll
        actions = ''
        if decision == OpenHoldem.getSymbol("Call") or decision == OpenHoldem.getSymbol("Check"):
            if self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 0:
                actions = 'c' ### first to check
            else:
                actions = 'c/' ### teminate current bet round
        elif decision == OpenHoldem.getSymbol("RaiseHalfPot"):
            actions = 'r'+ str(int(0.5 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("RaiseTwoThirdPot"):
            actions = 'r'+ str(int(0.67 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("RaiseThreeFourthPot"):
            actions = 'r'+ str(int(0.75 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("RaisePot"):
            actions = 'r'+ str(int(self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("f$RaisePotOneAndHalf"):
            actions = 'r'+ str(int(1.5 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("f$RaisePot2"):
            actions = 'r'+ str(int(2 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("f$RaisePot3"):
            actions = 'r'+ str(int(3 * self.pot/self.rate))
        elif decision == OpenHoldem.getSymbol("RaiseMax"):
            actions = 'r1000'
        #logger.info('actionsall:%s' % self.actionsAll)
        self.actionsAll = self.actionsAll + actions
        logger.info('actionsall:%s' % self.actionsAll)

        self.previousBetround = self.betround
        return decision

    def convertDecision(self, decision):
        if decision  == OpenHoldem.getSymbol("Call"):
            return 'Call'
        elif decision  == OpenHoldem.getSymbol("Check"):
            return 'Check'
        elif decision == OpenHoldem.getSymbol("Fold"):
            return 'Fold'
        elif  decision  == OpenHoldem.getSymbol("Raise"):
            return 'Raise'
        elif decision == OpenHoldem.getSymbol("RaiseHalfPot"):
            return 'RaiseHalfPot'
        elif decision == OpenHoldem.getSymbol("RaiseTwoThirdPot"):
            return 'RaiseTwoThirdPot'
        elif decision == OpenHoldem.getSymbol("RaiseThreeFourthPot"):
            return 'RaiseThreeFourthPot'
        elif decision == OpenHoldem.getSymbol("RaisePot"):
            return 'RaisePot'
        elif decision == OpenHoldem.getSymbol("RaiseMax"):
            return 'RaiseMax'
        return 'unknown'

    def convertCard(self, card_num):
        if card_num == 14:
            return 'A'
        if card_num == 13:
            return 'K'
        if card_num == 12:
            return 'Q'
        if card_num == 11:
            return 'J'
        if card_num == 10:
            return 'T'
        else:
            return str(int(card_num))

    def convertSuit(self, suit_num):
        if suit_num == 0:
            return 'H'
        if suit_num == 1:
            return 'D'
        if suit_num == 2:
            return 'C'
        if suit_num == 3:
            return 'S'
        return ''

    def getHand(self):
        card1 = self.convertCard(self.oh['$$pr0'])
        card2 = self.convertCard(self.oh['$$pr1'])
        suit1 = self.convertSuit(self.oh['$$ps0'])
        suit2 = self.convertSuit(self.oh['$$ps1'])
        return card1+suit1+card2+suit2

    def getBoard(self):
        if self.oh["betround"] >= 2:
            card1 = self.convertCard(self.oh['$$cr0'])
            card2 = self.convertCard(self.oh['$$cr1'])
            card3 = self.convertCard(self.oh['$$cr2'])
            suit1 = self.convertSuit(self.oh['$$cs0'])
            suit2 = self.convertSuit(self.oh['$$cs1'])
            suit3 = self.convertSuit(self.oh['$$cs2'])
        if self.oh["betround"] >= 3:
            card4 = self.convertCard(self.oh['$$cr3'])
            suit4 = self.convertSuit(self.oh['$$cs3'])
        if self.oh["betround"] >= 4:
            card5 = self.convertCard(self.oh['$$cr4'])
            suit5 = self.convertSuit(self.oh['$$cs4'])

        if self.oh["betround"] == 1:
            return ''
        elif self.oh["betround"] == 2:
            return card1+suit1+card2+suit2+card3+suit3
        elif self.oh["betround"] == 3:
            return card1+suit1+card2+suit2+card3+suit3+'/'+card4+suit4
        elif self.oh["betround"] == 4:
            return card1+suit1+card2+suit2+card3+suit3+'/'+card4+suit4+'/'+card5+suit5




#*****************************************************************************
    def calculateOpponentRaiseAmount(self):
        raiseRatio = int( self.oh['AmountToCall'] ) / ( int(self.oh['PotSize']) - int(self.oh['AmountToCall']) )
        logger.info('RaseRatio:%f' % raiseRatio)
        if raiseRatio <= 0.75:
            return 'RaiseHalfPot'
        elif raiseRatio <= 2:
            return 'RaisePot'
        else:
            return 'RaiseMax'
    
    def addOpponentAction(self, myPosition, betround, actionName):
        opponentPostion = 1 - myPosition
        if actionName == "Check" or actionName == "Call" or actionName == "CheckCall":
            action = [opponentPostion, 1, [0, 1, 2, 3, 4]]
        elif actionName == "RaiseHalfPot":
            action = [opponentPostion, 2, [0, 1, 2, 3, 4]]
        elif actionName == "RaisePot":
            action = [opponentPostion, 3, [0, 1, 2, 3, 4]]
        elif actionName == "RaiseMax":
            action = [opponentPostion, 4, [0, 1, 2, 3, 4]]
        else:
            logger.info('unknown actionName:%s' % actionName)
        logger.info('add opponent action:%s' % action)
        self.actionsAll[betround-1].append(action)
    
    def addMyAction(self, myPosition, betround, actionName):
        if actionName == "Fold":
            action = [myPosition, 0, [0, 1, 2, 3, 4]]
        elif actionName == "Check" or actionName == "Call" or actionName == "CheckCall":
            action = [myPosition, 1, [0, 1, 2, 3, 4]]
        elif actionName == "RaiseHalfPot":
            action = [myPosition, 2, [0, 1, 2, 3, 4]]
        elif actionName == "RaisePot":
            action = [myPosition, 3, [0, 1, 2, 3, 4]]
        elif actionName == "RaiseMax":
            action = [myPosition, 4, [0, 1, 2, 3, 4]]
        else:
            logger.info('unknown actionName:%s' % actionName)
        logger.info('add my action:%s' % action)
        self.actionsAll[betround-1].append(action)
    
    def generateOpponentAction(self, myPosition):
        opponentPostion = 1 - myPosition
        ### Add opponent actions to self.actionsAll
        actionName = ''
        if self.betround == 1: #pre-flop
            if self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 0:
                actionName = ''
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 1:
                if self.oh['Checks'] == 1 and self.oh['AmountToCall'] == 0.5:
                    actionName = "Check"
                elif self.oh['Calls'] == 1 and self.oh['AmountToCall'] == 0:
                    actionName = "Call"
                elif self.oh['Raises'] == 1 and self.oh['AmountToCall'] > 0:
                    actionName = self.calculateOpponentRaiseAmount()
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] > 1:
                if self.oh['AmountToCall'] == 0:
                    actionName = "Call"
                else:
                    actionName = self.calculateOpponentRaiseAmount()
            if actionName:
                self.addOpponentAction(myPosition, self.betround, actionName)
        else: #flop, turn, river
            if self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 0:
                logger.info('ActCount:0')
                #logger.info('actionsAll:%s' % self.actionsAll[self.betround-2])
                if self.actionsAll[self.betround-2][-1][0] == myPosition:
                    #last betround action not complete
                    if self.actionsAll[self.betround-2][-1][1] != 1: #My raise in previous round,
                        self.addOpponentAction(myPosition, self.betround-1, 'Check')
                    elif len(self.actionsAll[self.betround-2]) == 1: #My first call in previous round
                        self.addOpponentAction(myPosition, self.betround-1, 'Check')
                else:
                    pass
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] == 1:
                logger.info('ActCount:1')
                if self.oh['AmountToCall'] == 0:
                    if self.actionsAll[self.betround-2][-1][0] == myPosition: #My last action in this round.
                        #last betround action not complete
                        if self.actionsAll[self.betround-2][-1][1] != 1: #My raise in previous round,
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                            self.addOpponentAction(myPosition, self.betround, 'Check') #My opponent first check in this round
                        elif len(self.actionsAll[self.betround-2]) == 1: #My first call in previous round
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                            self.addOpponentAction(myPosition, self.betround, 'Check')
                        else:
                            self.addOpponentAction(myPosition, self.betround, 'Check')
                    else:
                        self.addOpponentAction(myPosition, self.betround, 'Check')
                elif self.oh['AmountToCall'] > 0:
                    if self.actionsAll[self.betround-2][-1][0] == myPosition: #My last action in this round
                        if self.oh['didchec'] + self.oh['didcall'] + self.oh['didrais'] + self.oh['didbetsize']== 0:
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                    actionName = self.calculateOpponentRaiseAmount()
                    self.addOpponentAction(myPosition, self.betround, actionName)
            else:
                logger.info('ActCount:>1')
                if self.oh['AmountToCall'] == 0:
                    self.addOpponentAction(myPosition, self.betround, 'Check')
                else:
                    actionName = self.calculateOpponentRaiseAmount()
                    self.addOpponentAction(myPosition, self.betround, actionName)
        
    

    def getDecision(self):
        """
        使用新的JSON格式与服务器通信来获取决策
        """
        import json
        import socket
        # 更新变量
        #return OpenHoldem.getSymbol("f$betpot_1_2")
        decision = 0.0
        position = -1
        self.pot = -1
        self.updateVars()
        self.rate = self.oh['bblind']/self.serverBigBlind
        # if self.oh['pot'] == 0:
        #     self.pot = self.oh['bblind'] + self.oh['sblind']
        # else:
        #     self.pot = self.oh['pot']
        self.pot = self.oh['pot']
        if self.oh['bigblindchair'] == self.oh['userchair']:
            position = 1
            self.opponentChair = self.oh['smallblindchair']
        else:
            position = 0
            self.opponentChair = self.oh['bigblindchair']
        #rlcard and alpha holdem server use different position
        position = 1 - position

        hole = self.getHand()
        board = self.getBoard()

        # logger.info(OpenHoldem.GetHandnumber())
        # handNo = int(OpenHoldem.GetHandnumber())
        ### New hand here
        # if self.lastHandNo != handNo:
        #     self.lastHandNo = handNo
        #     self.actionsAll = ''
        ### 888poker doesnt support handnumber, calc handnumber by our code.
        ### TODO: We may get 2 same hands.
        if self.lastHole != hole:
            self.lastHole = hole
            self.lastHandNo = self.lastHandNo + 1
            self.actionsAll = [[], [], [], []]  # preflop, flop, turn, river
            logger.info('\n\n')
            logger.info('Starting new handNo:%d', self.lastHandNo)
        handNo = self.lastHandNo

        logger.info('---------------------')
        logger.info('handNo:%s' % handNo)
        logger.info('betround:%s' % self.betround)
        logger.info('position:%s' % position)
        logger.info('sblind:%s' % self.oh['sblind'])
        logger.info('bblind:%s' % self.oh['bblind'])
        logger.info('MaxOpponentStackSize:%s' % self.oh['MaxOpponentStackSize'])
        logger.info('MaxStacksizeOfActiveOpponents:%s' % self.oh['MaxStacksizeOfActiveOpponents'])
        logger.info('StackSize:%s' % self.oh['StackSize'])
        logger.info('PotSize:%s' % self.oh['PotSize'])
        logger.info('hand:%s' % hole)
        logger.info('board:%s' % str(board))
        logger.info('pot:%s' % self.oh['pot'])
        logger.info('previousBetround:%d' % self.previousBetround)
        logger.info('---------------------')
        
        self.generateOpponentAction(position)
        logger.info('GenerateOpponentAction ActionsAll:%s' % self.actionsAll)

        # 获取手牌和公共牌
        hand_cards = []
        public_cards = []
        
        # 转换手牌
        if self.oh['$$pr0'] > 0:
            hand_cards.append(self.convertSuit(self.oh['$$ps0']) + self.convertCard(self.oh['$$pr0']))
        if self.oh['$$pr1'] > 0:
            hand_cards.append(self.convertSuit(self.oh['$$ps1']) + self.convertCard(self.oh['$$pr1']))
            
        # 转换公共牌
        for i in range(5):
            if self.oh[f'$$cr{i}'] > 0:
                public_cards.append(self.convertSuit(self.oh[f'$$cs{i}']) + self.convertCard(self.oh[f'$$cr{i}']))

        # 设置双方筹码量
        if self.oh['StackSize'] >= self.oh['MaxStacksizeOfActiveOpponents']:
            myStackSize = int( 2 * (self.oh['MaxStacksizeOfActiveOpponents'] + self.oh['AmountToCall']) )
            opponentStackSize = int( 2 * self.oh['MaxStacksizeOfActiveOpponents'] )
        else:
            myStackSize = int( 2 * self.oh['StackSize'] )
            opponentStackSize = int( 2 * (self.oh['StackSize'] - self.oh['AmountToCall']) )
        if position == 0:
            stakes = (myStackSize, opponentStackSize)
        else:
            stakes = (opponentStackSize, myStackSize)
        # 构建obs_dict
        obs_dict = {
            'hand_cards': hand_cards,
            'public_cards': public_cards,
            'history': self.actionsAll,
            'legal_actions': list(range(5)),  # 默认所有动作合法
            'stakes': stakes,  # 使用当前余额
            'current_player': position  # 0 SB, 1 BB
        }
        
        try:
            # 连接到服务器
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('10.164.136.173', 8888))
                
                # 发送obs_dict
                s.sendall(json.dumps(obs_dict).encode() + b'\n')
                
                # 接收响应
                response = s.recv(1024).decode().strip()
                time.sleep(1)
                # 将响应转换为决策
                if response == "FOLD":
                    self.addMyAction(position, self.betround, 'Fold')
                    return OpenHoldem.getSymbol("Fold")
                elif response == "CHECK_CALL":
                    self.addMyAction(position, self.betround, 'CheckCall')
                    return OpenHoldem.getSymbol("Call")
                elif response == "RAISE_HALF_POT":
                    self.addMyAction(position, self.betround, 'RaiseHalfPot')
                    if int(self.oh['pot']) <=  2 * int(self.oh['bblind']):
                        return OpenHoldem.getSymbol('RaisePot')
                    return OpenHoldem.getSymbol("RaiseHalfPot")
                elif response == "RAISE_POT":
                    self.addMyAction(position, self.betround, 'RaisePot')
                    if int(self.oh['pot']) <= 2* int(self.oh['bblind']):
                        return OpenHoldem.getSymbol('RaisePot')
                    return OpenHoldem.getSymbol("RaisePot")
                elif response == "ALL_IN":
                    self.addMyAction(position, self.betround, 'RaiseMax')
                    return OpenHoldem.getSymbol("RaiseMax")
                else:
                    self.addMyAction(position, self.betround, 'CheckCall')
                    return OpenHoldem.getSymbol("Call")  # 默认Call
                    
        except Exception as e:
            print(f"Error in getDecision: {e}")
            time.sleep(600)
            self.addMyAction(position, self.betround, 'CheckCall')
            return OpenHoldem.getSymbol("Call")  # 出错时默认Call