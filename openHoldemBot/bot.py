import OpenHoldem
import math
import sys
import time
#import slumbot_util
import socket
import logging
import requests
from pprint import pprint
import subprocess
import os


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
client_socket.settimeout(10)
#client_socket.connect(('10.164.142.211', 8888))

class Main:
    gotcaught = False
    ibluffed = False
    inv = -1
    phr = -1
    serverBigBlind = 20
    startStack = 1000
    smallStack = 1000
    rate = 1
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
        self.actionsAllReadable = ''
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


    def getDecision(self):
        #return OpenHoldem.getSymbol("f$betpot_1_2")
        decision = 0.0
        position = -1
        self.pot = -1
        self.updateVars()
        ### Hardcode rate to 1 ###
        #self.rate = self.oh['bblind']/self.serverBigBlind
        self.rate = 1
        # if self.oh['pot'] == 0:
        #     self.pot = self.oh['bblind'] + self.oh['sblind']
        # else:
        #     self.pot = self.oh['pot']
        self.pot = self.oh['pot']
        if self.oh['bigblindchair'] == self.oh['userchair']:
            #According to ACPC protocol, http://www.computerpokercompetition.org/downloads/documents/protocols/protocol.pdf
            position = 1 # We are SB
            self.opponentChair = self.oh['smallblindchair']
        else:
            position = 0 # We are BB
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
            self.actionsAllReadable = ''
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
                    #actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                    raiseToAmount = int( int(self.oh['PotSize']) * 0.5 + int(self.oh['AmountToCall']) )  * self.serverBigBlind
                    actions = 'r'+str(int(raiseToAmount))
            elif self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] > 1:
                if self.oh['AmountToCall'] == 0:
                    actions = 'c'
                else:
                    #actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                    raiseToAmount = int( int(self.oh['PotSize']) * 0.5 + int(self.oh['AmountToCall']) )  * self.serverBigBlind
                    actions = 'r'+str(int(raiseToAmount))
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
                        #actions = 'c/r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                        raiseToAmount = int( int(self.oh['PotSize']) * 0.5 + int(self.oh['AmountToCall']) )  * self.serverBigBlind
                        actions = 'c/r'+str(int(raiseToAmount))
                    else:
                        #actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                        raiseToAmount = int( int(self.oh['PotSize']) * 0.5 + int(self.oh['AmountToCall']) )  * self.serverBigBlind
                        actions = 'r'+str(int(raiseToAmount))
            else:
                logger.info('Act:>1')
                if self.oh['AmountToCall'] == 0:
                    actions = 'c'
                else:
                    #actions = 'r'+str(int(self.oh['BetSize'] * self.serverBigBlind))
                    raiseToAmount = int( int(self.oh['PotSize']) * 0.5 + int(self.oh['AmountToCall']) )  * self.serverBigBlind
                    actions = 'r'+str(int(raiseToAmount))

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
        # if self.betround == 1 and self.oh['Bets'] + self.oh['Calls'] + self.oh['Checks'] + self.oh['Raises'] < 2:
        #     if self.oh['balance'] > self.startStack:
        #         self.smallStack = 2 * self.startStack - self.oh['balance']
        #     else:
        #         self.smallStack = self.oh['balance']
        #     msg = str(math.floor(self.smallStack / self.rate))+';'
        
        #msg = str(self.oh['StackSize'])+';'
        
        #msg = msg + "MATCHSTATE:" + str(position) + ":" + str(handNo) + ":" + self.actionsAll + ":"
        
        msg = msg + "STATE:" + str(handNo) + ":" + self.actionsAll + ":"
        
        if position == 0: # We are SB
            msg += "|" + hole
        elif position == 1: # We are BB
            msg += hole + "|"
        if len(board) > 0:
            msg += "/" + board
        if position == 0: # We are SB
            msg += ':0|0:player1|DeepStack'
        elif position == 1: # We are BB
            msg += ':0|0:DeepStack|player1'
        msg += "\n"
        pokerstars_history = self.convert_to_pokerstars(msg)
        logger.info('sent:%s', msg.strip())
        logger.info(pokerstars_history)

        myAction = self.getLLamaFactoryResponse(pokerstars_history).strip('{"}')
        print('\n\n'+myAction+'\n\n\n')
        print('\n')
        if myAction == 'CheckCall':
            decision = OpenHoldem.getSymbol("Call")
        elif myAction == 'Call':
            decision = OpenHoldem.getSymbol("Call")
        elif myAction == 'Check':
            decision = OpenHoldem.getSymbol("Call")
        elif myAction == 'Fold':
            decision = OpenHoldem.getSymbol("Fold")
        elif myAction.startswith('Raise'):
            decision = OpenHoldem.getSymbol(myAction)
        
        
        # client_socket.send(msg.encode())
        # time_start = time.time()
        # try:
        #     advice = client_socket.recv(100).decode()
        #     logger.info('received:%s', advice)
        # except socket.timeout:
        #     logger.warn('Socket timeout')
        #     advice = 'c'
        # time_end = time.time()
        # logger.info(advice)
        # logger.info('time for betround %d:%.3f sec', self.betround, time_end - time_start)
        
        # if advice == 'c':
        #     decision = OpenHoldem.getSymbol("Call")
        # elif advice == 'f':
        #     decision = OpenHoldem.getSymbol("Fold")
        # elif advice.strip().isdigit():
        #     raiseAmount = int(advice.strip()) * self.rate
        #     logger.info('Raise:%d' % raiseAmount)

        #     if raiseAmount / self.pot <= 0.5:
        #         logger.info('RaiseHalfPot')
        #         decision = OpenHoldem.getSymbol("RaiseHalfPot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 0.67:
        #         logger.info('RaiseTwoThirdPot')
        #         decision = OpenHoldem.getSymbol("RaiseTwoThirdPot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 0.75:
        #         logger.info('RaiseThreeFourthPot')
        #         decision = OpenHoldem.getSymbol("RaiseThreeFourthPot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 1:
        #         logger.info('RaisePot')
        #         decision = OpenHoldem.getSymbol("RaisePot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 1.5:
        #         logger.info('RaisePotOneAndHalf')
        #         decision = OpenHoldem.getSymbol("RaisePot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 2:
        #         logger.info('RaisePot2')
        #         decision = OpenHoldem.getSymbol("RaisePot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 3:
        #         logger.info('RaisePot3')
        #         decision = OpenHoldem.getSymbol("RaisePot")
        #         logger.info(decision)
        #     elif raiseAmount / self.pot <= 10:
        #         logger.info('RaiseMax')
        #         logger.info(self.pot)
        #         decision = OpenHoldem.getSymbol("RaiseMax")
        #         logger.info(decision)
        #     else:
        #         logger.info('RaiseDefault:RaisePot')
        #         decision = OpenHoldem.getSymbol("RaisePot")
        # else:
        #     logger.info('Check')
        #     decision = OpenHoldem.getSymbol("Check")

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
            actions = 'r'+ str(int(self.oh['StackSize']/self.rate))
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
        logger.info('raiseRatio:%f' % raiseRatio)
        if raiseRatio <= 0.55:
            return 'RaiseHalfPot'
        elif raiseRatio <= 1.2:
            return 'RaisePot'
        else:
            return 'RaiseMax'
        
        ### Actions fit 10 action space
        # if raiseRatio <= 0.35:
            # return 'RAISE_ONETHIRD_POT'
        # elif raiseRatio <= 0.55:
            # return 'RaiseHalfPot'
        # elif raiseRatio <= 0.8:
            # return 'RAISE_THREEFOURTH_POT'
        # elif raiseRatio <= 1.2:
            # return 'RaisePot'
        # elif raiseRatio <= 1.6:
            # return 'RAISE_ONEANDHALF_POT'
        # elif raiseRatio <= 2.1:
            # return 'RAISE_TWO_POT'
        # elif raiseRatio <= 3.1:
            # return 'RAISE_THREE_POT'
        # else:
            # return 'RaiseMax'


    def addOpponentAction(self, myPosition, betround, actionName):
        betroundDict = {1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
        legalMoves = [0, 1, 2, 3, 4]
        opponentPostion = 1 - myPosition
        if actionName == "Check" or actionName == "Call" or actionName == "CheckCall":
            action = [opponentPostion, 1, legalMoves]
        elif actionName == "RaiseHalfPot":
            action = [opponentPostion, 2, legalMoves]
        elif actionName == "RaisePot":
            action = [opponentPostion, 3, legalMoves]
        elif actionName == "RaiseMax":
            action = [opponentPostion, 4, legalMoves]
        elif actionName == "RAISE_ONETHIRD_POT":
            action = [opponentPostion, 5, legalMoves]
        elif actionName == "RAISE_THREEFOURTH_POT":
            action = [opponentPostion, 6, legalMoves]
        elif actionName == "RAISE_ONEANDHALF_POT":
            action = [opponentPostion, 7, legalMoves]
        elif actionName == "RAISE_TWO_POT":
            action = [opponentPostion, 8, legalMoves]
        elif actionName == "RAISE_THREE_POT":
            action = [opponentPostion, 9, legalMoves]
        else:
            logger.info('unknown actionName:%s' % actionName)
        logger.info('add opponent action:%s' % action)
        self.actionsAll[betround-1].append(action)
        self.actionsAllReadable += 'betround %d %s, opponent action:%s\n' % (betround-1, betroundDict[betround], actionName)


    def addMyAction(self, myPosition, betround, actionName):
        betroundDict = {1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
        legalMoves = [0, 1, 2, 3, 4]
        if actionName == "Fold":
            action = [myPosition, 0, legalMoves]
        elif actionName == "Check" or actionName == "Call" or actionName == "CheckCall":
            action = [myPosition, 1, legalMoves]
        elif actionName == "RaiseHalfPot":
            action = [myPosition, 2, legalMoves]
        elif actionName == "RaisePot":
            action = [myPosition, 3, legalMoves]
        elif actionName == "RaiseMax":
            action = [myPosition, 4, legalMoves]
        elif actionName == "RAISE_ONETHIRD_POT":
            action = [myPosition, 5, legalMoves]
        elif actionName == "RAISE_THREEFOURTH_POT":
            action = [myPosition, 6, legalMoves]
        elif actionName == "RAISE_ONEANDHALF_POT":
            action = [myPosition, 7, legalMoves]
        elif actionName == "RAISE_TWO_POT":
            action = [myPosition, 8, legalMoves]
        elif actionName == "RAISE_THREE_POT":
            action = [myPosition, 9, legalMoves]
        else:
            logger.info('unknown actionName:%s' % actionName)
        logger.info('add my action:%s' % action)
        self.actionsAll[betround-1].append(action)
        self.actionsAllReadable += 'betround %d %s, my action:%s\n' % (betround-1, betroundDict[betround], actionName)


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
                    if self.actionsAll[self.betround-2][-1][0] == myPosition: #My last action in previous round.
                        #last betround action not complete
                        if self.actionsAll[self.betround-2][-1][1] != 1: #My raise in previous round,
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                        elif len(self.actionsAll[self.betround-2]) == 1: #My first call in previous round
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                    self.addOpponentAction(myPosition, self.betround, 'Check')
                elif self.oh['AmountToCall'] > 0:
                    if self.actionsAll[self.betround-2][-1][0] == myPosition: #My last action in previous round
                        #last betround action not complete
                        if self.actionsAll[self.betround-2][-1][1] != 1: #My raise in previous round,
                            self.addOpponentAction(myPosition, self.betround-1, 'Check')
                        elif len(self.actionsAll[self.betround-2]) == 1: #My first call in previous round
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


    def getChatgptResponse(self, prompt):
        prompt = 'As you are a professional Texas Holdem player, You are in a Headsup no limit holdem game\n' + prompt + \
                 'Choose my best action from "Fold", "CheckCall", "RaiseHalfPot", "RaisePot" only. Your response just json, no other text.'
        api_url = "https://api.openai.com/v1/chat/completions"  # Replace with the actual ChatGPT API endpoint
        headers = {
            "Authorization": f"Bearer sk-proj-QZRx-",  # Replace with your actual API key
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",  # Specify the model you want to use
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500  # Adjust the token limit as needed
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            print(response)
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}, {response.text}"


    def getDeepseekResponse(self, prompt):
        prompt = 'As you are a professional Texas Holdem player, You are in a Headsup no limit holdem game\n' + prompt + \
                 'Choose my best action from "Fold", "CheckCall", "RaiseHalfPot", "RaisePot", "RaseMax" only. Your response just action Name, no other text.'
        print(prompt)
        api_url = "https://api.deepseek.com/chat/completions"  # Replace with the actual DeepSeek API endpoint
        headers = {
            "Authorization": f"Bearer sk-",  # Replace with your actual API key
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2500,  # Adjust the token limit as needed
            "temperature": 0
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}, {response.text}"

    def getLLamaFactoryResponse(self, prompt):
        prompt = 'You are Deepstack, a specialist in playing heads up No Limit Texas Holdem. The following will be a game scenario and you need to make the opimal decision.\n\nHere is a game summary:\n' + prompt + \
                 '\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:'
        api_url = "http://10.0.0.129:8000/v1/chat/completions"  # Replace with the actual ChatGPT API endpoint
        headers = {
            "Authorization": f"aaa",  # Replace with your actual API key
            "Content-Type": "application/json"
        }
        data = {
            "model": "gemma3",  # Specify the model you want to use
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2408,  # Adjust the token limit as needed
            "temperature": 0
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            print(response)
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}, {response.text}"


    def convertToPokerStarsFormat(self, log_info):
        """Convert log info to PokerStars hand history format"""
        # Extract values from log_info
        handNo = log_info['handNo']
        position = int(log_info['position'])
        sblind = float(log_info['sblind'])
        bblind = float(log_info['bblind'])
        myStack = float(log_info['StackSize'])
        oppStack = float(log_info['MaxStacksizeOfActiveOpponents'])
        hand = log_info['hand']
        board = log_info['board']
        
        # Convert to PokerStars format
        history = []
        history.append(f"PokerStars Hand #{handNo}:  Hold'em No Limit (${sblind}/${bblind} USD)")
        if position == 1:  # We are SB
            history.append("Table 'ACPC Match' 2-max Seat #2 is the button")
        else:  # We are BB
            history.append("Table 'ACPC Match' 2-max Seat #1 is the button")
        history.append(f"Seat 1: Opponent (${oppStack} in chips)")
        history.append(f"Seat 2: DeepStack (${myStack} in chips)")
        
        
        # Add player info based on position
        # After position = 1 - position flip:
        # position = 0 means we are BB
        # position = 1 means we are SB
        if position == 1:  # We are SB
            history.append(f"DeepStack: posts small blind ${sblind}")
            history.append(f"Opponent: posts big blind ${bblind}")
        else:  # We are BB
            history.append(f"Opponent: posts small blind ${sblind}")
            history.append(f"DeepStack: posts big blind ${bblind}")
        
        # Add hole cards
        history.append("*** HOLE CARDS ***")
        history.append(f"Dealt to DeepStack [{hand}]")
        
        # Add action history from self.actionsAllReadable
        current_round = int(log_info['betround'])
        round_markers = ["", "*** FLOP ***", "*** TURN ***", "*** RIVER ***"]
        
        # Parse actionsAllReadable and convert to PokerStars format
        if self.actionsAllReadable:
            current_round_actions = []
            for line in self.actionsAllReadable.split('\n'):
                if not line.strip():
                    continue
                    
                # Extract betround and action info
                if 'betround' in line:
                    round_num = int(line.split()[1])
                    if round_num > current_round:
                        break
                        
                    # Add previous round's actions
                    if current_round_actions:
                        history.extend(current_round_actions)
                        current_round_actions = []
                        
                    # Add round marker and board cards if needed
                    if round_num > 0:
                        if round_num == 1:  # Flop
                            if board:
                                history.append(f"{round_markers[round_num]} [{board[:8]}]")
                        elif round_num == 2:  # Turn
                            if len(board) >= 11:
                                history.append(f"{round_markers[round_num]} [{board[:11]}]")
                        elif round_num == 3:  # River
                            if len(board) >= 14:
                                history.append(f"{round_markers[round_num]} [{board}]")
                
                # Convert action to PokerStars format
                elif 'action:' in line:
                    action_parts = line.split('action:')
                    if len(action_parts) == 2:
                        player = "DeepStack" if "my action" in line else "Opponent"
                        action = action_parts[1].strip()
                        
                        # Convert action names to PokerStars format
                        if action == "CheckCall":
                            action_str = f"{player}: checks" if "check" in line.lower() else f"{player}: calls"
                        elif action == "Fold":
                            action_str = f"{player}: folds"
                        elif "Raise" in action:
                            if "RaiseHalfPot" in action:
                                action_str = f"{player}: raises half pot"
                            elif "RaisePot" in action:
                                action_str = f"{player}: raises pot"
                            elif "RaiseMax" in action:
                                action_str = f"{player}: raises all-in"
                            else:
                                action_str = f"{player}: raises"
                        current_round_actions.append(action_str)
            
            # Add any remaining actions
            if current_round_actions:
                history.extend(current_round_actions)
        
        pprint(history)
        return "\n".join(history)
        
    def getDecision3(self):
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
            self.actionsAllReadable = ''
            logger.info('\n\n')
            logger.info('Starting new handNo:%d', self.lastHandNo)
        handNo = self.lastHandNo

        log_info = ''
        log_info += '\n---------------------\n'
        log_info += 'handNo:%s\n' % handNo
        betroundDict = {1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
        log_info += 'betround:%s\n' % betroundDict[self.betround]
        if position == 0:
            log_info += 'position:bblind\n'
        else:
            log_info += 'position:sblind\n'
        log_info += 'sblind:%s\n' % self.oh['sblind']
        log_info += 'bblind:%s\n' % self.oh['bblind']
        log_info += 'MaxOpponentStackSize:%s\n' % self.oh['MaxOpponentStackSize']
        log_info += 'MaxStacksizeOfActiveOpponents:%s bb\n' % self.oh['MaxStacksizeOfActiveOpponents']
        log_info += 'MyStackSize:%s bb\n' % self.oh['StackSize']
        log_info += 'PotSize:%s bb\n' % self.oh['PotSize']
        log_info += 'hand:%s\n' % hole
        log_info += 'board:%s\n' % str(board)
        log_info += 'pot:%s\n' % self.oh['pot']
        #log_info += 'previousBetround:%d\n' % betroundDict[self.previousBetround]
        self.generateOpponentAction(position)
        log_info += self.actionsAllReadable
        log_info += '---------------------\n'
        logger.info(log_info)
        #print(self.getChatgptResponse(log_info))
        #myAction = self.getDeepseekResponse(log_info).strip()
        pokerstars_history = self.convertToPokerStarsFormat({
            'handNo': handNo,
            'betround': self.betround,
            'position': position,
            'sblind': self.oh['sblind'],
            'bblind': self.oh['bblind'],
            'MaxOpponentStackSize': self.oh['MaxOpponentStackSize'],
            'MaxStacksizeOfActiveOpponents': self.oh['MaxStacksizeOfActiveOpponents'],
            'StackSize': self.oh['StackSize'],
            'PotSize': self.oh['PotSize'],
            'hand': hole,
            'board': board,
            'pot': self.oh['pot']
        })
        # myAction = self.getLLamaFactoryResponse(pokerstars_history).strip('{"}')
        # print('\n\n'+myAction+'\n\n\n')
        # print('\n')
        # self.addMyAction(position, self.betround, myAction)
        # if myAction == 'CheckCall':
            # return OpenHoldem.getSymbol('Call')
        # return OpenHoldem.getSymbol(myAction)
        
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
        if position == 0: #We are BB
            stakes = (myStackSize, opponentStackSize)
        else: #We are SB
            stakes = (opponentStackSize, myStackSize)
        # 构建obs_dict
        obs_dict = {
            'hand_cards': hand_cards,
            'public_cards': public_cards,
            'history': self.actionsAll,
            'legal_actions': list(range(5)),  # 默认所有动作合法
            'stakes': stakes,  # 使用当前余额
            'current_player': position  # 0 BB, 1 SB
        }
        
        try:
            # 连接到服务器
            # self.addMyAction(position, self.betround, 'Call')
            # return OpenHoldem.getSymbol("Call")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('10.0.0.207', 8888))
                
                # 发送obs_dict
                s.sendall(json.dumps(obs_dict).encode() + b'\n')
                
                # 接收响应
                response = s.recv(1024).decode().strip()
                print('\n\n' + response + '\n\n')
                time.sleep(2)
                # 将响应转换为决策
                if response == "FOLD":
                    self.addMyAction(position, self.betround, 'Fold')
                    return OpenHoldem.getSymbol("Fold")
                elif response == "CHECK_CALL":
                    self.addMyAction(position, self.betround, 'CheckCall')
                    return OpenHoldem.getSymbol("Call")
                elif response == "RAISE_HALF_POT":
                    self.addMyAction(position, self.betround, 'RaiseHalfPot')
                    return OpenHoldem.getSymbol("RaiseHalfPot")
                elif response == "RAISE_POT":
                    self.addMyAction(position, self.betround, 'RaisePot')
                    return OpenHoldem.getSymbol("RaisePot")
                elif response == "ALL_IN":
                    self.addMyAction(position, self.betround, 'RaiseMax')
                    return OpenHoldem.getSymbol("RaiseMax")
                # Newly added actions by Yang
                elif response == "RAISE_ONETHIRD_POT":
                    self.addMyAction(position, self.betround, 'RAISE_ONETHIRD_POT')
                    if self.oh['PotSize'] < 3:
                        return OpenHoldem.getSymbol("RaiseHalfPot")
                    else:
                        return int( self.oh['PotSize'] * 1/3)
                elif response == "RAISE_THREEFOURTH_POT":
                    self.addMyAction(position, self.betround, 'RAISE_THREEFOURTH_POT')
                    return int( self.oh['PotSize'] * 3/4)
                elif response == "RAISE_ONEANDHALF_POT":
                    self.addMyAction(position, self.betround, 'RAISE_ONEANDHALF_POT')
                    return int( self.oh['PotSize'] * 1.5)
                elif response == "RAISE_TWO_POT":
                    self.addMyAction(position, self.betround, 'RAISE_TWO_POT')
                    return int( self.oh['PotSize'] * 2)
                elif response == "RAISE_THREE_POT":
                    self.addMyAction(position, self.betround, 'RAISE_THREE_POT')
                    return int( self.oh['PotSize'] * 3)
                else:
                    self.addMyAction(position, self.betround, 'CheckCall')
                    return OpenHoldem.getSymbol("Call")  # 默认Call
                    
        except Exception as e:
            print(f"Error in getDecision: {e}")
            time.sleep(600)
            self.addMyAction(position, self.betround, 'CheckCall')
            return OpenHoldem.getSymbol("Call")  # 出错时默认Call


    def convert_to_pokerstars(self, msg):
        """
        Convert ACPC protocol message to PokerStars format using convert.py via subprocess
        
        Args:
            msg (str): ACPC protocol message
            
        Returns:
            str: PokerStars formatted hand history
        """
        try:
            # Get path to convert.py
            convert_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convert.py')
            
            # Prepare the command to run convert.py
            cmd = [
                'python', 
                convert_script,
                '--hero', 'DeepStack',
                '--table_name', 'AI Poker Match',
                '--small_blind', str(int(self.oh['sblind'])),
                '--big_blind', str(int(self.oh['bblind'])),
                '--stack_size', str( int( self.oh['StackSize'] * self.oh['bblind'] ))
            ]
            
            # Run the command and pass the message via stdin
            result = subprocess.run(
                cmd,
                input=msg.strip() + '\n',  # Ensure the message ends with a newline
                capture_output=True,
                text=True,
                check=True
            )
            
            # Return the converted hand history
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting hand history: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None