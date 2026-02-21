import torch
import math
import collections
import itertools
import random
import functools
import sys
import traceback
import threading
import concurrent.futures as futures
import multiprocessing as multip
import os
import resource
import numpy
from torch.nn import functional
import torch.distributed as dist
import torch.utils.data as datald
from torch.nn.parallel import DistributedDataParallel as distdp
from torch.utils.data.distributed import DistributedSampler as dsamp
import datetime
import torch.distributed.checkpoint as dcheck
from torch.distributed.fsdp import FullyShardedDataParallel as fsdp, StateDictType as sttype, FullStateDictConfig as fconfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as cosine


def setseed(seed=42):
    try:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[setseed] Seed set to {seed}")
    except Exception as err:
        raise RuntimeError(f"[setseed] Seed failed: {err}")


def redmean(val, device):
    if not dist.is_initialized():
        return val
    tval = torch.tensor(val, dtype=torch.float32, device=device)
    dist.all_reduce(tval, op=dist.ReduceOp.AVG)
    return tval.item()


class Tracker:
    def __init__(self, beta=0.999):
        self.beta = beta
        self.mean = 0.
        self.var = 1.
        self.time = 0

    def update(self, val):
        self.time += 1
        self.mean = self.beta * self.mean + (1 - self.beta) * val
        self.var = self.beta * self.var + (1 - self.beta) * ((val - self.mean) ** 2)
        denom = math.sqrt(max(self.var / (1 - self.beta ** self.time), 1e-8)) + 1e-8
        return (val - self.mean) / denom


class AstNode:
    def __init__(self, ntype, val, kids=None):
        self.ntype = ntype
        self.val = val
        self.kids = kids or []
        self.idx = -1
        self.depth = 0
        self.weight = 1


class Lexer:
    keys = {
        b'def': 1, b'return': 2, b'if': 3, b'elif': 4, b'else': 5, b'while': 6,
        b'for': 7, b'in': 8, b'break': 9, b'continue': 10, b'pass': 11, b'and': 12,
        b'or': 13, b'not': 14, b'True': 15, b'False': 16, b'None': 17, b'class': 18,
        b'yield': 19, b'try': 20, b'except': 21, b'finally': 22, b'with': 23,
        b'as': 24, b'import': 25, b'from': 26, b'global': 27, b'nonlocal': 28,
        b'assert': 29, b'del': 30, b'lambda': 31, b'is': 32, b'await': 33, b'async': 34
    }
    opers = {
        b'+': 1, b'-': 2, b'*': 3, b'/': 4, b'%': 5, b'**': 6, b'//': 7, b'==': 8,
        b'!=': 9, b'<': 10, b'>': 11, b'<=': 12, b'>=': 13, b'=': 14, b'+=': 15,
        b'-=': 16, b'*=': 17, b'/=': 18, b'//=': 19, b'%=': 20, b'**=': 21,
        b'&': 22, b'|': 23, b'^': 24, b'~': 25, b'<<': 26, b'>>': 27, b'&=': 28,
        b'|=': 29, b'^=': 30, b'<<=': 31, b'>>=': 32, b'->': 33, b':=': 34
    }
    puncts = {
        b'(': 1, b')': 2, b'[': 3, b']': 4, b'{': 5, b'}': 6, b':': 7, b',': 8,
        b'.': 9, b';': 10, b'@': 11
    }

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.length = len(data)
        self.char = self.data[self.pos] if self.length > 0 else 0
        self.indents = [0]
        self.queue = []
        self.flag = 1

    def advance(self):
        self.pos += 1
        self.char = self.data[self.pos] if self.pos < self.length else 0

    def gettok(self):
        if self.queue:
            return self.queue.pop(0)
        while self.char in (32, 9):
            self.advance()
        if self.char in (10, 13):
            self.advance()
            if self.char in (10, 13):
                return self.gettok()
            spaces = 0
            while self.char in (32, 9):
                spaces += 1 if self.char == 32 else 4
                self.advance()
            if self.char == 0:
                while len(self.indents) > 1:
                    self.indents.pop()
                    self.queue.append((5, b'', 0))
                return self.gettok()
            if spaces > self.indents[-1]:
                self.indents.append(spaces)
                return (4, b'', spaces)
            while spaces < self.indents[-1]:
                self.indents.pop()
                self.queue.append((5, b'', 0))
            if spaces != self.indents[-1]:
                self.indents.append(spaces)
            return self.gettok()
        if self.char == 0:
            while len(self.indents) > 1:
                self.indents.pop()
                self.queue.append((5, b'', 0))
            self.queue.append((0, b'', 0))
            return self.gettok()
        if self.char == 35:
            while self.char not in (10, 13, 0):
                self.advance()
            return self.gettok()
        if self.char in (34, 39):
            quote = self.char
            buf = bytearray()
            self.advance()
            while self.char != quote and self.char != 0:
                if self.char == 92:
                    self.advance()
                    if self.char in (110, 116, 114, 92, 34, 39):
                        buf.append({110: 10, 116: 9, 114: 13, 92: 92, 34: 34, 39: 39}[self.char])
                    else:
                        buf.append(92)
                        buf.append(self.char)
                else:
                    buf.append(self.char)
                self.advance()
            self.advance()
            return (10, bytes(buf), 0)
        if 65 <= self.char <= 90 or 97 <= self.char <= 122 or self.char == 95:
            buf = bytearray()
            while 65 <= self.char <= 90 or 97 <= self.char <= 122 or 48 <= self.char <= 57 or self.char == 95:
                buf.append(self.char)
                self.advance()
            bval = bytes(buf)
            return (1, bval, self.keys.get(bval, 0)) if bval in self.keys else (2, bval, 0)
        if 48 <= self.char <= 57:
            buf = bytearray()
            while 48 <= self.char <= 57 or self.char == 46:
                buf.append(self.char)
                self.advance()
            return (3, bytes(buf), 0)
        buf = bytearray([self.char])
        self.advance()
        if self.char in (61, 60, 62, 33, 43, 45, 42, 47, 38, 124, 94, 126) and bytes([buf[0], self.char]) in self.opers:
            buf.append(self.char)
            self.advance()
            if self.char == 61 and bytes([buf[0], buf[1], self.char]) in self.opers:
                buf.append(self.char)
                self.advance()
        bval = bytes(buf)
        if bval in self.opers:
            return (6, bval, self.opers[bval])
        if bval in self.puncts:
            return (7, bval, self.puncts[bval])
        return (8, buf, 0)


class Parser:
    def __init__(self, data):
        self.lexer = Lexer(data)
        self.curr = self.lexer.gettok()
        self.flag = 0

    def nextok(self):
        self.curr = self.lexer.gettok()

    def match(self, ttype, val=0):
        if self.curr[0] == ttype and (not val or self.curr[2] == val):
            res = self.curr
            self.nextok()
            return res
        return None

    def peek(self, ttype, val=0):
        if self.curr[0] == ttype and (not val or self.curr[2] == val):
            return True
        return False

    def expr(self):
        left = self.tern1()
        if self.match(1, 3):
            mid = self.tern1()
            self.match(1, 5)
            right = self.expr()
            return AstNode(41, b'', [left, mid, right])
        while self.curr[0] == 6 and self.curr[2] == 34:
            oper = self.curr
            self.nextok()
            right = self.tern1()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern1(self):
        left = self.tern2()
        while self.curr[0] == 1 and self.curr[2] == 13:
            oper = self.curr
            self.nextok()
            right = self.tern2()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern2(self):
        left = self.tern3()
        while self.curr[0] == 1 and self.curr[2] == 12:
            oper = self.curr
            self.nextok()
            right = self.tern3()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern3(self):
        if self.match(1, 14):
            return AstNode(24, b'not', [self.tern3()])
        left = self.tern4()
        while self.curr[0] == 6 and self.curr[2] in (8, 9, 10, 11, 12, 13):
            oper = self.curr
            self.nextok()
            right = self.tern4()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern4(self):
        left = self.tern5()
        while self.curr[0] == 6 and self.curr[2] == 23:
            oper = self.curr
            self.nextok()
            right = self.tern5()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern5(self):
        left = self.tern6()
        while self.curr[0] == 6 and self.curr[2] == 24:
            oper = self.curr
            self.nextok()
            right = self.tern6()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern6(self):
        left = self.tern7()
        while self.curr[0] == 6 and self.curr[2] == 22:
            oper = self.curr
            self.nextok()
            right = self.tern7()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern7(self):
        left = self.tern8()
        while self.curr[0] == 6 and self.curr[2] in (26, 27):
            oper = self.curr
            self.nextok()
            right = self.tern8()
            left = AstNode(1, oper[1], [left, right])
        return left

    def tern8(self):
        left = self.addop()
        while self.curr[0] == 6 and self.curr[2] in (1, 2):
            oper = self.curr
            self.nextok()
            right = self.addop()
            left = AstNode(1, oper[1], [left, right])
        return left

    def addop(self):
        left = self.mulop()
        while self.curr[0] == 6 and self.curr[2] in (3, 4, 5, 7):
            oper = self.curr
            self.nextok()
            right = self.mulop()
            left = AstNode(2, oper[1], [left, right])
        return left

    def mulop(self):
        if self.match(6, 2):
            return AstNode(25, b'-', [self.mulop()])
        if self.match(6, 1):
            return AstNode(25, b'+', [self.mulop()])
        if self.match(6, 25):
            return AstNode(25, b'~', [self.mulop()])
        left = self.powop()
        while self.curr[0] == 6 and self.curr[2] == 6:
            oper = self.curr
            self.nextok()
            right = self.mulop()
            left = AstNode(2, oper[1], [left, right])
        return left

    def powop(self):
        if self.match(1, 31):
            args = []
            while self.curr[0] == 2:
                args.append(AstNode(4, self.curr[1]))
                self.nextok()
                if not self.match(7, 8):
                    break
            self.match(7, 7)
            return AstNode(42, b'', args + [self.expr()])
        if self.match(7, 1):
            if self.match(7, 2):
                return AstNode(26, b'')
            node = self.expr()
            arr = [node]
            if self.match(1, 7):
                item = self.expr()
                self.match(1, 8)
                iterb = self.expr()
                cond = AstNode(0, b'')
                if self.match(1, 3):
                    cond = self.expr()
                self.match(7, 2)
                return AstNode(43, b'', [node, item, iterb, cond])
            while self.match(7, 8):
                if self.curr[0] == 7 and self.curr[2] == 2:
                    break
                arr.append(self.expr())
            self.match(7, 2)
            return AstNode(26, b'', arr) if len(arr) > 1 else node
        if self.match(7, 3):
            if self.match(7, 4):
                return AstNode(27, b'')
            node = self.expr()
            arr = [node]
            if self.match(1, 7):
                item = self.expr()
                self.match(1, 8)
                iterb = self.expr()
                cond = AstNode(0, b'')
                if self.match(1, 3):
                    cond = self.expr()
                self.match(7, 4)
                return AstNode(44, b'', [node, item, iterb, cond])
            while self.match(7, 8):
                if self.curr[0] == 7 and self.curr[2] == 4:
                    break
                arr.append(self.expr())
            self.match(7, 4)
            return AstNode(27, b'', arr)
        if self.match(7, 5):
            if self.match(7, 6):
                return AstNode(28, b'')
            kvar = self.expr()
            arr = []
            if self.match(7, 7):
                vvar = self.expr()
                arr.append(AstNode(29, b'', [kvar, vvar]))
                if self.match(1, 7):
                    item = self.expr()
                    self.match(1, 8)
                    iterb = self.expr()
                    cond = AstNode(0, b'')
                    if self.match(1, 3):
                        cond = self.expr()
                    self.match(7, 6)
                    return AstNode(45, b'', [AstNode(29, b'', [kvar, vvar]), item, iterb, cond])
                while self.match(7, 8):
                    if self.curr[0] == 7 and self.curr[2] == 6:
                        break
                    ksec = self.expr()
                    self.match(7, 7)
                    vsec = self.expr()
                    arr.append(AstNode(29, b'', [ksec, vsec]))
                self.match(7, 6)
                return AstNode(28, b'', arr)
            else:
                arr.append(kvar)
                while self.match(7, 8):
                    if self.curr[0] == 7 and self.curr[2] == 6:
                        break
                    arr.append(self.expr())
                self.match(7, 6)
                return AstNode(46, b'', arr)
        left = self.value()
        while self.curr[0] in (7, 2):
            if self.curr[0] == 2:
                self.nextok()
                arr = []
                if self.match(7, 1):
                    while self.curr[0] != 0 and not (self.curr[0] == 7 and self.curr[2] == 2):
                        arr.append(self.expr())
                        if not self.match(7, 8):
                            break
                    self.match(7, 2)
                left = AstNode(3, left.val, arr + [left])
            elif self.curr[0] == 7 and self.curr[2] == 3:
                self.nextok()
                idx1 = AstNode(0, b'')
                idx2 = AstNode(0, b'')
                idx3 = AstNode(0, b'')
                if not self.peek(7, 7):
                    idx1 = self.expr()
                if self.match(7, 7):
                    if not self.peek(7, 4) and not self.peek(7, 7):
                        idx2 = self.expr()
                    if self.match(7, 7):
                        if not self.peek(7, 4):
                            idx3 = self.expr()
                    left = AstNode(47, b'', [left, idx1, idx2, idx3])
                else:
                    left = AstNode(10, b'', [left, idx1])
                self.match(7, 4)
            elif self.curr[0] == 7 and self.curr[2] == 9:
                self.nextok()
                attr = self.match(2)
                left = AstNode(30, attr[1] if attr else b'', [left])
            else:
                break
        return left

    def value(self):
        if self.curr[0] == 2:
            ident = self.curr
            self.nextok()
            return AstNode(4, ident[1])
        if self.curr[0] == 3:
            numb = self.curr
            self.nextok()
            return AstNode(5, numb[1])
        if self.match(1, 15):
            return AstNode(5, b'True')
        if self.match(1, 16):
            return AstNode(5, b'False')
        if self.match(1, 17):
            return AstNode(5, b'None')
        if self.curr[0] == 10:
            strg = self.curr
            self.nextok()
            return AstNode(31, strg[1])
        return AstNode(0, b'')

    def stmt(self):
        decs = []
        while self.match(7, 11):
            decs.append(self.expr())
            self.match(7, 10)
        if self.match(1, 1):
            ident = self.match(2)
            self.match(7, 1)
            prms = []
            while self.curr[0] == 2:
                prms.append(AstNode(4, self.curr[1]))
                self.nextok()
                if self.match(7, 7):
                    self.expr()
                if not self.match(7, 8):
                    break
            self.match(7, 2)
            if self.match(7, 33):
                self.expr()
            self.match(7, 7)
            body = self.block()
            return AstNode(6, ident[1] if ident else b'', decs + [body] + prms)
        if self.match(1, 18):
            ident = self.match(2)
            base = []
            if self.match(7, 1):
                while self.curr[0] != 0 and not (self.curr[0] == 7 and self.curr[2] == 2):
                    base.append(self.expr())
                    if not self.match(7, 8):
                        break
                self.match(7, 2)
            self.match(7, 7)
            body = self.block()
            return AstNode(32, ident[1] if ident else b'', decs + [body] + base)
        if self.match(1, 2):
            arr = []
            if self.curr[0] != 0 and self.curr[0] != 5:
                arr.append(self.expr())
                while self.match(7, 8):
                    arr.append(self.expr())
            return AstNode(7, b'', arr)
        if self.match(1, 19):
            if self.match(1, 26):
                return AstNode(48, b'', [self.expr()])
            return AstNode(49, b'', [self.expr()])
        if self.match(1, 3):
            cond = self.expr()
            self.match(7, 7)
            blk1 = self.block()
            blk2 = AstNode(0, b'')
            while self.match(1, 4):
                csec = self.expr()
                self.match(7, 7)
                blk2 = AstNode(8, b'', [csec, self.block(), blk2])
            if self.match(1, 5):
                self.match(7, 7)
                blk2 = self.block()
            return AstNode(8, b'', [cond, blk1, blk2])
        if self.match(1, 6):
            cond = self.expr()
            self.match(7, 7)
            body = self.block()
            if self.match(1, 5):
                self.match(7, 7)
                eblk = self.block()
                return AstNode(50, b'', [cond, body, eblk])
            return AstNode(9, b'', [cond, body])
        if self.match(1, 7):
            vvar1 = self.expr()
            self.match(1, 8)
            vvar2 = self.expr()
            self.match(7, 7)
            body = self.block()
            return AstNode(33, b'', [vvar1, vvar2, body])
        if self.match(1, 20):
            self.match(7, 7)
            tblk = self.block()
            eblks = []
            while self.match(1, 21):
                exc = self.expr() if self.curr[0] != 7 else AstNode(0, b'')
                ali = AstNode(0, b'')
                if self.match(1, 24):
                    ali = self.expr()
                self.match(7, 7)
                eblks.append(AstNode(51, b'', [exc, ali, self.block()]))
            fblk = AstNode(0, b'')
            if self.match(1, 22):
                self.match(7, 7)
                fblk = self.block()
            return AstNode(52, b'', [tblk, fblk] + eblks)
        if self.match(1, 23):
            cond = self.expr()
            vvar = AstNode(0, b'')
            if self.match(1, 24):
                vvar = self.expr()
            self.match(7, 7)
            body = self.block()
            return AstNode(53, b'', [cond, vvar, body])
        if self.match(1, 9):
            return AstNode(34, b'')
        if self.match(1, 10):
            return AstNode(35, b'')
        if self.match(1, 11):
            return AstNode(36, b'')
        if self.match(1, 29):
            cond = self.expr()
            msg = AstNode(0, b'')
            if self.match(7, 8):
                msg = self.expr()
            return AstNode(54, b'', [cond, msg])
        if self.match(1, 27):
            arr = []
            while self.curr[0] == 2:
                arr.append(AstNode(4, self.curr[1]))
                self.nextok()
                if not self.match(7, 8):
                    break
            return AstNode(55, b'', arr)
        if self.match(1, 28):
            arr = []
            while self.curr[0] == 2:
                arr.append(AstNode(4, self.curr[1]))
                self.nextok()
                if not self.match(7, 8):
                    break
            return AstNode(56, b'', arr)
        if self.match(1, 30):
            return AstNode(57, b'', [self.expr()])
        if self.match(1, 25):
            arr = []
            while self.curr[0] == 2:
                arr.append(AstNode(4, self.curr[1]))
                self.nextok()
                if not self.match(7, 8):
                    break
            return AstNode(37, b'', arr)
        if self.match(1, 26):
            mod = self.expr()
            self.match(1, 25)
            arr = []
            if self.match(6, 3):
                arr.append(AstNode(4, b'*'))
            else:
                while self.curr[0] == 2:
                    arr.append(AstNode(4, self.curr[1]))
                    self.nextok()
                    if not self.match(7, 8):
                        break
            return AstNode(58, b'', [mod] + arr)
        left = self.expr()
        if self.curr[0] == 6 and self.curr[2] in range(14, 33):
            oper = self.curr
            self.nextok()
            right = self.expr()
            if self.curr[0] == 6 and self.curr[2] in range(14, 33):
                self.nextok()
                zvar = self.expr()
                return AstNode(38, b'', [left, right, zvar])
            return AstNode(11, oper[1], [left, right])
        return AstNode(12, b'', [left])

    def block(self):
        if self.match(4):
            arr = []
            while not self.match(5) and self.curr[0] != 0:
                arr.append(self.stmt())
            return AstNode(13, b'', arr)
        return AstNode(13, b'', [self.stmt()])

    def run(self):
        arr = []
        while self.curr[0] != 0:
            arr.append(self.stmt())
        return AstNode(14, b'', arr)


class Emitter:
    def __init__(self):
        self.indt = 0

    def form(self, node):
        if node.ntype == 0:
            return ""
        if node.ntype == 1:
            return f"({self.form(node.kids[0])} {node.val.decode()} {self.form(node.kids[1])})"
        if node.ntype == 2:
            return f"({self.form(node.kids[0])}{node.val.decode()}{self.form(node.kids[1])})"
        if node.ntype == 3:
            return f"{node.val.decode()}({','.join([self.form(k) for k in node.kids[:-1]])})"
        if node.ntype == 4:
            return node.val.decode()
        if node.ntype == 5:
            return node.val.decode()
        if node.ntype == 31:
            return f"'{node.val.decode()}'"
        if node.ntype == 6:
            rstr = "".join([f"{' ' * self.indt}@{self.form(k)}\n" for k in node.kids[:-len(node.kids) + 1]])
            rstr += f"{' ' * self.indt}def {node.val.decode()}({','.join([self.form(k) for k in node.kids[2:]])}):\n"
            self.indt += 1
            rstr += self.form(node.kids[1])
            self.indt -= 1
            return rstr
        if node.ntype == 7:
            return f"{' ' * self.indt}return {','.join([self.form(k) for k in node.kids])}\n"
        if node.ntype == 8:
            rstr = f"{' ' * self.indt}if {self.form(node.kids[0])}:\n"
            self.indt += 1
            rstr += self.form(node.kids[1])
            self.indt -= 1
            if node.kids[2].ntype != 0:
                rstr += f"{' ' * self.indt}else:\n"
                self.indt += 1
                rstr += self.form(node.kids[2])
                self.indt -= 1
            return rstr
        if node.ntype == 9:
            rstr = f"{' ' * self.indt}while {self.form(node.kids[0])}:\n"
            self.indt += 1
            rstr += self.form(node.kids[1])
            self.indt -= 1
            return rstr
        if node.ntype == 10:
            return f"{self.form(node.kids[0])}[{self.form(node.kids[1])}]"
        if node.ntype == 11:
            return f"{' ' * self.indt}{self.form(node.kids[0])} {node.val.decode()} {self.form(node.kids[1])}\n"
        if node.ntype == 12:
            return f"{' ' * self.indt}{self.form(node.kids[0])}\n"
        if node.ntype == 13:
            return "".join([self.form(k) for k in node.kids])
        if node.ntype == 14:
            return "".join([self.form(k) for k in node.kids])
        if node.ntype == 24:
            return f"(not {self.form(node.kids[0])})"
        if node.ntype == 25:
            return f"({node.val.decode()}{self.form(node.kids[0])})"
        if node.ntype == 26:
            return f"({','.join([self.form(k) for k in node.kids])})"
        if node.ntype == 27:
            return f"[{','.join([self.form(k) for k in node.kids])}]"
        if node.ntype == 28:
            return f"{{{','.join([self.form(k) for k in node.kids])}}}"
        if node.ntype == 29:
            return f"{self.form(node.kids[0])}:{self.form(node.kids[1])}"
        if node.ntype == 30:
            return f"{self.form(node.kids[0])}.{node.val.decode()}"
        if node.ntype == 32:
            rstr = "".join([f"{' ' * self.indt}@{self.form(k)}\n" for k in node.kids[:-len(node.kids) + 1]])
            rstr += f"{' ' * self.indt}class {node.val.decode()}({','.join([self.form(k) for k in node.kids[2:]])}):\n"
            self.indt += 1
            rstr += self.form(node.kids[1])
            self.indt -= 1
            return rstr
        if node.ntype == 33:
            rstr = f"{' ' * self.indt}for {self.form(node.kids[0])} in {self.form(node.kids[1])}:\n"
            self.indt += 1
            rstr += self.form(node.kids[2])
            self.indt -= 1
            return rstr
        if node.ntype == 34:
            return f"{' ' * self.indt}break\n"
        if node.ntype == 35:
            return f"{' ' * self.indt}continue\n"
        if node.ntype == 36:
            return f"{' ' * self.indt}pass\n"
        if node.ntype == 37:
            return f"{' ' * self.indt}import {','.join([self.form(k) for k in node.kids])}\n"
        if node.ntype == 38:
            return f"{' ' * self.indt}{self.form(node.kids[0])}={self.form(node.kids[1])}={self.form(node.kids[2])}\n"
        if node.ntype == 41:
            return f"({self.form(node.kids[0])} if {self.form(node.kids[1])} else {self.form(node.kids[2])})"
        if node.ntype == 42:
            return f"(lambda {','.join([self.form(k) for k in node.kids[:-1]])}: {self.form(node.kids[-1])})"
        if node.ntype == 43:
            return f"({self.form(node.kids[0])} for {self.form(node.kids[1])} in {self.form(node.kids[2])}{' if ' + self.form(node.kids[3]) if node.kids[3].ntype != 0 else ''})"
        if node.ntype == 44:
            return f"[{self.form(node.kids[0])} for {self.form(node.kids[1])} in {self.form(node.kids[2])}{' if ' + self.form(node.kids[3]) if node.kids[3].ntype != 0 else ''}]"
        if node.ntype == 45:
            return f"{{{self.form(node.kids[0])} for {self.form(node.kids[1])} in {self.form(node.kids[2])}{' if ' + self.form(node.kids[3]) if node.kids[3].ntype != 0 else ''}}}"
        if node.ntype == 46:
            return f"{{{','.join([self.form(k) for k in node.kids])}}}"
        if node.ntype == 47:
            return f"{self.form(node.kids[0])}[{self.form(node.kids[1])}:{self.form(node.kids[2])}:{self.form(node.kids[3])}]"
        if node.ntype == 48:
            return f"{' ' * self.indt}yield from {self.form(node.kids[0])}\n"
        if node.ntype == 49:
            return f"{' ' * self.indt}yield {self.form(node.kids[0])}\n"
        if node.ntype == 50:
            rstr = f"{' ' * self.indt}while {self.form(node.kids[0])}:\n"
            self.indt += 1
            rstr += self.form(node.kids[1])
            self.indt -= 1
            rstr += f"{' ' * self.indt}else:\n"
            self.indt += 1
            rstr += self.form(node.kids[2])
            self.indt -= 1
            return rstr
        if node.ntype == 52:
            rstr = f"{' ' * self.indt}try:\n"
            self.indt += 1
            rstr += self.form(node.kids[0])
            self.indt -= 1
            for excp in node.kids[2:]:
                rstr += f"{' ' * self.indt}except {self.form(excp.kids[0])}{' as ' + self.form(excp.kids[1]) if excp.kids[1].ntype != 0 else ''}:\n"
                self.indt += 1
                rstr += self.form(excp.kids[2])
                self.indt -= 1
            if node.kids[1].ntype != 0:
                rstr += f"{' ' * self.indt}finally:\n"
                self.indt += 1
                rstr += self.form(node.kids[1])
                self.indt -= 1
            return rstr
        if node.ntype == 53:
            rstr = f"{' ' * self.indt}with {self.form(node.kids[0])}{' as ' + self.form(node.kids[1]) if node.kids[1].ntype != 0 else ''}:\n"
            self.indt += 1
            rstr += self.form(node.kids[2])
            self.indt -= 1
            return rstr
        if node.ntype == 54:
            return f"{' ' * self.indt}assert {self.form(node.kids[0])}{',' + self.form(node.kids[1]) if node.kids[1].ntype != 0 else ''}\n"
        if node.ntype == 55:
            return f"{' ' * self.indt}global {','.join([self.form(k) for k in node.kids])}\n"
        if node.ntype == 56:
            return f"{' ' * self.indt}nonlocal {','.join([self.form(k) for k in node.kids])}\n"
        if node.ntype == 57:
            return f"{' ' * self.indt}del {self.form(node.kids[0])}\n"
        if node.ntype == 58:
            return f"{' ' * self.indt}from {self.form(node.kids[0])} import {','.join([self.form(k) for k in node.kids[1:]])}\n"
        return ""


class Graph:
    def __init__(self, root):
        self.root = root
        self.verts = []
        self.edges = []
        self.build(self.root, 0)
        self.numv = len(self.verts)
        self.adjac = torch.zeros(self.numv, self.numv)
        self.degre = torch.zeros(self.numv, self.numv)
        for uidx, vidx in self.edges:
            self.adjac[uidx, vidx] = self.adjac[vidx, uidx] = 1.0
            self.degre[uidx, uidx] += 1.0
            self.degre[vidx, vidx] += 1.0
        self.lapla = self.degre - self.adjac

    def build(self, node, depth):
        node.idx = len(self.verts)
        node.depth = depth
        self.verts.append(node)
        for kid in node.kids:
            self.build(kid, depth + 1)
            self.edges.append((node.idx, kid.idx))

    def solve(self):
        if self.numv == 0:
            return 0.0, 0.0, torch.zeros(1)
        bmat1 = torch.zeros(self.numv, max(1, len(self.edges)))
        for idx, (uidx, vidx) in enumerate(self.edges):
            bmat1[uidx, idx] = 1.0
            bmat1[vidx, idx] = -1.0
        lmat0 = bmat1 @ bmat1.T
        lmat1 = bmat1.T @ bmat1
        eval0 = torch.linalg.eigvalsh(lmat0 + torch.eye(self.numv) * 1e-6)
        eval1 = torch.linalg.eigvalsh(lmat1 + torch.eye(max(1, len(self.edges))) * 1e-6)
        bnum0 = torch.sum(eval0 < 1e-5).float()
        bnum1 = torch.sum(eval1 < 1e-5).float()
        return bnum0, bnum1, eval0


class Feature:
    def __init__(self):
        self.fname = b''
        self.accs = []

    def walk(self, node):
        if node.ntype == 6:
            self.fname = node.val
        if node.ntype == 3 and node.val == self.fname:
            vala, valb, valc = 1.0, 1.0, 0.0
            if len(node.kids) > 0:
                for arg in node.kids:
                    if arg.ntype == 1 and len(arg.kids) > 1 and arg.kids[1].ntype == 5:
                        if arg.val == b'/':
                            valb = float(arg.kids[1].val)
                        elif arg.val == b'-':
                            valc = float(arg.kids[1].val)
                    if arg.ntype == 2 and len(arg.kids) > 1 and arg.kids[1].ntype == 5:
                        if arg.val == b'/':
                            valb = float(arg.kids[1].val)
                        elif arg.val == b'-':
                            valc = float(arg.kids[1].val)
            self.accs.append((vala, valb, valc))
        for kid in node.kids:
            self.walk(kid)

    def extract(self, node):
        self.walk(node)
        if not self.accs:
            return torch.tensor([1.0, 1.0, 0.0, 0.0], requires_grad=True)
        asum = sum([itm[0] for itm in self.accs])
        bmax = max([itm[1] for itm in self.accs])
        return torch.tensor([float(len(self.accs)), asum, bmax, max([itm[2] for itm in self.accs])], requires_grad=True)


class Solver:
    def __init__(self):
        self.iters = 25

    def search(self, vect):
        pval = torch.full((vect.shape[0],) if vect.dim() > 1 else (), 2.0, device=vect.device)
        for _ in range(self.iters):
            pval = pval.detach().requires_grad_(True)
            v0, v1, v2, v3 = torch.abs(vect[..., 0]) + 1e-4, torch.abs(vect[..., 1]) + 1e-4, torch.abs(vect[..., 2]) + 1e-4, torch.abs(vect[..., 3]) + 1e-4
            yval = v0 * (v2 ** (-pval)) + v1 * (v3 ** (-pval)) - 1.0
            grad = torch.autograd.grad(yval.sum(), pval, create_graph=True)[0]
            dval = torch.clamp(yval / (grad + 1e-9), -0.5, 0.5)
            pval = torch.clamp(pval - dval, -1.0, 5.0)
        return pval.detach()


def measure(stra, strb):
    if not isinstance(stra, (list, tuple)) or not isinstance(strb, (list, tuple)):
        return max(0.0, 1.0 - abs(stra - strb) / (abs(strb) + 1e-5)) if isinstance(stra, (int, float)) and isinstance(strb, (int, float)) else float(stra == strb)
    nlen, mlen = len(stra), len(strb)
    dmat = [[0] * (mlen + 1) for _ in range(nlen + 1)]
    for i in range(nlen + 1):
        dmat[i][0] = i
    for j in range(mlen + 1):
        dmat[0][j] = j
    for i in range(1, nlen + 1):
        for j in range(1, mlen + 1):
            cost = 0 if stra[i - 1] == strb[j - 1] else 1
            dmat[i][j] = min(dmat[i - 1][j] + 1, dmat[i][j - 1] + 1, dmat[i - 1][j - 1] + cost)
    return max(0.0, 1.0 - dmat[nlen][mlen] / max(1, max(nlen, mlen)))


def execute(code, ins, outq):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        built = {'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict, 'enumerate': enumerate, 'float': float, 'int': int, 'len': len, 'list': list, 'map': map, 'max': max, 'min': min, 'range': range, 'set': set, 'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip}
        globs = {'__builtins__': built}
        exec(code, globs, globs)
        func = next((val for key, val in globs.items() if callable(val)), None)
        if not func:
            outq.put(0.0)
            return
        score = 0.0
        for arg, targ in ins:
            try:
                if isinstance(arg, tuple):
                    rout = func(*arg)
                else:
                    rout = func(arg)
                score += measure(rout, targ)
            except:
                pass
        outq.put(float(score) / len(ins))
    except:
        outq.put(0.0)


class Sandbox:
    def test(self, code, tcase):
        raise RuntimeError("mp_disabled_for_ddp")


class RMSNorm(torch.nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.wmat = torch.nn.Parameter(torch.ones(dims))

    def forward(self, xmat):
        return self.wmat * xmat * torch.rsqrt(xmat.pow(2).mean(-1, keepdim=True) + self.eps)


class Rotary(torch.nn.Module):
    def __init__(self, dims, base=10000):
        super().__init__()
        self.dims = dims
        self.base = base
        self.idxs = torch.arange(0, dims, 2).float()

    def forward(self, xmat):
        bnum, lnum, hnum, dnum = xmat.shape
        pmat = torch.arange(lnum, device=xmat.device).float().unsqueeze(1)
        tmat = pmat / (self.base ** (self.idxs.to(xmat.device) / dnum))
        smat, cmat = torch.sin(tmat), torch.cos(tmat)
        xmat0, xmat1 = xmat[..., 0::2], xmat[..., 1::2]
        return torch.stack([xmat0 * cmat - xmat1 * smat, xmat0 * smat + xmat1 * cmat], dim=-1).flatten(-2)


class Attn(torch.nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.xdim = dims // heads
        self.qmat = torch.nn.Linear(dims, dims, bias=False)
        self.kmat = torch.nn.Linear(dims, dims, bias=False)
        self.vmat = torch.nn.Linear(dims, dims, bias=False)
        self.omat = torch.nn.Linear(dims, dims, bias=False)
        self.rmat = Rotary(self.xdim)
        for wgts in [self.qmat, self.kmat, self.vmat, self.omat]:
            torch.nn.init.orthogonal_(wgts.weight)

    def forward(self, xmat, mask):
        bnum, lnum, _ = xmat.shape
        qvec = self.qmat(xmat).view(bnum, lnum, self.heads, self.xdim)
        kvec = self.kmat(xmat).view(bnum, lnum, self.heads, self.xdim)
        vvec = self.vmat(xmat).view(bnum, lnum, self.heads, self.xdim)
        qvec = self.rmat(qvec).transpose(1, 2)
        kvec = self.rmat(kvec).transpose(1, 2)
        vvec = vvec.transpose(1, 2)
        amask = mask.masked_fill(mask == 0, float('-inf')) if mask is not None else None
        return self.omat(functional.scaled_dot_product_attention(qvec, kvec, vvec, attn_mask=amask).transpose(1, 2).reshape(bnum, lnum, self.dims))


class MLP(torch.nn.Module):
    def __init__(self, dims, expns):
        super().__init__()
        self.lmat1 = torch.nn.Linear(dims, expns, bias=False)
        self.lmat2 = torch.nn.Linear(dims, expns, bias=False)
        self.lmat3 = torch.nn.Linear(expns, dims, bias=False)
        for wgts in [self.lmat1, self.lmat2, self.lmat3]:
            torch.nn.init.orthogonal_(wgts.weight)

    def forward(self, xmat):
        return self.lmat3(functional.silu(self.lmat1(xmat)) * self.lmat2(xmat))


class MoE(torch.nn.Module):
    def __init__(self, dims, expns, nexp=4, kexp=2):
        super().__init__()
        self.nexp = nexp
        self.kexp = kexp
        self.gmat = torch.nn.Linear(dims, nexp, bias=False)
        self.emats = torch.nn.ModuleList([MLP(dims, expns) for _ in range(nexp)])

    def forward(self, xmat):
        bnum, lnum, dnum = xmat.shape
        gvec = self.gmat(xmat)
        vvec, ivec = torch.topk(gvec, self.kexp, dim=-1)
        vvec = functional.softmax(vvec - vvec.max(dim=-1, keepdim=True)[0], dim=-1)
        ymat = torch.zeros_like(xmat)
        fmatx = xmat.reshape(-1, dnum)
        fmati = ivec.reshape(-1, self.kexp)
        fmatv = vvec.reshape(-1, self.kexp)
        for exprt in range(self.nexp):
            mmask = (fmati == exprt)
            if mmask.any():
                idxes = mmask.nonzero(as_tuple=False)[:, 0]
                exinp = fmatx[idxes]
                exout = self.emats[exprt](exinp)
                wgts = fmatv[mmask].view(-1)
                ymat.view(-1, dnum)[idxes] += exout * wgts.unsqueeze(-1)
        return ymat


class Block(torch.nn.Module):
    def __init__(self, dims, heads, expns):
        super().__init__()
        self.amat = Attn(dims, heads)
        self.fmat = MoE(dims, expns)
        self.nmat1 = RMSNorm(dims)
        self.nmat2 = RMSNorm(dims)

    def forward(self, xmat, mask):
        return xmat + self.fmat(self.nmat2(xmat + self.amat(self.nmat1(xmat), mask)))


class Model(torch.nn.Module):
    def __init__(self, vnum, dims, heads, expns, lnum):
        super().__init__()
        self.vnum = vnum
        self.dims = dims
        self.embd = torch.nn.Embedding(vnum, dims)
        self.blks = torch.nn.ModuleList([Block(dims, heads, expns) for _ in range(lnum)])
        self.norm = RMSNorm(dims)
        self.omat = torch.nn.Linear(dims, vnum, bias=False)

    def forward(self, xmat):
        bnum, lnum = xmat.shape
        mask = torch.tril(torch.ones(lnum, lnum, device=xmat.device)).view(1, 1, lnum, lnum)
        xmat = self.embd(xmat)
        for blck in self.blks:
            xmat = blck(xmat, mask)
        return self.omat(self.norm(xmat))

    def gen(self, xmat, klen=64):
        bnum, ddev = xmat.shape[0], xmat.device
        rmat = torch.empty((bnum, 0), dtype=torch.long, device=ddev)
        pmat = torch.zeros(bnum, device=ddev)
        for _ in range(klen):
            lmat = self(torch.cat([xmat, rmat], dim=1))[:, -1, :]
            cmat = torch.distributions.Categorical(logits=lmat)
            vmat = cmat.sample()
            rmat = torch.cat([rmat, vmat.unsqueeze(1)], dim=1)
            pmat += torch.gather(functional.log_softmax(lmat - lmat.max(dim=-1, keepdim=True)[0], dim=-1), -1, vmat.unsqueeze(-1)).squeeze(-1)
        return rmat, pmat


class Encoder(torch.nn.Module):
    def __init__(self, vnum, dims):
        super().__init__()
        self.emat = torch.nn.Linear(vnum, dims, bias=False)
        self.cmat1 = torch.nn.Conv1d(dims, dims, 3, padding=1)
        self.cmat2 = torch.nn.Conv1d(dims, dims, 3, padding=1)
        self.qmat = torch.nn.Linear(dims, dims // 2)
        self.kmat = torch.nn.Linear(dims, dims // 2)
        self.amat = torch.nn.Sequential(torch.nn.Linear(dims, dims), torch.nn.ReLU(), torch.nn.Linear(dims, 4), torch.nn.Softplus())

    def forward(self, ysamp, tdim=122):
        xmat = self.emat(ysamp).transpose(1, 2)
        xmat = functional.relu(self.cmat1(xmat))
        xmat = functional.relu(self.cmat2(xmat)).transpose(1, 2)
        qvec, kvec = self.qmat(xmat), self.kmat(xmat)
        qkvec = qvec @ kvec.transpose(-2, -1) / math.sqrt(max(qvec.size(-1), 1e-6))
        amat = functional.softmax(qkvec - qkvec.max(dim=-1, keepdim=True)[0], dim=-1)
        dvec = torch.diag_embed(amat.sum(dim=-1))
        lvec = dvec - amat
        evals = torch.linalg.eigvalsh(lvec + torch.eye(lvec.size(-1), device=lvec.device) * 1e-5)
        evals = functional.normalize(evals[:, :tdim], dim=-1)
        afmat = self.amat(xmat.mean(dim=1)) + 1e-3
        return evals, afmat


class Topol:
    def __init__(self, dims=[2, 2, 2, 2]):
        self.dims = dims
        self.nodes = dims[0] * dims[1] * dims[2] * dims[3]
        self.graph = [[] for _ in range(self.nodes)]
        for xval0, xval1, xval2, xval3 in itertools.product(range(dims[0]), range(dims[1]), range(dims[2]), range(dims[3])):
            uval = xval0 * dims[1] * dims[2] * dims[3] + xval1 * dims[2] * dims[3] + xval2 * dims[3] + xval3
            for jdx, djdx in enumerate(dims):
                vval = list((xval0, xval1, xval2, xval3))
                vval[jdx] = (vval[jdx] + 1) % djdx
                vidx = vval[0] * dims[1] * dims[2] * dims[3] + vval[1] * dims[2] * dims[3] + vval[2] * dims[3] + vval[3]
                self.graph[uval].append(vidx)
                self.graph[vidx].append(uval)
        self.spath = torch.zeros(self.nodes, self.nodes) + float('inf')
        for idx in range(self.nodes):
            self.spath[idx, idx] = 0
        for uidx in range(self.nodes):
            for vidx in self.graph[uidx]:
                self.spath[uidx, vidx] = 1
        for kdx in range(self.nodes):
            for idx in range(self.nodes):
                for jdx in range(self.nodes):
                    if self.spath[idx, jdx] > self.spath[idx, kdx] + self.spath[kdx, jdx]:
                        self.spath[idx, jdx] = self.spath[idx, kdx] + self.spath[kdx, jdx]

    def link(self, uidx, vidx, szval):
        return self.spath[uidx, vidx] * 0.5 + szval * 0.01 + random.random() * 0.1


class Critic(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.nmat = torch.nn.Sequential(torch.nn.Linear(dims, dims * 4), torch.nn.Mish(), torch.nn.Linear(dims * 4, dims * 4), torch.nn.Mish(), torch.nn.Linear(dims * 4, 1))

    def forward(self, xmat):
        return self.nmat(xmat)


class Custom:
    def __init__(self, prms, lrval=1e-4, alph=0.01):
        self.prms = list(prms)
        self.lrval = lrval
        self.alph = alph
        self.mmat = [torch.zeros_like(prm) for prm in self.prms]
        self.vmat = [torch.zeros_like(prm) for prm in self.prms]
        self.time = 0
        self.topol = Topol()

    def zero(self):
        for prm in self.prms:
            if prm.grad is not None:
                prm.grad = None

    def step(self):
        self.time += 1
        uval = random.randint(0, 15)
        vval = random.randint(0, 15)
        for idx, prm in enumerate(self.prms):
            if prm.grad is None:
                continue
            grad = prm.grad
            if dist.is_initialized() and not hasattr(prm, "_fsdp_flattened"):
                try:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                except:
                    pass
            ltval = self.topol.link(uval, vval, grad.numel() / (1024 * 1024))
            nval = grad.norm()
            dval = math.sqrt(prm.numel())
            nzval = torch.randn_like(grad) * (self.alph * nval / (dval + 1e-9))
            grad = (grad + nzval) * (1.0 / (1.0 + ltval * 0.001))
            self.mmat[idx] = 0.9 * self.mmat[idx] + 0.1 * grad
            self.vmat[idx] = 0.999 * self.vmat[idx] + 0.001 * (grad ** 2)
            mhval = self.mmat[idx] / (1 - 0.9 ** self.time)
            vhval = self.vmat[idx] / (1 - 0.999 ** self.time)
            try:
                with torch.no_grad():
                    prm.copy_(prm - self.lrval * mhval / (torch.sqrt(torch.clamp(vhval, min=1e-6)) + 1e-9))
            except:
                pass


def hpdist(xmat, ymat):
    nmatx = torch.clamp(xmat.norm(2, -1, True) ** 2, max=0.999)
    nmaty = torch.clamp(ymat.norm(2, -1, True) ** 2, max=0.999)
    return torch.acosh(torch.clamp(1 + 2 * (xmat - ymat).norm(2, -1) ** 2 / (torch.clamp((1 - nmatx) * (1 - nmaty), min=1e-6) + 1e-7), min=1.0001))


def extwgt(vval):
    bdata = vval.detach().to(dtype=torch.uint8).cpu().numpy().tobytes()
    astnd = Parser(bdata).run()
    hgrph = Graph(astnd)
    bnum0, bnum1, evals = hgrph.solve()
    fvals = Feature().extract(astnd)
    return bnum0, bnum1, evals, fvals, astnd


def extra(vval, ttype):
    return vval


def rnode(xmat, astnd, tcase):
    return xmat


class Trainer:
    def __init__(self, vnum=256, dims=256, heads=8, expns=1024, lnum=8):
        self.vnum = vnum
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            if dist.is_initialized():
                torch.cuda.set_device(self.rank)
                ddev = torch.device(f"cuda:{self.rank}")
            else:
                ddev = torch.device("cuda:0")
        else:
            ddev = torch.device("cpu")
        self.model = torch.compile(Model(vnum, dims, heads, expns, lnum).to(ddev))
        self.crit = torch.compile(Critic(128).to(ddev))
        self.encod = torch.compile(Encoder(vnum, 128).to(ddev))
        if dist.is_initialized():
            from torch.distributed.fsdp import MixedPrecision as mprec
            mtype = mprec(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
            didx = self.rank if torch.cuda.is_available() else None
            self.model = fsdp(self.model, device_id=didx, sync_module_states=True, mixed_precision=mtype)
            self.crit = fsdp(self.crit, device_id=didx, sync_module_states=True, mixed_precision=mtype)
            self.encod = fsdp(self.encod, device_id=didx, sync_module_states=True, mixed_precision=mtype)
        self.mopt = Custom(list(self.model.parameters()) + list(self.encod.parameters()), lrval=5e-5, alph=0.1)
        self.qopt = Custom(self.crit.parameters(), lrval=2e-4, alph=0.0)
        self.solve = Solver()
        self.wtv, self.wtc, self.wts, self.wtw = Tracker(), Tracker(), Tracker(), Tracker()

    def save(self, pth, epc, sched=None):
        if self.rank == 0:
            os.makedirs(pth, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        try:
            from torch.distributed.fsdp import ShardedStateDictConfig as sdconf
            with fsdp.state_dict_type(self.model, sttype.SHARDED_STATE_DICT, sdconf(offload_to_cpu=True)):
                cstat = self.model.state_dict()
            cpt = {"model_state": cstat, "optimizer_state": self.mopt.mmat, "scheduler_state": sched.state_dict() if sched else None, "epoch": epc}
            dcheck.save(cpt, checkpoint_id=pth)
        except Exception as err:
            raise RuntimeError(f"[_re] Checkpoint save failed: {err}")

    def load(self, pth, sched=None):
        try:
            from torch.distributed.fsdp import ShardedStateDictConfig as sdconf
            with fsdp.state_dict_type(self.model, sttype.SHARDED_STATE_DICT, sdconf(offload_to_cpu=True)):
                cstat = self.model.state_dict()
                cpt = {"model_state": cstat, "optimizer_state": self.mopt.mmat, "scheduler_state": sched.state_dict() if sched else None, "epoch": 0}
                dcheck.load(cpt, checkpoint_id=pth)
                self.model.load_state_dict(cpt["model_state"])
                self.mopt.mmat = cpt["optimizer_state"]
                if sched and cpt["scheduler_state"]:
                    sched.load_state_dict(cpt["scheduler_state"])
                return cpt["epoch"]
        except Exception as err:
            raise RuntimeError(f"[_re] Checkpoint load failed: {err}")

    def trace(self, pth="m.pt", dshpe=(1, 128)):
        try:
            spcfg = fconfig(offload_to_cpu=True, rank0_only=True)
            with fsdp.state_dict_type(self.model, sttype.FULL_STATE_DICT, spcfg):
                cstat = self.model.state_dict()
            if self.rank == 0:
                umod = Model(self.vnum, 256, 8, 1024, 8)
                umod.load_state_dict(cstat)
                umod.eval()
                inpt = torch.randint(0, self.vnum, dshpe)
                if inpt.shape != dshpe:
                    print(f"[_re] JIT input shape mismatch: expected {dshpe}, got {inpt.shape}")
                tmod = torch.jit.trace(umod, inpt)
                tmod.save(pth)
        except Exception as err:
            print(f"[_re] JIT trace failed: {err}")

    def train(self, dldr, smpl, epchs=100, accum=4, verbs=False):
        lvc, lvv, lvs, lvw = 0., 0., 0., 0.
        sched = cosine(torch.optim.AdamW(self.model.parameters(), lr=1e-4), T_0=10, T_mult=2)
        for epc in range(epchs):
            if smpl:
                smpl.set_epoch(epc)
            if epc > 5:
                lvv = 0.2
            if epc > 15:
                lvc = 0.4
            if epc > 25:
                lvs = 0.4
                lvw = 0.6
            self.mopt.zero()
            self.qopt.zero()
            for step, (xmat, ymat, cmat, tcase) in enumerate(dldr):
                if not isinstance(xmat, torch.Tensor) or not isinstance(ymat, torch.Tensor):
                    raise TypeError("[_re] Expected torch.Tensor")
                for _ in range(3):
                    try:
                        ddev = torch.device(f"cuda:{self.rank}")
                        if xmat.device != ddev:
                            xmat = xmat.to(ddev)
                        if ymat.device != ddev:
                            ymat = ymat.to(ddev)
                        if cmat.device != ddev:
                            cmat = cmat.to(ddev)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            try:
                                lgits = self.model(xmat)
                            except RuntimeError as err:
                                if 'out of memory' in str(err).lower():
                                    torch.cuda.empty_cache()
                                    lgits = self.model(xmat)
                                else:
                                    raise err
                            mloss = functional.cross_entropy(lgits.view(-1, lgits.size(-1)), ymat.view(-1))
                            bnum, lnum, vnum = lgits.shape
                            ysamp = functional.gumbel_softmax(lgits - lgits.max(dim=-1, keepdim=True)[0], tau=0.5, hard=True)
                            yhard = ysamp.argmax(dim=-1)
                            esig, eabf, eaout = extra(ymat, None)
                            gsig, gabf, gaout = extra(yhard, None)
                            egsig, absig = self.encod(ysamp, gsig[:, 6:].shape[1])
                            targs = functional.normalize(gsig[:, 6:].detach(), dim=-1)
                            closs = functional.mse_loss(absig, gabf.detach()) + functional.mse_loss(egsig, targs)
                            if lvw > 0:
                                for _ in range(5):
                                    gdtch = gsig.detach().requires_grad_(True)
                                    try:
                                        creal, cfake = self.crit(esig), self.crit(gdtch)
                                    except RuntimeError as err:
                                        if 'out of memory' in str(err).lower():
                                            torch.cuda.empty_cache()
                                            creal, cfake = self.crit(esig), self.crit(gdtch)
                                        else:
                                            raise err
                                    grads = torch.autograd.grad(cfake.sum(), gdtch, create_graph=True)[0]
                                    gpen = ((grads.norm(2, 1) - 1) ** 2).mean()
                                    qloss = -(creal.mean() - cfake.mean()) + 10. * gpen
                                    try:
                                        qloss.backward()
                                    except Exception as err:
                                        print(f"[_re] backward failed: {err}")
                                    try:
                                        with torch.no_grad():
                                            torch.nn.utils.clip_grad_norm_(self.crit.parameters(), 1.0)
                                            self.qopt.step()
                                            self.qopt.zero()
                                    except Exception as err:
                                        print(f"[_re] optim failed: {err}")
                            rval = rnode(yhard, gaout, tcase)
                            rcost = -torch.abs(self.solve.search(absig) - cmat)
                            sfeat = torch.zeros(bnum, 128, device=xmat.device)
                            sfeat[:, 0] = gsig[:, 0]
                            sfeat[:, 1] = gsig[:, 1]
                            sfeat[:, 2:6] = absig * 100.0
                            szval = min(122, egsig.size(1))
                            sfeat[:, 6:6 + szval] = egsig[:, :szval]
                            sfeat[:, 6:] = functional.normalize(sfeat[:, 6:].clone(), p=2, dim=-1) * 10.0
                            rscor = -hpdist(functional.normalize(sfeat, p=2, dim=-1), esig)
                            rwght = torch.clamp(self.crit(functional.normalize(sfeat, p=2, dim=-1)).squeeze(), -10.0, 10.0) if lvw > 0 else torch.zeros_like(rval)
                            twght = lvv * self.wtv.update(rval) + lvc * self.wtc.update(rcost) + lvs * self.wts.update(rscor) + lvw * self.wtw.update(rwght)
                            advan = (twght - twght.mean().detach()).detach()
                            lprob = functional.log_softmax(lgits[:, -1, :] - lgits[:, -1, :].max(dim=-1, keepdim=True)[0], dim=-1)
                            ploss = -(advan * torch.gather(lprob, -1, ymat[:, -1:].long()).squeeze(-1)).mean()
                            tloss = (mloss + closs + ploss) / accum
                        try:
                            tloss.backward()
                        except Exception as err:
                            print(f"[_re] backward failed: {err}")
                        if (step + 1) % accum == 0:
                            try:
                                with torch.no_grad():
                                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                                    torch.nn.utils.clip_grad_norm_(self.encod.parameters(), 1.0)
                                    sched.step()
                                    self.mopt.step()
                                    self.mopt.zero()
                                    if verbs:
                                        print(f"[_re] Step {step}, loss={tloss.item() * accum}, grad_norm={gnorm}")
                            except Exception as err:
                                print(f"[_re] optim failed: {err}")
                        break
                    except RuntimeError as err:
                        if "out of memory" in str(err).lower():
                            torch.cuda.empty_cache()
                            self.mopt.zero()
                            self.qopt.zero()
                        elif "timeout" in str(err).lower():
                            sys.exit(1)
                        else:
                            print(f"[_re] Error in loop: {err}")
                            raise err
            if self.rank == 0:
                if verbs:
                    print(f"Ep {epc} done.")
                self.save(f'ckpt_{epc}', epc, sched)
        return self.model


class Dataset(datald.Dataset):
    def __init__(self, count=50):
        self.data = []
        dset = []
        dset.append((b"def f(x):\n if x==0:\n  return 0\n return f(x-1)+1", 1.0, [((idx,), idx) for idx in range(10)]))
        dset.append((b"def f(x):\n if x<=1:\n  return x\n return f(x//2)+1", 0.0, [((idx,), int(math.log2(idx)) + 1 if idx > 1 else idx) for idx in range(1, 10)]))
        dset.append((b"def f(x):\n if x<=1:\n  return x\n return 2*f(x//2)+x", 1.0, [((idx,), idx * int(math.log2(idx)) + idx if idx > 1 else idx) for idx in range(1, 10)]))
        dset.append((b"def f(x):\n if x<=1:\n  return x\n return f(x-1)+x", 2.0, [((idx,), sum(range(idx + 1))) for idx in range(10)]))
        dset.append((b"def f(x):\n if x<=1:\n  return x\n return f(x-1)+f(x-2)", 2.0, [((idx,), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34][idx]) for idx in range(10)]))
        dset.append((b"def f(x):\n return x*x", 0.0, [((idx,), idx * idx) for idx in range(10)]))
        for _ in range(count):
            for sval, cval, tval in dset:
                xmat = torch.tensor(list(sval), dtype=torch.long)
                ymat = torch.tensor(list(sval[1:] + b"\0"), dtype=torch.long)
                cmat = torch.tensor([cval])
                for _ in range(4):
                    self.data.append((xmat, ymat, cmat, tval))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def Loaders(dset, bsze=4):
    cfunc = lambda args: (torch.stack([itm[0] for itm in args]), torch.stack([itm[1] for itm in args]), torch.stack([itm[2] for itm in args]), [itm[3] for itm in args])
    if dist.is_initialized():
        smpl = dsamp(dset, drop_last=True)
        return datald.DataLoader(dset, batch_size=bsze, sampler=smpl, collate_fn=cfunc, num_workers=4, pin_memory=True, prefetch_factor=2), smpl
    return datald.DataLoader(dset, batch_size=bsze, shuffle=True, collate_fn=cfunc, num_workers=4, pin_memory=True), None


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        setseed()
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dset = Dataset()
    dldr, smpl = Loaders(dset, bsze=4)
    train = Trainer()
    modl = train.train(dldr, smpl, epchs=1, accum=4, verbs=True)
    if dist.is_initialized():
        dist.destroy_process_group()