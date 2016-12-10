from builtins import range
try:
    import numpypy as numpy
except ImportError:
    import numpy
from abc import ABCMeta, abstractmethod

from alignment_dhm.sequence import *


# Scoring ---------------------------------------------------------------------

class Scoring(object, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, firstElement, secondElement):
        return 0


class SimpleScoring(Scoring):

    def __init__(self, matchScore, mismatchScore):
        self.matchScore = matchScore
        self.mismatchScore = mismatchScore

    def __call__(self, firstElement, secondElement):
        if firstElement == secondElement:
            return self.matchScore
        else:
            return self.mismatchScore


# Alignment -------------------------------------------------------------------

class SequenceAlignment(object):

    def __init__(self, first, second, gap=GAP_CODE, other=None, firstOffset=None, secondOffset=None):
        self.first = first
        self.firstOffset = firstOffset
        self.second = second
        self.secondOffset = secondOffset
        self.gap = gap
        if other is None:
            self.scores = [0] * len(first)
            self.score = 0
            self.identicalCount = 0
            self.similarCount = 0
            self.gapCount = 0
        else:
            self.scores = list(other.scores)
            self.score = other.score
            self.identicalCount = other.identicalCount
            self.similarCount = other.similarCount
            self.gapCount = other.gapCount

    def push(self, firstElement, secondElement, score=0):
        self.first.push(firstElement)
        self.second.push(secondElement)
        self.scores.append(score)
        self.score += score
        if firstElement == secondElement:
            self.identicalCount += 1
        if score > 0:
            self.similarCount += 1
        if firstElement == self.gap or secondElement == self.gap:
            self.gapCount += 1
        pass

    def pop(self):
        firstElement = self.first.pop()
        secondElement = self.second.pop()
        score = self.scores.pop()
        self.score -= score
        if firstElement == secondElement:
            self.identicalCount -= 1
        if score > 0:
            self.similarCount -= 1
        if firstElement == self.gap or secondElement == self.gap:
            self.gapCount -= 1
        return firstElement, secondElement

    def key(self):
        return self.first.key(), self.second.key()

    def reversed(self):
        first = self.first.reversed()
        second = self.second.reversed()
        firstOffset = self.firstOffset
        secondOffset = self.secondOffset
        return type(self)(first, second, self.gap, self, \
              firstOffset=firstOffset, secondOffset=secondOffset)

    def percentIdentity(self):
        try:
            return float(self.identicalCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def percentSimilarity(self):
        try:
            return float(self.similarCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def percentGap(self):
        try:
            return float(self.gapCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def quality(self):
        return self.score, \
            self.percentIdentity(), \
            self.percentSimilarity(), \
            -self.percentGap()

    def __len__(self):
        assert len(self.first) == len(self.second)
        return len(self.first)

    def __getitem__(self, item):
        return self.first[item], self.second[item]

    def __repr__(self):
        return repr((self.first, self.second))

    def __str__(self):
        first = [str(e) for e in self.first.elements]
        second = [str(e) for e in self.second.elements]
        for i in range(len(first)):
            n = max(len(first[i]), len(second[i]))
            format = '%-' + str(n) + 's'
            first[i] = format % first[i]
            second[i] = format % second[i]
        return '%s\n%s' % (' '.join(first), ' '.join(second))

    def __unicode__(self):
        first = [str(e) for e in self.first.elements]
        second = [str(e) for e in self.second.elements]
        for i in range(len(first)):
            n = max(len(first[i]), len(second[i]))
            format = '%-' + str(n) + 's'
            first[i] = format % first[i]
            second[i] = format % second[i]
        return '%s\n%s' % (' '.join(first), ' '.join(second))


# Aligner ---------------------------------------------------------------------

class SequenceAligner(object, metaclass=ABCMeta):
    def __init__(self, scoring, gapScore):
        self.scoring = scoring
        self.gapScore = gapScore

    def align(self, first, second, backtrace=False):
        #print("blah")
        f = self.computeAlignmentMatrix(first, second)
        #print(f)
        #print("figs")
        score = self.bestScore(f)
        if backtrace:
            #print("asdf")
            if self.backtraceIter:
                #print("as1f")
                alignments = self.backtraceIter(first, second, f)
            else:
                #print("as2f")
                alignments = self.backtrace(first, second, f)
                #print("as3f")
            return score, alignments
        else:
            return score

    def emptyAlignment(self, first, second):
        # Pre-allocate sequences.
        return SequenceAlignment(
            EncodedSequence(len(first) + len(second), id=first.id),
            EncodedSequence(len(first) + len(second), id=second.id),
        )

    @abstractmethod
    def computeAlignmentMatrix(self, first, second):
        return numpy.zeros(0, int)

    @abstractmethod
    def bestScore(self, f):
        return 0

    @abstractmethod
    def backtrace(self, first, second, f):
        return list()


class GlobalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore):
        super(GlobalSequenceAligner, self).__init__(scoring, gapScore)

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = numpy.zeros((m, n), int)
        for i in range(1, m):
            for j in range(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on first sequence.
                if i == m - 1:
                    ga = f[i, j - 1]
                else:
                    ga = f[i, j - 1] + self.gapScore

                # Gap on second sequence.
                if j == n - 1:
                    gb = f[i - 1, j]
                else:
                    gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(ab, max(ga, gb))
        return f

    def bestScore(self, f):
        return f[-1, -1]

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        self.backtraceFrom(first, second, f, m - 1, n - 1,
                           alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if i == 0 or j == 0:
            alignment.firstOffset=i
            alignment.secondOffset=j
            alignments.append(alignment.reversed())
        else:
            m, n = f.shape
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                alignment.push(a, b, c - p)
                self.backtraceFrom(first, second, f, i - 1, j - 1,
                                   alignments, alignment)
                alignment.pop()
            else:
                if i == m - 1:
                    if c == y:
                        self.backtraceFrom(first, second, f, i, j - 1,
                                           alignments, alignment)
                elif c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
                if j == n - 1:
                    if c == x:
                        self.backtraceFrom(first, second, f, i - 1, j,
                                           alignments, alignment)
                elif c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()


    #TODO only returning 1 alignment currently to make our life easier
    def backtraceIter(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        i = m - 1
        j = n - 1

        while i != 0 and j != 0 :
            print ("[i,j]=[", i, ",", j, "]")
            m, n = f.shape
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                print("a")
                alignment.push(a, b, c - p)
                i -= 1
                j -= 1
                continue
            else:
                print("b")
                if i == m - 1:
                    print("c")
                    if c == y:
                        print("d")
                        j -= 1
                        continue
                elif c == y + self.gapScore:
                    print("e")
                    alignment.push(alignment.gap, b, c - y)
                    j -= 1
                    continue
                if j == n - 1:
                    print("f")
                    if c == x:
                        print("g")
                        i -= 1
                        continue
                elif c == x + self.gapScore:
                    print("h")
                    alignment.push(a, alignment.gap, c - x)
                    i -= 1
                    continue

        return [alignment.reversed()]



class StrictGlobalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore):
        super(StrictGlobalSequenceAligner, self).__init__(scoring, gapScore)

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = numpy.zeros((m, n), int)
        for i in range(1, m):
            f[i, 0] = f[i - 1, 0] + self.gapScore
        for j in range(1, n):
            f[0, j] = f[0, j - 1] + self.gapScore
        for i in range(1, m):
            for j in range(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on first sequence.
                ga = f[i, j - 1] + self.gapScore

                # Gap on second sequence.
                gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(ab, max(ga, gb))
        return f

    def bestScore(self, f):
        return f[-1, -1]

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        self.backtraceFrom(first, second, f, m - 1, n - 1,
                           alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if i == 0 and j == 0:
            alignment.firstOffset=i
            alignment.secondOffset=j
            alignments.append(alignment.reversed())
        else:
            c = f[i, j]
            if i != 0:
                x = f[i - 1, j]
                a = first[i - 1]
                if c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()
                    return
            if j != 0:
                y = f[i, j - 1]
                b = second[j - 1]
                if c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
            if i != 0 and j != 0:
                p = f[i - 1, j - 1]
                #noinspection PyUnboundLocalVariable
                if c == p + self.scoring(a, b):
                    #noinspection PyUnboundLocalVariable
                    alignment.push(a, b, c - p)
                    self.backtraceFrom(first, second, f, i - 1, j - 1,
                                       alignments, alignment)
                    alignment.pop()


class LocalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore, minScore=None):
        super(LocalSequenceAligner, self).__init__(scoring, gapScore)
        self.minScore = minScore

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = numpy.zeros((m, n), int)
        for i in range(1, m):
            for j in range(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on sequenceA.
                ga = f[i, j - 1] + self.gapScore

                # Gap on sequenceB.
                gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(0, max(ab, max(ga, gb)))
        return f

    def bestScore(self, f):
        return f.max()

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        if self.minScore is None:
            minScore = self.bestScore(f)
        else:
            minScore = self.minScore
        for i in range(m):
            for j in range(n):
                if f[i, j] >= minScore:
                    self.backtraceFrom(first, second, f, i, j,
                                       alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if f[i, j] == 0:
            alignment.firstOffset=i
            alignment.secondOffset=j
            alignments.append(alignment.reversed())
        else:
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                alignment.push(a, b, c - p)
                self.backtraceFrom(first, second, f, i - 1, j - 1,
                                   alignments, alignment)
                alignment.pop()
            else:
                if c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
                if c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()
