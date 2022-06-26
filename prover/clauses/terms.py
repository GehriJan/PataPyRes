"""
A simple implementation of first-order terms. We use nested Python
lists in the style of s-expressions as the term data type.

Definition: Let F be a finite set of function symbols and V be an
enumerable set of variable symbols. Let ar:F->N be the arity function
associating a natural number (the "arity") with each function
symbol. The set of all terms over F and V, Terms(F,V) is defined as
follows:
- For all X in V, X in Term(F,V)
- For all f|n in F and t1,..,tn in Term(F,V), f(t1, ..., tn) in
  Term(F,V).
- Term(F,V) is the smallest set with the above two properties.


In the concrete syntax (i.e. the syntax we use to write terms in ASCII
text form), we represent elements of F by identifers starting with a
lower-case letter. The arity is implicitly given by the number of
argument terms in a term. For function symbols with arity 0, we omit
the parenthesis of the empty argument list.

We represent elements of V by identifiers starting with an upper-case
letter or underscore.

A composite term f(t1, ..., tn) is represented by the list
[f lt1, ..., ltn], where lt1, ..., ltn are lists representing the
subterms. See below for exmples:

"X"          -> "X"
"a"          -> ["a"]
"g(a,b)"     -> ["g", ["a"], ["b"]]
"g(X, f(Y))" -> ["g", "X", ["f", "Y"]]

Note in particular that constant terms are lists with one elements,
not plain strings.

Copyright 2010-2019 Stephan Schulz, schulz@eprover.org

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program ; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston,
MA  02111-1307 USA

The original copyright holder can be contacted as

Stephan Schulz
Auf der Altenburg 7
70376 Stuttgart
Germany
Email: schulz@eprover.org
"""

from prover.clauses.signature import Signature
from prover.parser.lexer import Token, Lexer


def termIsVar(t):
    """
    Check if the term is a variable. This assumes that t is a
    well-formed term.
    """
    return type(t) != type([])


def termIsCompound(t):
    """
    Check if the term is a compound term. This assumes that t is a
    well-formed term.
    """
    return not termIsVar(t)


def termFunc(t):
    """
    Return the function symbol of the compound term t.
    """
    assert termIsCompound(t)
    return t[0]


def termArgs(t):
    """
    Return the argument list of the compound term t.
    """
    assert termIsCompound(t)
    return t[1:]


def term2String(t):
    """
    Convert a term t into a string.
    """
    if termIsVar(t):
        return t
    else:
        # We need to handle the case of constants separatly
        if not termArgs(t):
            return termFunc(t)
        else:
            arg_rep = ",".join([term2String(s) for s in termArgs(t)])
            return termFunc(t) + "(" + arg_rep + ")"


def parseTermList(lexer):
    """
    Parse a comma-delimited list of terms.
    """
    res = [parseTerm(lexer)]
    while lexer.TestTok(Token.Comma):
        lexer.AcceptTok(Token.Comma)
        res.append(parseTerm(lexer))
    return res


def parseTerm(lexer):
    """
    Read a complete term from the lexer provided.
    """
    if lexer.TestTok(Token.IdentUpper):
        res = lexer.Next().literal
    else:
        res = []
        lexer.CheckTok([Token.IdentLower, Token.DefFunctor, Token.SQString])
        res.append(lexer.Next().literal)
        if lexer.TestTok(Token.OpenPar):
            # It's a term with proper subterms, so parse them
            lexer.AcceptTok(Token.OpenPar)
            res.extend(parseTermList(lexer))
            lexer.AcceptTok(Token.ClosePar)
    return res


def string2Term(string):
    """
    Convert a string into a term.
    """
    lexer = Lexer(string)
    return parseTerm(lexer)


def termListEqual(l1, l2):
    """
    Compare two lists of terms.
    """
    if len(l1) != len(l2):
        return False
    if not l1:
        # l1 is empty, and so, by the previous test, is l2
        return True
    for i in range(len(l1)):
        if not termEqual(l1[i], l2[i]):
            return False
    return True


def termEqual(t1, t2):
    """
    Compare two terms for syntactic equality.
    """
    if termIsVar(t1):
        return t1 == t2
    elif termIsVar(t2):
        return False
    else:
        if termFunc(t1) != termFunc(t2):
            return False
        return termListEqual(termArgs(t1), termArgs(t2))


def termCopy(t):
    """
    Return a (deep) copy of t. This is the lazy man's way...
    """
    if type(t) == type([]):
        # t is a list, so we copy the elements of the list
        return [termCopy(x) for x in t]
    return t


def termIsGround(t):
    """
    termIsGround(t): Return True if term has no variables, False otherwise
    """
    if termIsVar(t):
        return False
    else:
        for term in termArgs(t):
            if not termIsGround(term):
                return False
        return True


def termCollectVars(t, res=None):
    """
    Insert all variables in t into the set res. For convenience,
    return res. If res is not given, create and return it.
    """
    if res is None:
        res = set()
    if termIsVar(t):
        res.add(t)
    else:
        for s in termArgs(t):
            termCollectVars(s, res)
    return res


def termCollectFuns(t, res=None):
    """
    Insert all function symbols in t into the set res. For
    convenience, return res. If res is not given, create and return
    it.
    """
    if res is None:
        res = set()
    if termIsCompound(t):
        res.add(termFunc(t))
        for s in termArgs(t):
            termCollectFuns(s, res)
    return res


def termCollectSig(t, sig=None):
    """
    Insert all function symbols and their associated arities in t into
    the signature sig. For convenience, return it. If sig is not
    given, create it.
    """
    if sig is None:
        sig = Signature()
    if termIsCompound(t):
        sig.addFun(termFunc(t), len(t) - 1)
        for s in termArgs(t):
            termCollectSig(s, sig)
    return sig


def termWeight(t, fweight, vweight):
    """
    Return the weight of the term, counting fweight for each function
    symbol occurance, vweight for each variable occurance.
    Examples:
      termWeight(f(a,b), 1, 1) = 3
      termWeight(f(a,b), 2, 1) = 6
      termWeight(f(X,Y), 2, 1) = 4
      termWeight(X, 2, 1)      = 1
      termWeight(g(a), 3, 1)   = 6
    """
    if termIsVar(t):
        return vweight
    else:
        res = fweight
        for s in termArgs(t):
            res = res + termWeight(s, fweight, vweight)
        return res


def termocbweight(t, ocb):
    """
    Return weight of term / var from ocb
    """
    if termIsVar(t):
        return ocb.var_weight
    else:
        res = ocb.ocb_funs.get(termFunc(t), 1)
        for s in termArgs(t):
            res = res + termocbweight(s, ocb)
        return res


def subterm(t, pos):
    """
    Return the subterm of t at position pos (or None if pos is not a
    position in term). pos is a list of integers denoting branches,
    e.g.
       subterm(f(a,b), [])        = f(a,b)
       subterm(f(a,g(b)), [0])    = a
       subterm(f(a,g(b)), [1])    = g(b)
       subterm(f(a,g(b)), [1,0])  = b
       subterm(f(a,g(b)), [3,0])  = None
    """
    if not pos:
        return t
    index = pos.pop(0)
    if index >= len(t):
        return None
    return subterm(t[index], pos)


def termIsSubterm(term, test):
    """
    Return if term is subterm of another one
    """
    if term == test:
        return True
    if not termIsCompound(term):
        return False
    for i in range(len(termArgs(term))):
        if termIsSubterm(subterm(term, [i]), test):
            return True

    return False


def countvaroccurrences(term, increment, occurrences=None):
    """
    Returns a dict with the occurrences of each var within the term
    """
    if occurrences is None:
        occurrences = {}
    if termIsVar(term):
        old = occurrences.get(term)
        if old is None:
            occurrences.update({term: 0})
            old = occurrences.get(term)
        occurrences.update({term: old + increment})
    elif not termIsVar(term):
        for pos in range(len(termArgs(term))):
            occurrences = countvaroccurrences(subterm(term, [pos + 1]), increment, occurrences)
    return occurrences
