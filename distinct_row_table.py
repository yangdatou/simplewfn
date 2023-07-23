from __future__ import annotations
from typing import Callable, Generator, Optional
from functools import reduce
from itertools import accumulate
import numpy as np


class DistinctRowTable:
    def __init__(self, a: int = 0, b: int = 0, c: int = 0,
                       ncor: int=0, nvir: int=0,
                       nelec: Optional[int]=None, spin: int=0,
                       norb: Optional[int]=None, ipg: int=0, orb_sym: list[int]=[],
                       ci_order: Optional[int]=None, 
                       is_su2: bool=True) -> None:
        """
        Initialize the DistinctRowTable object with specific properties

        Parameters
        ----------
        a : int, optional
            The number of alpha electrons in the core
        b : int, optional
            The number of alpha electrons in the virtual space
        c : int, optional
            The number of beta electrons in the core
        norb : int, optional
            The number of sites
        nelec : int, optional
            The number of electrons
        spin : int, optional
            The spin of the system
        ipg : int, optional
            The parity of the system
        orb_sym : list[int], optional
            The orbital symmetry of the system
        ci_order : int, optional
            The order of the CI
        n_core : int, optional
            The number of core orbitals
        n_virt : int, optional
            The number of virtual orbitals
        is_su2 : bool, optional
            Whether the system is SU2 symmetric
        """

        self.ci_order: Optional[int] = ci_order
        self.ncor: int = ncor
        self.nvir: int = nvir

        if norb is not None and nelec is not None:
            # If norb and nelec are provided, calculate the values for a, b, c
            # based on the provided values and the equation (nelec - abs(spin)) // 2, spin, norb - (nelec + abs(spin)) // 2
            a, b, c = (nelec - abs(spin)) // 2, spin, norb - (nelec + abs(spin)) // 2

        self.abc: np.ndarray = np.array([(a, b, c, ipg)], dtype=int)
        self.jds: np.ndarray = np.array([], dtype=int)
        self.xs: np.ndarray = np.array([(0, 0, 0, 0, 1)], dtype=int)
        self.orb_sym: np.ndarray = np.array([0] * self.norb if len(orb_sym) == 0 else orb_sym, dtype=int)
        self.is_su2 = is_su2
        self.init(is_su2=is_su2)

    @property
    def norb(self) -> int:
        # Property getter for the norb attribute
        # Returns the sum of absolute values of the first three elements of abc array
        return np.sum(np.abs(self.abc[0][:3]))

    @property
    def nelec(self) -> int:
        # Property getter for the nelec attribute
        # Returns the value calculated using the formula: abc[0][0] * 2 + abs(abc[0][1])
        return self.abc[0][0] * 2 + np.abs(self.abc[0][1])

    def init(self, is_su2: bool=True) -> None:
        ci_order = self.nelec if self.ci_order is None else self.ci_order
        make_abc = lambda a, b, c, d: [(a, b, c - 1), (a, b - 1, c) if b > 0 else (a - 1, b - 1, c - 1),
                (a, b + 1, c) if b < 0 else (a - 1, b + 1, c - 1), (a - 1, b, c)][d]
        allow_abc = lambda a, b, c, d: [c, b if is_su2 else (1 if b > 0 else a * c),
                a * c if is_su2 else (1 if b < 0 else a * c), a][d] != 0
        make_pg = lambda g, gk, d: ([g, gk ^ g, gk ^ g, g][d], )
        allow_pg = lambda k, g, gk, d: k != 0 or make_pg(g, gk, d)[0] == 0
        allow_virt = lambda a, b, d, k: k < self.norb - self.nvir or \
            a + a + abs(b) - (d + 1) // 2 >= self.nelec - ci_order
        allow_core = lambda a, b, d, k: k >= self.ncor or a + a + abs(b) - (d + 1) // 2 >= k * 2 - ci_order
        labc = reduce(lambda abcs, k: abcs + [sorted(set(make_abc(a, b, c, d) + make_pg(g, self.orb_sym[k], d)
            for a, b, c, g in abcs[-1] for d in range(4) if allow_abc(a, b, c, d) and allow_pg(k, g, self.orb_sym[k], d)
            and allow_core(a, b, d, k) and allow_virt(a, b, d, k)))[::-1]], range(self.norb)[::-1], [[tuple(self.abc[0])]])
        pabc = reduce(lambda abcs, kabc: [[(a, b, c, g) for a, b, c, g in kabc[1] if any(make_abc(a, b, c, d) +
            make_pg(g, self.orb_sym[kabc[0]], d) in abcs[0] for d in range(4))]] + abcs,
            zip(range(self.norb), labc[:-1][::-1]), labc[-1:])
        iabc = reduce(lambda x, y: x + y, pabc)
        jds = reduce(lambda x, y: x + y, reduce(lambda jds, kabc: jds + [[tuple(iabc.index(new_abc) if new_abc in
            iabc else 0 for d in range(4) for new_abc in [make_abc(a, b, c, d) + make_pg(g, self.orb_sym[kabc[0]], d)
            if kabc[0] >= -1 else ()]) for a, b, c, g in kabc[1]]], zip(range(-1, self.norb)[::-1], pabc), [[]]))
        xs = reduce(lambda xs, jd: [(0, ) + tuple(accumulate(xs[jd[d] - len(iabc) + len(xs)][-1]
            if jd[d] != 0 else 0 for d in range(4)))] + xs, jds[::-1][1:], [tuple(self.xs[-1])])
        self.abc, self.jds, self.xs = np.array(iabc), np.array(jds), np.array(xs)

    def __xor__(self, ci_order: Optional[int]) -> DistinctRowTable:
        return DistinctRowTable(*self.abc[0][:3], ipg=self.abc[0][3], orb_sym=list(self.orb_sym),
            ci_order=ci_order, ncor=self.ncor, nvir=self.nvir, is_su2=self.is_su2)

    def __repr__(self) -> str:
        fmt = "%4s%6s" + "%4s" * 4 + "%6s" * 4 + "%12s" * 5 + "\n"
        header = fmt % tuple("J K A B C PG JD0 JD1 JD2 JD3 X0 X1 X2 X3 X4".split())
        ks: list[str | int] = reduce(lambda g, k: (g[0] + [""], g[1])
            if g[1] == k else (g[0] + [k], k), [np.sum(np.abs(abc[:3])) for abc in self.abc], ([], -1))[0]
        return (header + "".join(fmt % (i + 1, k, *abc, *map(lambda jd: "" if jd == 0 else jd + 1, jd), *x)
            for i, (k, abc, jd, x) in enumerate(zip(ks, self.abc, self.jds, self.xs))))

    def __len__(self) -> int:
        return self.xs[0][-1]

    def __getitem__(self, i: int) -> str:
        get_d: Callable[[int, int], int] = lambda i, p: len([x for x in self.xs[p][1:] if i >= x])
        dvs: list[int] = reduce(lambda ipd, _: [(ipd[0] - self.xs[ipd[1]][d], self.jds[ipd[1]][d],
            [d] + ipd[2]) for d in [get_d(*ipd[:2])]][0], range(self.norb), (i, 0, []))[-1]
        return "".join(["0+-2"[d] for d in dvs])

    def __iter__(self) -> Generator[str, None, None]:
        return (self[i] for i in range(len(self)))

    def index(self, ds: str) -> int:
        return reduce(lambda rx, d: (-1, -1) if rx[1] == -1 else
            (rx[0] + self.xs[rx[1]][d], self.jds[rx[1]][d] or -1),
            ["00+a-b22".index(d) // 2 for d in ds[::-1]], (0, 0))[0]

    def __rshift__(self, other: DistinctRowTable) -> np.ndarray:
        pbr, pb, pkr = [np.array([0], dtype=int) for _ in range(3)]
        assert self.norb == other.norb
        d = np.array([0, 1, 2, 3], dtype=int)
        for _ in range(self.norb):
            new_pbr = self.jds[pbr][:, d].reshape(-1)
            new_pkr = other.jds[pkr][:, d].reshape(-1)
            new_pb = (pb[:, None] + self.xs[pbr][:, d]).reshape(-1)
            mask = (new_pbr != 0) & (new_pkr != 0)
            pbr, pb, pkr = new_pbr[mask], new_pb[mask], new_pkr[mask]
        return pb