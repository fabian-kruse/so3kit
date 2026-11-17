# CPU-only: index bookkeeping must match the reference implementation in `spherical`.
import pytest
import spherical

from so3kit import WignerDsize, WignerDindex


@pytest.mark.parametrize("ell_max", [0, 1, 4, 9])
def test_dsize_matches_spherical(ell_max):
    assert WignerDsize(0, ell_max, ell_max) == spherical.Wigner(ell_max).Dsize


@pytest.mark.parametrize("ell_max", [1, 4, 7])
def test_dindex_matches_spherical(ell_max):
    w = spherical.Wigner(ell_max)
    for ell in range(ell_max + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                assert WignerDindex(ell, mp, m, 0, ell_max) == w.Dindex(ell, mp, m)
