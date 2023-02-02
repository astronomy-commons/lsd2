#
# Healpix utilities
#

import numpy as np
import healpy as hp

# see the documentation for get_edge() about this variable
_edge_vectors = [
    np.asarray([1, 3]), # NE edge
    np.asarray([1]),    # E corner
    np.asarray([0, 1]), # SE edge
    np.asarray([0]),    # S corner
    np.asarray([0, 2]), # SW edge
    np.asarray([2]),    # W corner
    np.asarray([2, 3]), # NW edge
    np.asarray([3])     # N corner
]

# cache used by get_margin()
_suffixes = {}

def get_edge(dk, pix, edge):
    # Given a pixel pix of at some order, return all
    # pixels order dk _higher_ than pix's order that line
    # pix's edge (or a corner).
    #
    # pix: the pixel ID for which the margin is requested
    # dk: the requested order of edge pixel IDs, relative to order of pix
    # edge: which edge/corner to return (NE edge=0, E corner=1, SE edge = 2, ....)
    #
    # ## Algorithm:
    #
    # If you look at how the NEST indexing scheme works, a pixel at some order is
    # subdivided into four subpixel at every subsequent order in such a way that the south
    # subpixel index equals 4*pix, east is 4*pix+1, west is 4*pix+2, north is 4*pix+3:
    #
    #                   4*pix+3
    #   pix ->     4*pix+2    4*pix+1
    #                    4*pix
    #
    # Further subdivisions split up each of those sub pixels accordingly. For example,
    # the eastern subpixel (4*pix+1) gets divide up into four more:
    #
    #     S=4*(4*pix+1), E=4*(4*pix+1)+1, W=4*(4*pix+1)+2, N=4*(4*pix+1)+3
    #
    #                                                               4*(4*pix+3)+3
    #                   4*pix+3                             4*(4*pix+3)+2   4*(4*pix+3)+1
    #                                               4*(4*pix+2)+3   4*(4*pix+3)   4*(4*pix+1)+3
    #   pix ===>  4*pix+2    4*pix+1  ===>  4*(4*pix+2)+2   4*(4*pix+2)+1   4*(4*pix+1)+2   4*(4*pix+1)+1
    #                                                4*(4*pix+2)    4*(4*pix)+3    4*(4*pix+1)
    #                    4*pix                                4*(4*pix)+2   4*(4*pix)+1
    #                                                                  4*(4*pix)
    # etcetera...
    #
    # We can see that the edge indices follow a pattern. For example, after two
    # subdivisions the south-east edge would consist of pixels:
    #     4*(4*pix), 4*(4*pix)+1
    #     4*(4*pix+1), 4*(4*pix+1)+1
    # with the top coming from subdividing the southern, and bottom the eastern pixel.
    # and so on, recursively, for more subdivisions. Similar patterns are identifiable
    # with other edges.
    #
    # This can be compactly written as:
    #
    #   4*(4*(4*pix + i) + j) + k ....
    #
    # with i, j, k, ... being indices whose choice specifies which edge we get.
    # For example iterating through, i = {0, 1}, j = {0, 1}, k = {0, 1} generates indices
    # for the 8 pixels of the south-east edge for 3 subdivisions. Similarly, for
    # the north-west edge the index values would loop through {2, 3}, etc.
    #
    # This can be expanded as:
    #
    #   4**dk*pix  +  4**(dk-1)*i + 4**(dk-2)*j + 4**(dk-3) * k + ...
    #
    # where dk is the number of subdivisions. E.g., for dk=3, this would equal:
    #
    #   4**3*pix + 4**2*i + 4**1*j + k
    #
    # When written with bit-shift operators, another interpretation becomes clearer:
    #
    #   pix << 6 + i << 4 + j << 2 + k
    #
    # or if we look at the binary representation of the resulting number:
    #
    #   [pix][ii][jj][kk]
    #
    # Where ii, jj, kk are the bits of _each_ of the i,j,k indices. That is, to get the
    # list of subpixels on the edge, we bit-shift the base index pix by 2*dk to the left,
    # and fill in the rest with all possible combinations of ii, jj, kk indices. For example,
    # the northeast edge has index values {2, 3} = {0b10, 0b11}, so for (say) pix=8=0b1000, the
    # northeast edge indices after two subdivisions are equal to:
    #
    #   0b1000 10 10 = 138
    #   0b1000 10 11 = 139
    #   0b1000 11 10 = 148
    #   0b1000 11 11 = 143
    #
    # Note that for a given edge and dk, the suffix of each edge index does not depend on
    # the value of pix (pix is bit-shifted to the right and added). This means it can be
    # precomputed & stored for repeated reuse.
    #
    # ## Implementation:
    #
    # The implementation is based on the binary digit interpretation above. For a requested
    # edge and dk subdivision level, we generate (and cache) the suffixes. Then, for a given
    # pixel pix, the edge pixels are readily computed by left-shifting pix by 2*dk places,
    # and summing (== or-ing) it with the suffix array.
    #

    try:
        a = _suffixes[(edge, dk)]
    except KeyError:
        # generate and cache the suffix:

        # generate all combinations of i,j,k,... suffixes for the requested edge
        # See https://stackoverflow.com/a/35608701
        a = np.array(np.meshgrid(*[_edge_vectors[edge]]*dk)).T.reshape(-1, dk)
        # bit-shift each suffix by the required number of bits
        a <<= np.arange(2*(dk-1),-2,-2)
        # sum them up row-wise, generating the suffix
        a = a.sum(axis=1)
        # cache for further reuse
        _suffixes[(edge, dk)] = a

    # append the 'pix' preffix
    a = (pix << 2*dk) + a
    return a

def get_margin(kk, pix, dk):
    # Returns the list of indices of pixels of order (kk+dk) that
    # border the pixel pix of order kk.
    #
    # Algorithm: given a pixel pix, find all of its neighbors. Then find the
    # edge at level (kk+dk) for each of the neighbors.
    #
    # This is relatively straightforward in the equatorial faces of the Healpix
    # sphere -- e.g., one takes the SW neighbor and requests its NE edge to get
    # the margin on that side of the pixel, then the E corner of the W neighbor,
    # etc...
    #
    # This breaks down on pixels that are at the edge of polar faces. There,
    # the neighbor's sense of direction _rotates_ when switching from face to
    # face. For example at order=2, pixel 5 is bordered by pixel 26 to the
    # northeast (see the Fig3, bottom, https://healpix.jpl.nasa.gov/html/intronode4.htm).
    # But to pixel 5 that is the **northwest** edge (and not southwest, as you'd
    # expect; run hp.get_all_neighbours(4, 26, nest=True) and
    # hp.get_all_neighbours(4, 5, nest=True) to verify these claims.)
    #
    # This is because directions rotate 90deg clockwise when moving from one
    # polar face to another in easterly direction (e.g., from face 0 to face 1).
    # We have to identify when this happens, and change the edges we request
    # for such neighbors. Mathematically, this rotation is equal to adding +2
    # (modulo 8) to the requested edge in get_edge(). I.e., for the
    # pixel 5 example, rather than requesting SE=4 edge of pixel 26,
    # we request SE+2=6=NE edge (see the comments in the definition of _edge_vectors
    # near get_edge())
    #
    # This index shift generalizes to 2*(neighbor_face - pix_face) for the case
    # when _both_ faces are around the pole (either north or south). In the south,
    # the rotation is in the opposite direction (ccw) -- so the sign of the shift
    # changes. The generalized formula also handles the pixels whose one vertex
    # is the pole (where all three neighbors sharing that vertex are on different
    # faces).
    #
    # Implementation: pretty straightforward implementation of the algorithm
    # above.
    #
    nside = hp.order2nside(kk)

    # get all neighboring pixels
    n = hp.get_all_neighbours(nside, pix, nest=True)

    # get the healpix faces IDs of pix and the neighboring pixels
    _, _, f0 = hp.pix2xyf(nside, pix, nest=True)
    _, _, f  = hp.pix2xyf(nside, n, nest=True)

    # indices which tell get_edge() which edge/verted to return
    # for a given pixel. The default order is compatible with the
    # order returned by hp.get_all_neighbours().
    which = np.arange(8)
    if f0 < 4: # northern hemisphere; 90deg cw rotation for every +1 face increment
        mask = (f < 4)
        which[mask] += 2*(f-f0)[mask]
        which %= 8
    elif f0 >= 8: # southern hemisphere; 90deg ccw rotation for every +1 face increment
        mask = (f >= 8)
        which[mask] -= 2*(f-f0)[mask]
        which %= 8

    # get all edges/vertices (making sure we skip -1 entries, for pixels with seven neighbors)
    nb = list(get_edge(dk, px, edge) for edge, px in zip(which, n) if px != -1)
    nb = np.concatenate(nb)
    return nb
