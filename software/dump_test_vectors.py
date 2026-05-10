"""Generate test vectors for the systolic array's SystemVerilog testbench.

For step 4.5 (random tile verification) this writes N_TESTS independent
8x8 tiles, concatenated into a single hex file per signal. The SV
testbench reads the whole file and loops over test cases.

Files written to tb/data/:
  a_tile.hex:      N_TESTS * 64 int16 values (test 0 row-major, then test 1, ...)
  b_tile.hex:      N_TESTS * 64 int16 values
  c_expected.hex:  N_TESTS * 64 int32 values  (= A_tile @ B_tile per case)

Each value is zero-padded hex in two's complement (4 chars for int16,
8 for int32). $readmemh in SV reads this format directly into an
unpacked array of length N_TESTS*64.

NOTE: N_TESTS in this file MUST match the parameter of the same name
in tb/systolic_array_tb.sv. If you change one, change the other.
"""
import os
import numpy as np

N_TESTS = 100   # number of independent random tiles
N       = 8     # tile dimension (8x8 systolic array)


def to_hex_twos_complement(val, bits):
    """Convert a (possibly signed) integer to its zero-padded hex string in
    two's complement. Width = bits / 4 hex chars."""
    mask = (1 << bits) - 1
    n_hex = bits // 4
    return f"{int(val) & mask:0{n_hex}x}"


def dump_tiles_for_sv(filename, tiles, width_bits):
    """Dump a stack of tiles (shape [N_TESTS, N, N]) to a hex file,
    one value per line, in (test, row, col) order."""
    flat = tiles.flatten()
    with open(filename, 'w') as f:
        for v in flat:
            f.write(to_hex_twos_complement(v, width_bits) + '\n')


def main():
    np.random.seed(0)

    # Bounded random ints. Keep values small enough that 8 multiplications of
    # two int16 values stay well within int32 range (worst case ~8*50*50 = 20000
    # per term * 8 terms = 160000, far below 2^31).
    A_tiles = np.random.randint(-50, 51, size=(N_TESTS, N, N), dtype=np.int16)
    B_tiles = np.random.randint(-50, 51, size=(N_TESTS, N, N), dtype=np.int16)

    # Batched matmul: NumPy treats the leading axis as a batch dimension.
    # Result has shape (N_TESTS, N, N), int32 to match the array's accumulator.
    C_expected = (A_tiles.astype(np.int32) @ B_tiles.astype(np.int32)).astype(np.int32)

    # Path: this file is at software/, target is tb/data/
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'tb', 'data')
    os.makedirs(out_dir, exist_ok=True)

    dump_tiles_for_sv(os.path.join(out_dir, 'a_tile.hex'),     A_tiles,    width_bits=16)
    dump_tiles_for_sv(os.path.join(out_dir, 'b_tile.hex'),     B_tiles,    width_bits=16)
    dump_tiles_for_sv(os.path.join(out_dir, 'c_expected.hex'), C_expected, width_bits=32)

    print(f"Wrote {N_TESTS} test cases ({N_TESTS * N * N} values per file) to {out_dir}/")
    print(f"  Test 0 A[0,0] = {A_tiles[0, 0, 0]:5d}     (hex {to_hex_twos_complement(A_tiles[0, 0, 0], 16)})")
    print(f"  Test 0 B[0,0] = {B_tiles[0, 0, 0]:5d}     (hex {to_hex_twos_complement(B_tiles[0, 0, 0], 16)})")
    print(f"  Test 0 C[0,0] = {C_expected[0, 0, 0]:6d}    (hex {to_hex_twos_complement(C_expected[0, 0, 0], 32)})")
    print(f"  Test {N_TESTS-1} C[7,7] = {C_expected[N_TESTS-1, 7, 7]:6d}    (hex {to_hex_twos_complement(C_expected[N_TESTS-1, 7, 7], 32)})")


if __name__ == '__main__':
    main()
